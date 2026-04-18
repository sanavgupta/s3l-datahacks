"""
CalCOFI ETL pipeline for physical oceanography and biological data fusion.

This module ingests multiple historical CalCOFI data sources, normalizes
inconsistent biological records, enriches them with available temperature and
salinity metadata, and exports normalized data for dashboard and database use.
"""

import os
import re
import json
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    fish_egg_legacy_path: Path = Path('fish_egg_1')
    fish_egg_salinity_path: Path = Path('fish_egg_2_salinty_included.csv')
    zooplankton_path: Path = Path('zooplankton_1.csv')
    output_sqlite: Path = Path('marine_observations.db')
    output_json: Path = Path('dashboard_data.json')
    llm_enabled: bool = False
    llm_model: str = 'gpt-4'
    llm_api_key: Optional[str] = None


class DataExtractor:
    """Extracts raw files and normalizes header structure."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def read_source(self, source_path: Path) -> pd.DataFrame:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing data source: {source_path}")

        logger.info(f"Reading source file: {source_path}")
        raw = pd.read_csv(source_path, header=None, nrows=2, dtype=str)
        if raw.shape[0] == 2 and self._has_units_row(raw.iloc[1]):
            df = pd.read_csv(source_path, skiprows=[1])
        else:
            df = pd.read_csv(source_path)

        df = self._standardize_columns(df)
        return df

    @staticmethod
    def _has_units_row(row: pd.Series) -> bool:
        text = ' '.join(str(x).lower() for x in row.tolist() if pd.notna(x))
        units_keywords = ['utc', 'degrees', 'celsius', 'salinity', 'm^3', 'knots', 'count']
        count = sum(1 for token in units_keywords if token in text)
        return count >= 2

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        clean_columns = []
        for col in df.columns:
            if isinstance(col, str):
                norm = col.strip().lower()
                norm = re.sub(r'[ /\-]+', '_', norm)
                norm = re.sub(r'[^a-z0-9_]', '', norm)
                clean_columns.append(norm)
            else:
                clean_columns.append(str(col))
        df.columns = clean_columns
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        result = {
            'fish_egg_legacy': self.read_source(self.config.fish_egg_legacy_path),
            'fish_egg_salinity': self.read_source(self.config.fish_egg_salinity_path),
            'zooplankton': self.read_source(self.config.zooplankton_path),
        }
        return result


class LLMExtractor:
    """Optional LLM layer for normalizing messy biological labels."""

    def __init__(self, config: ETLConfig):
        self.enabled = config.llm_enabled and config.llm_api_key is not None
        self.model = config.llm_model
        self.api_key = config.llm_api_key
        self.client = None

        if self.enabled:
            try:
                import openai
                self.client = openai
                self.client.api_key = self.api_key
                logger.info("LLM extraction enabled")
            except Exception as exc:
                logger.warning("LLM library not available; falling back to direct normalization: %s", exc)
                self.enabled = False
        else:
            logger.info("LLM extraction disabled")

    def parse_species_label(self, text: str) -> Dict[str, Optional[str]]:
        if not self.enabled or not text or pd.isna(text):
            return {'scientific_name': None, 'common_name': None, 'confidence': 0.0}

        prompt = (
            f"Parse this messy biological label into a normalized marine species name. "
            f"Return JSON with scientific_name, common_name, confidence.\n\nLabel: {text}\n"
        )

        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a marine biology data normalization assistant.'},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            payload = response.choices[0].message['content']
            parsed = json.loads(payload)
            return {
                'scientific_name': parsed.get('scientific_name'),
                'common_name': parsed.get('common_name'),
                'confidence': float(parsed.get('confidence', 0.0)),
            }
        except Exception as exc:
            logger.warning("LLM parse failed for '%s': %s", text, exc)
            return {'scientific_name': None, 'common_name': None, 'confidence': 0.0}


class DataNormalizer:
    """Applies dataset-specific normalization and feature engineering."""

    def __init__(self, config: ETLConfig, llm_extractor: Optional[LLMExtractor] = None):
        self.config = config
        self.llm_extractor = llm_extractor

    def normalize(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        frames.append(self.normalize_fish_egg_legacy(sources['fish_egg_legacy']))
        frames.append(self.normalize_fish_egg_salinity(sources['fish_egg_salinity']))
        frames.append(self.normalize_zooplankton(sources['zooplankton']))

        df = pd.concat(frames, ignore_index=True, sort=False)
        df = self._normalize_common_fields(df)
        df = self._populate_spatiotemporal(df)
        df = self._add_quality_flags(df)
        return df

    def normalize_fish_egg_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing fish egg legacy larval records")
        df = df.copy()
        df['source_file'] = 'fish_egg_1'
        df['biological_type'] = 'larval_fish'
        df['species_group'] = df['scientific_name'].fillna('unknown')
        df['common_name_normalized'] = df['common_name'].fillna('unknown')
        df['volume_sampled_m3'] = pd.to_numeric(df.get('volume_sampled', np.nan), errors='coerce')
        df['larvae_count'] = pd.to_numeric(df.get('larvae_count', np.nan), errors='coerce')
        df['density_per_m3'] = df['larvae_count'] / df['volume_sampled_m3']
        df['density_per_m3'] = df['density_per_m3'].replace([np.inf, -np.inf], np.nan)
        return df[
            [
                'source_file', 'cruise', 'ship', 'ship_code', 'time', 'latitude', 'longitude',
                'line', 'station', 'biological_type', 'species_group', 'scientific_name',
                'common_name', 'itis_tsn', 'calcofi_species_code', 'larvae_count',
                'larvae_10m2', 'larvae_100m3', 'volume_sampled_m3', 'density_per_m3',
            ]
        ]

    def normalize_fish_egg_salinity(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing fish egg salinity records")
        df = df.copy()
        df['source_file'] = 'fish_egg_2_salinty_included'
        df['biological_type'] = 'fish_eggs'
        egg_columns = [
            'sardine_eggs', 'anchovy_eggs', 'jack_mackerel_eggs',
            'hake_eggs', 'squid_eggs', 'other_fish_eggs'
        ]
        df[egg_columns] = df[egg_columns].apply(pd.to_numeric, errors='coerce')
        df['start_temperature'] = pd.to_numeric(df.get('start_temperature', np.nan), errors='coerce')
        df['stop_temperature'] = pd.to_numeric(df.get('stop_temperature', np.nan), errors='coerce')
        df['start_salinity'] = pd.to_numeric(df.get('start_salinity', np.nan), errors='coerce')
        df['stop_salinity'] = pd.to_numeric(df.get('stop_salinity', np.nan), errors='coerce')
        df['mean_temperature'] = df[['start_temperature', 'stop_temperature']].mean(axis=1)
        df['mean_salinity'] = df[['start_salinity', 'stop_salinity']].mean(axis=1)

        df['sample_minutes'] = (
            pd.to_datetime(df['stop_time'], utc=True) - pd.to_datetime(df['time'], utc=True)
        ).dt.total_seconds() / 60.0
        df['pump_speed_m3_per_min'] = pd.to_numeric(df.get('start_pump_speed', np.nan), errors='coerce')
        df['volume_sampled_m3'] = df['pump_speed_m3_per_min'] * df['sample_minutes']

        long = df.melt(
            id_vars=[
                'source_file', 'cruise', 'ship', 'ship_code', 'sample_number', 'time',
                'latitude', 'longitude', 'start_temperature', 'stop_temperature',
                'mean_temperature', 'start_salinity', 'stop_salinity', 'mean_salinity',
                'start_wind_speed', 'stop_wind_speed', 'sample_minutes', 'pump_speed_m3_per_min',
                'volume_sampled_m3'
            ],
            value_vars=egg_columns,
            var_name='species_group',
            value_name='egg_count'
        )
        long = long[long['egg_count'].notna() & (long['egg_count'] > 0)].copy()
        long['biological_type'] = 'fish_eggs'
        long['common_name'] = (
            long['species_group']
            .str.replace('_eggs', '')
            .str.replace('_', ' ')
            .str.title()
        )
        long['scientific_name'] = long['common_name']
        long['density_per_m3'] = long['egg_count'] / long['volume_sampled_m3']
        long['density_per_m3'] = long['density_per_m3'].replace([np.inf, -np.inf], np.nan)
        return long

    def normalize_zooplankton(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing zooplankton records")
        df = df.copy()
        df['source_file'] = 'zooplankton_1'
        df['biological_type'] = 'zooplankton'
        df['small_plankton'] = pd.to_numeric(df.get('small_plankton', np.nan), errors='coerce')
        df['total_plankton'] = pd.to_numeric(df.get('total_plankton', np.nan), errors='coerce')

        long = df.melt(
            id_vars=['source_file', 'cruise', 'ship', 'ship_code', 'order_occupied',
                     'tow_type', 'tow_number', 'net_location', 'time', 'latitude', 'longitude',
                     'line', 'station', 'volume_sampled'],
            value_vars=['small_plankton', 'total_plankton'],
            var_name='species_group',
            value_name='plankton_density_ml_per_1000m3'
        )
        long = long[long['plankton_density_ml_per_1000m3'].notna()].copy()
        long['scientific_name'] = long['species_group'].str.replace('_', ' ').str.title()
        long['common_name'] = long['scientific_name']
        long['density_per_m3'] = long['plankton_density_ml_per_1000m3'] / 1000.0
        return long

    def _normalize_common_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing common fields across all datasets")
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        for col in ['latitude', 'longitude', 'line', 'station', 'volume_sampled_m3', 'volume_sampled']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if self.llm_extractor is not None:
            df = self._apply_llm_normalization(df)

        return df

    def _apply_llm_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.llm_extractor.enabled:
            return df

        if 'scientific_name' not in df.columns or 'common_name' not in df.columns:
            return df

        parsed = df.apply(self._llm_normalize_row, axis=1, result_type='expand')
        parsed.columns = ['llm_scientific_name', 'llm_common_name', 'llm_confidence']
        df = pd.concat([df, parsed], axis=1)
        df['scientific_name'] = df['scientific_name'].fillna(df['llm_scientific_name'])
        df['common_name'] = df['common_name'].fillna(df['llm_common_name'])
        return df

    def _llm_normalize_row(self, row: pd.Series) -> List[Optional[str]]:
        original_label = row.get('scientific_name') or row.get('common_name')
        parsed = self.llm_extractor.parse_species_label(str(original_label))
        return [parsed['scientific_name'], parsed['common_name'], parsed['confidence']]

    @staticmethod
    def _populate_spatiotemporal(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
        df['spatial_cell'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)
        return df

    @staticmethod
    def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['quality_flag'] = 'good'
        missing = df[['time', 'latitude', 'longitude']].isna().any(axis=1)
        df.loc[missing, 'quality_flag'] = 'incomplete'
        if 'density_per_m3' in df.columns:
            q99 = df['density_per_m3'].quantile(0.99)
            df.loc[df['density_per_m3'] > q99, 'quality_flag'] = 'outlier'
        return df


class DataLoader:
    """Loads normalized data into SQLite and exports JSON dashboards."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def save_to_sqlite(self, df: pd.DataFrame, table_name: str = 'marine_observations') -> None:
        logger.info("Saving normalized data to SQLite: %s", self.config.output_sqlite)
        self.config.output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.output_sqlite) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        logger.info("Saved %d rows into table %s", len(df), table_name)

    def export_dashboard_json(self, df: pd.DataFrame) -> None:
        logger.info("Exporting dashboard JSON to %s", self.config.output_json)
        self.config.output_json.parent.mkdir(parents=True, exist_ok=True)
        summary = self._build_dashboard_payload(df)
        with open(self.config.output_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, default=str, indent=2)
        logger.info("Exported dashboard data with %d groups", len(summary))

    @staticmethod
    def _build_dashboard_payload(df: pd.DataFrame) -> List[Dict]:
        df = df.copy()
        summary = (
            df.groupby(['year', 'month', 'spatial_cell', 'biological_type'])
              .agg(
                  latitude=('latitude', 'mean'),
                  longitude=('longitude', 'mean'),
                  records=('time', 'count'),
                  mean_density=('density_per_m3', 'mean')
              )
              .reset_index()
              .sort_values(['year', 'month', 'spatial_cell'])
        )
        return summary.to_dict(orient='records')


class ETLPipeline:
    """Orchestrates extraction, transformation, and load."""

    def __init__(self, config: ETLConfig):
        self.config = config
        self.extractor = DataExtractor(config)
        self.llm_extractor = LLMExtractor(config)
        self.normalizer = DataNormalizer(config, self.llm_extractor)
        self.loader = DataLoader(config)

    def run(self) -> Dict[str, int]:
        logger.info("Starting ETL pipeline")
        raw = self.extractor.load_all()

        df_norm = self.normalizer.normalize(raw)
        self.loader.save_to_sqlite(df_norm)
        self.loader.export_dashboard_json(df_norm)

        summary = {
            'fish_egg_legacy': len(raw['fish_egg_legacy']),
            'fish_egg_salinity': len(raw['fish_egg_salinity']),
            'zooplankton': len(raw['zooplankton']),
            'normalized_rows': len(df_norm),
        }
        logger.info("ETL pipeline complete")
        return summary


def summary_to_console(summary: Dict[str, int]) -> None:
    print('\n=== ETL SUMMARY ===')
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    print('===================\n')


def main() -> None:
    config = ETLConfig(
        llm_enabled=os.getenv('CALCOFI_LLM_ENABLED', 'false').lower() == 'true',
        llm_api_key=os.getenv('OPENAI_API_KEY')
    )
    pipeline = ETLPipeline(config)
    summary = pipeline.run()
    summary_to_console(summary)


if __name__ == '__main__':
    main()
