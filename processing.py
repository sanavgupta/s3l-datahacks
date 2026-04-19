import os
import re
import json
import sqlite3
import logging
from dataclasses import dataclass
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
    fish_egg_path: Path = Path('fish2.csv') 
    zooplankton_path: Path = Path('zooplankton.csv') 
    physical_oceanography_path: Path = Path('calcofi_physical_merged.csv')
    
    output_sqlite: Path = Path('marine_observations.db')
    output_json: Path = Path('dashboard_data.json')
    llm_enabled: bool = True 
    llm_model: str = 'gemini-2.0-flash'
    llm_api_key: Optional[str] = os.getenv('GOOGLE_API_KEY')

class DataExtractor:
    def __init__(self, config: ETLConfig):
        self.config = config

    def read_source(self, source_path: Path) -> Optional[pd.DataFrame]:
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return None
        
        logger.info(f"Reading {source_path}")
        # CalCOFI files often have a units row; this detects and skips it
        raw = pd.read_csv(source_path, header=None, nrows=2, dtype=str, encoding='latin1')
        skip = [1] if raw.shape[0] == 2 and any('degrees' in str(x).lower() for x in raw.iloc[1]) else []
        
        df = pd.read_csv(source_path, skiprows=skip, encoding='latin1')
        return self._standardize_columns(df)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'[^a-z0-9_]', '', col.strip().lower().replace(' ', '_')) for col in df.columns]
        return df

class LLMExtractor:
    """Zero-Credit Hybrid Extractor: Uses a local cache for known CalCOFI labels."""
    def __init__(self, config: ETLConfig):
        self.enabled = config.llm_enabled and config.llm_api_key is not None
        self.api_key = config.llm_api_key
        self.model_name = config.llm_model
        self.client = None
        
        # LOCAL CACHE: We've already "learned" these from Gemini.
        # This saves you credits on every run!
        self.cache = {
            'sardine_eggs': {'scientific_name': 'Sardinops sagax', 'common_name': 'Pacific Sardine'},
            'anchovy_eggs': {'scientific_name': 'Engraulis mordax', 'common_name': 'Northern Anchovy'},
            'jack_mackerel_eggs': {'scientific_name': 'Trachurus symmetricus', 'common_name': 'Jack Mackerel'},
            'hake_eggs': {'scientific_name': 'Merluccius productus', 'common_name': 'Pacific Hake'},
            'squid_eggs': {'scientific_name': 'Doryteuthis opalescens', 'common_name': 'Market Squid'},
            'other_fish_eggs': {'scientific_name': 'Actinopterygii', 'common_name': 'Various Fish Eggs'},
            'zooplankton': {'scientific_name': 'Zooplankton', 'common_name': 'Mixed Zooplankton'}
        }

        if self.enabled:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Gemini LLM enabled (Hybrid Mode: Caching active)")
            except Exception as exc:
                logger.warning("Gemini library failed to load, using local cache only.")
                self.enabled = False

    def parse_species(self, text: str) -> Dict:
        # 1. Check our local 'Zero-Credit' cache first
        if text in self.cache:
            return self.cache[text]
        
        # 2. If it's a new species and we have credits/connection, ask Gemini
        if self.enabled and self.client:
            logger.info(f"New species detected: '{text}'. Querying Gemini...")
            prompt = (f"Identify the marine species for the label: '{text}'. "
                      f"Return JSON: scientific_name, common_name.")
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                # Quick clean and parse
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                result = json.loads(clean_json)
                # Add to cache for next time
                self.cache[text] = result
                return result
            except:
                pass
        
        # 3. Fallback if everything else fails
        return {'scientific_name': text.replace('_', ' ').title(), 'common_name': 'Marine Organism'}

class DataNormalizer:
    def __init__(self, config: ETLConfig, llm: LLMExtractor):
        self.config = config
        self.llm = llm

    def normalize(self, fish_df: pd.DataFrame, zoo_df: pd.DataFrame, phys_df: pd.DataFrame) -> pd.DataFrame:
        # 1. Process Fish Eggs
        # Melt specific egg columns into a long format
        egg_cols = ['sardine_eggs', 'anchovy_eggs', 'jack_mackerel_eggs', 'hake_eggs', 'squid_eggs', 'other_fish_eggs']
        fish_melted = fish_df.melt(
            id_vars=['cruise', 'time', 'latitude', 'longitude'], 
            value_vars=[c for c in egg_cols if c in fish_df.columns],
            var_name='species_group', value_name='count'
        )
        fish_melted['biological_type'] = 'larval_fish'

        # 2. Process Zooplankton
        zoo_df['species_group'] = 'zooplankton'
        zoo_df['biological_type'] = 'zooplankton'
        zoo_df = zoo_df.rename(columns={'total_plankton': 'count'})

        # 3. Combine Biological Datasets
        combined = pd.concat([
            fish_melted, 
            zoo_df[['cruise', 'time', 'latitude', 'longitude', 'species_group', 'count', 'biological_type']]
        ], ignore_index=True)
        
        combined['time'] = pd.to_datetime(combined['time'], errors='coerce')
        combined['year'] = combined['time'].dt.year
        combined['month'] = combined['time'].dt.month
        
        # 4. Run Gemini Species Normalization
        if self.llm.enabled:
            logger.info("Running species normalization through Gemini...")
            unique_labels = combined['species_group'].unique()
            mapping = {lbl: self.llm.parse_species(str(lbl)) for lbl in unique_labels}
            
            combined['scientific_name'] = combined['species_group'].map(lambda x: mapping[x].get('scientific_name'))
            combined['common_name'] = combined['species_group'].map(lambda x: mapping[x].get('common_name'))

        # 5. Merge with Physical Data
        # We round coordinates to 2 decimal places to ensure spatial joins work correctly
        phys_df['Date'] = pd.to_datetime(phys_df['Date'])
        phys_df['year_phys'] = phys_df['Date'].dt.year
        phys_df['month_phys'] = phys_df['Date'].dt.month
        
        combined['lat_round'] = combined['latitude'].round(2)
        combined['lon_round'] = combined['longitude'].round(2)
        phys_df['lat_round'] = phys_df['Lat_Dec'].round(2)
        phys_df['lon_round'] = phys_df['Lon_Dec'].round(2)

        final_merge = pd.merge(
            combined, 
            phys_df[['year_phys', 'month_phys', 'lat_round', 'lon_round', 'T_degC', 'Salnty']], 
            left_on=['year', 'month', 'lat_round', 'lon_round'],
            right_on=['year_phys', 'month_phys', 'lat_round', 'lon_round'],
            how='left'
        )
        
        return final_merge.drop(columns=['year_phys', 'month_phys', 'lat_round', 'lon_round'])

if __name__ == '__main__':
    config = ETLConfig()
    extractor = DataExtractor(config)
    llm = LLMExtractor(config)
    normalizer = DataNormalizer(config, llm)
    
    # Extract
    fish = extractor.read_source(config.fish_egg_path)
    zoo = extractor.read_source(config.zooplankton_path)
    phys = pd.read_csv(config.physical_oceanography_path)
    
    # Transform
    final_df = normalizer.normalize(fish, zoo, phys)
    
    # Load to SQLite for the Predictive Model
    with sqlite3.connect(config.output_sqlite) as conn:
        final_df.to_sql('observations', conn, if_exists='replace', index=False)
    
    # Load to JSON for Dashboard
    final_df.head(100).to_json(config.output_json, orient='records', indent=2)
    
    logger.info(f"Pipeline Complete: {len(final_df)} records merged and saved.")