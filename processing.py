import os
import re
import json
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ETLConfig:
    # Updated to match your exact filenames
    fish_egg_path: Path = Path('fish2.csv') 
    zooplankton_path: Path = Path('zooplankton.csv') 
    physical_oceanography_path: Path = Path('calcofi_physical_merged.csv')
    
    output_sqlite: Path = Path('marine_observations.db')
    output_json: Path = Path('dashboard_data.json')
    llm_enabled: bool = True 
    llm_model: str = 'gemini-1.5-flash'
    llm_api_key: Optional[str] = os.getenv('GOOGLE_API_KEY')

class DataExtractor:
    def __init__(self, config: ETLConfig):
        self.config = config

    def read_source(self, source_path: Path) -> Optional[pd.DataFrame]:
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            return None
        
        logger.info(f"Reading {source_path}")
        # Detect if there's a units row (common in CalCOFI data)
        raw = pd.read_csv(source_path, header=None, nrows=2, dtype=str, encoding='latin1')
        skip = [1] if raw.shape[0] == 2 and any('degrees' in str(x).lower() for x in raw.iloc[1]) else []
        
        df = pd.read_csv(source_path, skiprows=skip, encoding='latin1')
        return self._standardize_columns(df)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'[^a-z0-9_]', '', col.strip().lower().replace(' ', '_')) for col in df.columns]
        return df

class LLMExtractor:
    def __init__(self, config: ETLConfig):
        self.enabled = config.llm_enabled and config.llm_api_key is not None
        self.client = None
        if self.enabled:
            import google.generativeai as genai
            genai.configure(api_key=config.llm_api_key)
            self.client = genai.GenerativeModel(config.llm_model)

    def parse_species(self, text: str) -> Dict:
        if not self.enabled: return {}
        prompt = f"Return JSON with scientific_name and common_name for marine species: {text}"
        try:
            response = self.client.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except:
            return {}

class DataNormalizer:
    def __init__(self, config: ETLConfig, llm: LLMExtractor):
        self.config = config
        self.llm = llm

    def normalize(self, fish_df: pd.DataFrame, zoo_df: pd.DataFrame, phys_df: pd.DataFrame) -> pd.DataFrame:
        # 1. Process Fish Eggs (Melt multiple columns)
        egg_cols = ['sardine_eggs', 'anchovy_eggs', 'jack_mackerel_eggs', 'hake_eggs', 'squid_eggs', 'other_fish_eggs']
        fish_melted = fish_df.melt(
            id_vars=['cruise', 'time', 'latitude', 'longitude'], 
            value_vars=[c for c in egg_cols if c in fish_df.columns],
            var_name='species_group', value_name='count'
        )
        fish_melted['biological_type'] = 'fish_egg'

        # 2. Process Zooplankton
        zoo_df['species_group'] = 'zooplankton'
        zoo_df['biological_type'] = 'zooplankton'
        zoo_df = zoo_df.rename(columns={'total_plankton': 'count'})

        # Combine
        combined = pd.concat([fish_melted, zoo_df[['cruise', 'time', 'latitude', 'longitude', 'species_group', 'count', 'biological_type']]])
        combined['time'] = pd.to_datetime(combined['time'], errors='coerce')
        combined['year'] = combined['time'].dt.year
        combined['month'] = combined['time'].dt.month
        
        # 3. Gemini Normalization
        if self.llm.enabled:
            unique_species = combined['species_group'].unique()
            mapping = {s: self.llm.parse_species(s) for s in unique_species}
            combined['scientific_name'] = combined['species_group'].map(lambda x: mapping[x].get('scientific_name'))
            combined['common_name'] = combined['species_group'].map(lambda x: mapping[x].get('common_name'))

        # 4. Merge Physical Data
        phys_df['Date'] = pd.to_datetime(phys_df['Date'])
        phys_df['year'] = phys_df['Date'].dt.year
        phys_df['month'] = phys_df['Date'].dt.month
        
        return pd.merge(
            combined, 
            phys_df[['year', 'month', 'Lat_Dec', 'Lon_Dec', 'T_degC', 'Salnty']], 
            left_on=['year', 'month', 'latitude', 'longitude'],
            right_on=['year', 'month', 'Lat_Dec', 'Lon_Dec'],
            how='left'
        )

if __name__ == '__main__':
    config = ETLConfig()
    extractor = DataExtractor(config)
    llm = LLMExtractor(config)
    normalizer = DataNormalizer(config, llm)
    
    fish = extractor.read_source(config.fish_egg_path)
    zoo = extractor.read_source(config.zooplankton_path)
    phys = pd.read_csv(config.physical_oceanography_path)
    
    final_df = normalizer.normalize(fish, zoo, phys)
    
    # Save results
    with sqlite3.connect(config.output_sqlite) as conn:
        final_df.to_sql('observations', conn, if_exists='replace')
    final_df.head(100).to_json(config.output_json, orient='records')
    
    logger.info(f"Success! Processed {len(final_df)} records.")