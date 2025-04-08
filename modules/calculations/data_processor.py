import os
import sys

# Add 'libs' path to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from libs.utils.config_variables import SEP_FORMAT
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_logger import get_logger

import pandas as pd
import numpy as np
import tomli
from pathlib import Path
from datetime import datetime

logger = get_logger("modules.calculations")

class DataProcessor:
    def __init__(self, config_name: str):
        self.config = self._load_config(config_name)

    def _load_config(self, config_name: str) -> dict:
        """
        Load configuration from a TOML file.

        Args:
            config_name: Name of the configuration (e.g., 'prisms').

        Returns:
            dict with the configuration.
        """
        toml_path = Path(__file__).parent / 'data' / f'{config_name}.toml'
        if not toml_path.is_file():
            raise ValueError(f"Configuration file '{config_name}.toml' not found in 'data' directory.")
        
        with open(toml_path, "rb") as f:
            return tomli.load(f)

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file with date parsing using pandas to_datetime.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            pd.DataFrame with the loaded data.
        """
        df = read_df_on_time_from_csv(csv_path)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate and future records based on the 'time' column.

        Args:
            df: DataFrame with the data.

        Returns:
            pd.DataFrame with the cleaned data.
        """
        
        # Remove any rows where time parsing failed
        df = df.dropna(subset=['time'])
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset='time', keep='first')
        
        # Remove records with 'time' after the current date
        current_time = datetime.now()
        df = df[df['time'] <= current_time]
        
        # Sort by time
        df.sort_values("time", ignore_index=True)
        
        # Aesgurar base line
        df.loc[df.index[0], "base_line"] = True
        
        return df

    def process_absolute_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process absolute values based on multiple base lines.
        Each segment is processed using its preceding base line.
        """
        config = self.config
        base_mask = df[config["base"]["base_line_column"]]
        if base_mask.sum() == 0:
            raise ValueError("No base_line=True records found")

        # Get indices where base_line is True
        base_indices = df[base_mask].index.tolist()
        
        # Add the last index of the dataframe to process the final segment
        all_indices = base_indices + [len(df)]
        
        # Process each segment
        result_dfs = []
        for i in range(len(base_indices)):
            # Get current base line
            current_base = df.loc[base_indices[i]]
            
            # Get the segment (from current base to next base or end)
            start_idx = base_indices[i]
            end_idx = all_indices[i + 1]
            segment = df.iloc[start_idx:end_idx].copy()
            
            # Create base values dictionary
            base_values = {
                col: current_base[col.replace("_0", "")] 
                for col in config["base"]["base_columns"]
            }
            
            # Process each column for the segment
            for col, formula in config["abs"].items():
                lambda_func = eval(formula)
                segment[col] = lambda_func(segment, base_values) if "base" in formula else lambda_func(segment)
                if col not in segment.columns:
                    segment[col] = None
            
            result_dfs.append(segment)
        
        # Combine all processed segments
        return pd.concat(result_dfs)

    def process_relative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process relative values based on the configuration.

        Args:
            df: DataFrame with the data.

        Returns:
            pd.DataFrame with the processed relative values.
        """
        config = self.config
        for col, formula in config["rel"].items():
            lambda_func = eval(formula)
            df[col] = lambda_func(df)
            if col not in df.columns:
                df[col] = None  # Assume missing attributes as null

        return df

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process deformation data using configurations from a TOML file.

        Args:
            df: DataFrame with the data.

        Returns:
            pd.DataFrame with the processed data.
        """
        config = self.config

        # Process absolute values
        if "abs" in config:
            df = self.process_absolute_values(df)

        # Process relative values
        if "rel" in config:
            df = self.process_relative_values(df)

        return df
