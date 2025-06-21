import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd

from datetime import datetime

from .matrix_processor import process_self_operations, process_base_operations

from libs.utils.config_variables import CALC_CONFIG_DIR

from libs.utils.config_loader import load_toml
from libs.utils.df_helpers import read_df_on_time_from_csv

from libs.utils.config_logger import get_logger

logger = get_logger("modules.calculations.data_processor")


class DataProcessor:
    def __init__(self, config_name: str):
        self.config = load_toml(CALC_CONFIG_DIR, config_name)

    def load_data(self, csv_path: str, set_index: bool = False) -> pd.DataFrame:
        """
        Load data from a CSV file with date parsing using pandas to_datetime.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            pd.DataFrame with the loaded data.
        """
        df = read_df_on_time_from_csv(csv_path, set_index)

        return df

    def prepare_data(
        self, df: pd.DataFrame, match_columns, overall_columns
    ) -> pd.DataFrame:
        """
        Remove duplicate and future records based on the 'time' column, and fill missing
        values in overall_columns with the previous record's values.

        Args:
            df: DataFrame with the data.
            match_columns: List of columns to sort and identify duplicates.
            overall_columns: List of columns to forward-fill if values are missing.

        Returns:
            pd.DataFrame with the cleaned data.
        """

        # Sort by match_columns
        df = df.sort_values(match_columns, ignore_index=True)

        # Remove rows with NaN in match_columns
        df = df.dropna(subset=match_columns)

        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=match_columns, keep="first")

        # Remove records with 'time' after the current date
        current_time = datetime.now()
        df = df[df["time"] <= current_time]

        # Forward-fill missing values in overall_columns
        df[overall_columns] = df[overall_columns].ffill()

        # Asegurar base_line
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
                segment[col] = (
                    lambda_func(segment, base_values)
                    if "base" in formula
                    else lambda_func(segment)
                )
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
        """Process data using configurations from a TOML file."""
        config = self.config

        # Process absolute values
        if "abs" in config:
            df = self.process_absolute_values(df)

        # Process relative values
        if "rel" in config:
            df = self.process_relative_values(df)

        # Process self operations in matrix data
        if "self_operations" in config:
            df = process_self_operations(df, self.config)

        # Process base operations in matrix data
        if "base_operations" in config:
            df = process_base_operations(df, self.config)

        return df
