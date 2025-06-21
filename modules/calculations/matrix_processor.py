import numpy as np
import pandas as pd


def apply_self_operations(df, self_operations):
    """
    Apply self_operations to the DataFrame.

    Args:
        df: DataFrame to process.
        self_operations: Dictionary of self_operations formulas.

    Returns:
        DataFrame with self_operations applied.
    """
    if not self_operations:
        return df
    for formula_name, formula in self_operations.items():
        df[formula_name] = eval(formula)(df)
    return df


def apply_on_base_operations(df, base_df, on_base_operations, match_columns):
    """
    Apply on_base_operations using the closest base DataFrame.

    Args:
        df: DataFrame to process.
        base_df: Base DataFrame for reference.
        on_base_operations: Dictionary of on_base_operations formulas.
        match_columns: Columns to match between df and base_df.

    Returns:
        DataFrame with on_base_operations applied.
    """
    if not on_base_operations:
        return df
    # Ensure match_columns are complete with respect to the base
    df = df.merge(
        base_df[match_columns], on=match_columns, how="left", suffixes=("", "_base")
    )

    # Fill missing values with NaN for unmatched records
    df.fillna(value=np.nan, inplace=True)

    # Perform calculations using on_base_operations
    for formula_name, formula in on_base_operations.items():
        df[formula_name] = eval(formula)(df, base_df)
    return df


def prepare_data(df, config):
    """Prepare data for processing by setting up time and base line."""
    base_line_col = config["base"]["base_line_column"]
    match_columns = config["process_config"]["match_columns"]

    # If no base_line=True exists, set the oldest record as base_line
    if not df[base_line_col].any():
        oldest_idx = df["time"].idxmin()
        df.loc[oldest_idx, base_line_col] = True

    # Sort and group by time
    return df.sort_values(match_columns).reset_index(drop=True)


def process_self_operations(df, config):
    """Process only self operations on the DataFrame."""
    if "self_operations" not in config:
        return df

    match_columns = config["process_config"]["match_columns"]
    self_operations = config["self_operations"]

    df = prepare_data(df, config)
    grouped = df.groupby("time")
    results = []

    for _, group in grouped:
        group = group.sort_values(match_columns).reset_index(drop=True)
        group = apply_self_operations(group, self_operations)
        results.append(group)

    return pd.concat(results).reset_index(drop=True)


def process_base_operations(df, config):
    """Process only base operations on the DataFrame."""
    if "base_operations" not in config:
        return df

    match_columns = config["process_config"]["match_columns"]
    base_line_col = config["base"]["base_line_column"]
    base_operations = config["base_operations"]

    df = prepare_data(df, config)
    grouped = df.groupby("time")
    group_list = []

    # Group data and identify base lines
    for time, group in grouped:
        group = group.sort_values(match_columns).reset_index(drop=True)
        base_line_status = group[base_line_col].any()
        group_list.append((time, base_line_status, group))

    # Process base operations
    base_groups = [(t, g) for t, is_base, g in group_list if is_base]
    base_times = [t for t, _ in base_groups]
    base_dict = {t: g for t, g in base_groups}

    def find_closest_base(time):
        candidates = [bt for bt in base_times if bt <= time]
        return max(candidates) if candidates else None

    results = []
    for time, base_line_status, group in group_list:
        if not base_line_status:
            closest_time = find_closest_base(time)
            if closest_time:
                base_df = base_dict[closest_time]
                group = apply_on_base_operations(
                    group, base_df, base_operations, match_columns
                )
        results.append(group)

    return pd.concat(results).reset_index(drop=True)
