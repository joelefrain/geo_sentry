import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd

from libs.utils.config_variables import VALID_TEXT_EXTENSIONS
from libs.utils.config_logger import get_logger
from libs.utils.text_helpers import (
    extract_str_suffix,
    parse_datetime,
    read_lines,
    extract_line,
    extract_matrix,
)

logger = get_logger("modules.calculations.text_processor")


def build_dataframe(raw_data, headers, target_columns, new_column_names):
    full_df = pd.DataFrame(raw_data, columns=headers)

    if len(target_columns) != len(new_column_names):
        raise ValueError(
            "Longitudes de columnas objetivo y nombres nuevos no coinciden."
        )

    df = full_df[target_columns].copy()
    df.columns = new_column_names

    return df.apply(lambda x: pd.to_numeric(x.astype(str).str.strip(), errors="coerce"))


def insert_datetime(df, lines, params):
    try:
        date_str = extract_str_suffix(lines[params["date_line"]])
        time_str = (
            extract_str_suffix(lines[params["time_line"]])
            if "time_line" in params
            else ""
        )
        dt = parse_datetime(
            date_str, time_str, params["date_format"], params.get("time_format", "")
        )
        df.insert(0, "time", dt)
    except Exception as e:
        logger.warning(f"No se pudo parsear la fecha y hora: {e}")
    return df


def parse_text_file(filepath, match_columns, **kwargs):
    logger.info(f"Procesando archivo de texto plano: {filepath}")

    params = kwargs or None
    if not params:
        raise ValueError("No se proporcionaron parámetros para el procesamiento.")

    try:
        lines = read_lines(filepath)
        headers = extract_line(lines[params["head_line"]])
        raw_data = extract_matrix(lines[params["data_lines_start"] :])
        df = build_dataframe(
            raw_data, headers, params["target_columns"], params["columns"]
        )

        if "date_line" in params:
            df = insert_datetime(df, lines, params)

        df.dropna(subset=match_columns, inplace=True)

    except Exception as e:
        logger.exception(f"Error al procesar {filepath}: {e}")
        return None

    return df


def text_folder_to_csv(folder, match_columns, **kwargs):
    all_dfs = []
    for f in os.listdir(folder):
        if f.lower().endswith(VALID_TEXT_EXTENSIONS):
            df = parse_text_file(os.path.join(folder, f), match_columns, **kwargs)
            if df is not None and not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["time"] + match_columns)

    # Sort the combined DataFrame by time and FLEVEL
    return pd.concat(all_dfs, ignore_index=True).sort_values(by=match_columns)
