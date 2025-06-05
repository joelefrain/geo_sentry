import os
import pandas as pd
from datetime import datetime

DEFAULT_INC_PARAMS = {
    "date_format": "%m/%d/%y %H",
    "date_lines": (4, 5),
    "data_lines_start": 9,
    "azimuth_rad": 0.0,
    "enbankment_slope_rad": 0.0,
    "a_axis_scale": 0.005,
    "b_axis_scale": 0.005,
}

def extract_non_time_suffix(text_line):
    """Devuelve el texto que sigue al primer ':' de la línea, sin incluir el ':'."""
    if ':' in text_line:
        return text_line.split(':', 1)[1].strip()
    return ''

def parse_datetime(date_str, time_str, date_format):
    """
    Intenta convertir 'date_str' y 'time_str' según 'date_format'.
    Si 'time_str' está vacío, intenta solo con la fecha.
    Si solo hay hora o ambos fallan, devuelve pd.NaT.
    """
    if time_str:
        datetime_str = f"{date_str} {time_str}"
    else:
        datetime_str = date_str

    try:
        return datetime.strptime(datetime_str, date_format)
    except ValueError:
        return pd.NaT


def parse_gkn_file(filepath, match_columns, **kwargs):
    params = {**DEFAULT_INC_PARAMS, **kwargs}

    with open(filepath, "r") as file:
        lines = file.readlines()

        date_line = lines[params["date_lines"][0]].strip()
        time_line = lines[params["date_lines"][1]].strip()
        data_lines_start = params["data_lines_start"]

        date_str = extract_non_time_suffix(date_line)
        time_str = extract_non_time_suffix(time_line)

        dt = parse_datetime(date_str, time_str, params["date_format"])

        data_lines = lines[data_lines_start:]
        data = [line.strip().split(",") for line in data_lines if line.strip()]
        df = pd.DataFrame(data, columns=["FLEVEL", "A+", "A-", "B+", "B-"])
        df = df.apply(lambda x: pd.to_numeric(x.str.strip(), errors="coerce"))

        df.insert(0, "time", dt)
        df["base_line"] = False

        # Asignar parámetros usando dict comprehension
        for key in [
            "azimuth_rad",
            "enbankment_slope_rad",
            "a_axis_scale",
            "b_axis_scale",
        ]:
            df[key] = params[key]
        df.dropna(subset=match_columns, inplace=True)
        return df


def gkn_folder_to_csv(folder, match_columns, **kwargs):
    all_dfs = [
        parse_gkn_file(os.path.join(folder, f), match_columns, **kwargs)
        for f in os.listdir(folder)
        if f.lower().endswith(".gkn")
    ]
    # Sort the combined DataFrame by time and FLEVEL
    return pd.concat(all_dfs, ignore_index=True).sort_values(by=match_columns)
