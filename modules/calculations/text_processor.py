import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import re
import pandas as pd
from datetime import datetime

from libs.utils.config_variables import VALID_TEXT_EXTENSIONS
from libs.utils.config_logger import get_logger

logger = get_logger("modules.calculations.text_processor")


def extract_non_time_suffix(text_line):
    """Devuelve el texto que sigue al primer ':' de la línea, sin incluir el ':'."""
    if ":" in text_line:
        return text_line.split(":", 1)[1].strip()
    return ""


def clean_str(s, allowed=r"[^\w\s:/\-.]", remove_space=False):
    """
    Limpia el string 's' eliminando los caracteres no permitidos definidos por 'allowed'.
    Si remove_space es True, también elimina todos los espacios.

    - 'allowed' debe ser una expresión regular de *caracteres a eliminar* (se usa con re.sub).
    - Por ejemplo: r"[^\d\s:/\-.]" elimina todo excepto dígitos, espacio, :, /, -, .
    """
    cleaned = re.sub(allowed, "", s)
    if remove_space:
        cleaned = cleaned.replace(" ", "")
    return cleaned.strip()


def parse_datetime(date_str, time_str, date_formats, time_formats):
    # Mantiene dígitos, espacio, :, /, -, .
    allowed_chars = r"[^\d\s:/\-.]"

    date_str = clean_str(date_str, allowed=allowed_chars, remove_space=False)
    time_str = (
        clean_str(time_str, allowed=allowed_chars, remove_space=False)
        if time_str
        else ""
    )

    if time_str:
        for d_fmt in date_formats:
            for t_fmt in time_formats:
                dt_fmt = f"{d_fmt} {t_fmt}"
                try:
                    return datetime.strptime(f"{date_str} {time_str}", dt_fmt)
                except ValueError:
                    continue

    for d_fmt in date_formats:
        try:
            return datetime.strptime(date_str, d_fmt)
        except ValueError:
            continue

    return pd.NaT


def parse_text_file(filepath, match_columns, **kwargs):
    logger.info(f"Procesando archivo ARTE: {filepath}")

    params = kwargs if kwargs else None
    if not params:
        raise ValueError("No se proporcionaron parámetros para el procesamiento.")

    with open(filepath, "r") as file:
        lines = file.readlines()

        date_line = lines[params["date_line"]]
        time_line = lines[params["time_line"]]
        head_line = lines[params["head_line"]]
        data_lines = lines[params["data_lines_start"] :]

        date_str = extract_non_time_suffix(date_line)
        time_str = extract_non_time_suffix(time_line)
        dt = parse_datetime(
            date_str, time_str, params["date_format"], params["time_format"]
        )

        # Solo letras, números y signos +, -, (, ), sin espacios
        head = [
            clean_str(col, allowed=r"[^\w()+\-]", remove_space=True)
            for col in re.split(r"[;,]", head_line.strip())
        ]


        # Procesar datos limpiando los campos
        raw_data = [
            [cell.strip() for cell in re.split(r"[;,]", line.strip())]
            for line in data_lines
            if line.strip()
        ]

        try:
            full_df = pd.DataFrame(raw_data, columns=head)

            if len(params["target_columns"]) != len(params["columns"]):
                logger.error(
                    f"Las columnas seleccionadas no coinciden con las columnas destino en {filepath}"
                )
                return None

            df = full_df[params["target_columns"]].copy()
            df.columns = params["columns"]

            df = df.apply(
                lambda x: pd.to_numeric(x.astype(str).str.strip(), errors="coerce")
            )

            df.insert(0, "time", dt)
            df.dropna(subset=match_columns, inplace=True)

        except Exception as e:
            logger.error(f"Error al procesar el DataFrame de {filepath}: {e}")
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
