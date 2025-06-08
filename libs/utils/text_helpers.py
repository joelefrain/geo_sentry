import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import re
import pandas as pd
from datetime import datetime

from libs.utils.config_variables import (
    ALLOWED_TIME_CHARS,
    ALLOWED_SEP_CHARS,
    ALLOWD_HEADER_CHARS,
)


def to_sentence_format(text: str, mode: str = "lower") -> str:
    """
    Formatea un nombre de serie aplicando un tipo de conversión a la parte antes del paréntesis
    y conservando intacto el contenido entre paréntesis.

    Parámetros:
        text (str): El texto original.
        mode (str): Tipo de conversión aplicado a la parte antes del paréntesis. Opciones:
            - 'lower':        convierte todo a minúsculas (ej. "TURBIDEZ (NTU)" → "turbidez (NTU)")
            - 'sentence':     primera letra en minúscula, el resto igual (ej. "Presión (kPa)" → "presión (kPa)")
            - 'capitalize':   primera letra en mayúscula, el resto en minúsculas (ej. "Presión (kPa)" → "Presión (kPa)")
            - 'title':        tipo título (mayúscula inicial de cada palabra) (ej. "oxígeno disuelto (mg/L)" → "Oxígeno Disuelto (mg/L)")
            - 'original':     mantiene el texto tal como está (ej. "Presión (kPa)" → "Presión (kPa)")
            - 'decapitalize': convierte solo la primera letra a minúscula, dejando el resto intacto.

    Retorna:
        str: El texto formateado.

    Ejemplos:
        >>> to_sentence_format("Presión (kPa)", mode="lower")
        'presión (kPa)'

        >>> to_sentence_format("TURBIDEZ (NTU)", mode="sentence")
        'turbidez (NTU)'

        >>> to_sentence_format("oxígeno disuelto (mg/L)", mode="capitalize")
        'Oxígeno disuelto (mg/L)'

        >>> to_sentence_format("oxígeno disuelto (mg/L)", mode="title")
        'Oxígeno Disuelto (mg/L)'

        >>> to_sentence_format("pH in-situ (mg/L)", mode="original")
        'pH in-situ (mg/L)'

        >>> to_sentence_format("PH in-situ (mg/L)", mode="decapitalize")
        'pH in-situ (mg/L)'"""
    text = text.strip()

    if "(" in text:
        main = text.split("(")[0].strip()
        suffix = text[text.find("(") :]
    else:
        main = text
        suffix = ""

    if mode == "lower":
        main = main.lower()
    elif mode == "sentence":
        main = main.lower()
    elif mode == "capitalize":
        main = main.capitalize()
    elif mode == "title":
        main = main.title()
    elif mode == "original":
        pass
    elif mode == "decapitalize":
        main = main[0].lower() + main[1:] if main else ""
    else:
        raise ValueError(f"Modo de conversión no válido: {mode}")

    return f"{main} {suffix}".strip()


def extract_str_suffix(text_line, caracter=":"):
    """Devuelve el texto que sigue al primer 'caracter' de la línea, sin incluir el 'caracter'."""
    if caracter in text_line:
        return text_line.split(caracter, 1)[1].strip()
    return ""


def clean_str(s, header_chars=r"[^\w\s:/\-.]", remove_space=False):
    """
    Limpia el string 's' eliminando los caracteres no permitidos definidos por 'allowed'.
    Si remove_space es True, también elimina todos los espacios.

    - 'allowed' debe ser una expresión regular de *caracteres a eliminar* (se usa con re.sub).
    - Por ejemplo: r"[^\d\s:/\-.]" elimina todo excepto dígitos, espacio, :, /, -, .
    """
    cleaned = re.sub(header_chars, "", s)
    if remove_space:
        cleaned = cleaned.replace(" ", "")
    return cleaned.strip()


def parse_datetime(date_str, time_str, date_formats, time_formats):
    # Mantiene dígitos, espacio, :, /, -, .

    date_str = clean_str(date_str, header_chars=ALLOWED_TIME_CHARS, remove_space=False)
    time_str = (
        clean_str(time_str, header_chars=ALLOWED_TIME_CHARS, remove_space=False)
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


def read_lines(filepath):
    with open(filepath, "r") as file:
        return file.readlines()


def extract_line(line, delimiter=ALLOWED_SEP_CHARS, header_chars=ALLOWD_HEADER_CHARS):
    return [
        clean_str(col, header_chars=header_chars, remove_space=True)
        for col in re.split(delimiter, line.strip())
    ]


def extract_matrix(data_lines, delimiter=ALLOWED_SEP_CHARS):
    return [
        [cell.strip() for cell in re.split(delimiter, line.strip())]
        for line in data_lines
        if line.strip()
    ]
