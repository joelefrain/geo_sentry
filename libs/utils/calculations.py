import numpy as np
from typing import Tuple
from datetime import datetime
import locale

from .config_variables import DECIMAL_CHAR


def round_decimal(value: float, decimals: int, decimal_char: str = DECIMAL_CHAR) -> str:
    """
    Rounds a float to a specified number of decimal places and replaces the decimal point with the specified character.

    Parameters
    ----------
    value : float
        The float value to round.
    decimals : int
        The number of decimal places.
    decimal_char : str
        The character to use as the decimal point ('.' or ',').

    Returns
    -------
    str
        The rounded value as a string with the specified decimal character.
    """
    rounded_value = round(value, decimals)
    formatted_value = f"{rounded_value:.{decimals}f}"
    return formatted_value.replace(".", decimal_char)

def get_iqr_limits(data: list, margin_factor: float = 1.5) -> Tuple[float, float]:
    """
    Calculate plot limits using the Interquartile Range (IQR) method.

    Parameters
    ----------
    data : list
        List of numeric values.
    margin_factor : float, optional
        Factor to multiply IQR for margin calculation, by default 1.5.

    Returns
    -------
    Tuple[float, float]
        Lower and upper limits (y_min, y_max).
    """
    data_arr = np.asarray(data)
    q1, q3 = np.nanquantile(data_arr, [0.25, 0.75])
    margin = (q3 - q1) * margin_factor
    return q1 - margin, q3 + margin

def round_lower(value):
    return int(value // 1)

def round_upper(value):
    return int(-(-value // 1))

def format_date_long(date):
    """Convert date to 'mmmm yyyy' format in Spanish using locale."""
    locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")  # Set locale to Spanish
    return date.strftime("%B %Y").lower()

def format_date_short(date):
    """Convert date to 'dd-mm-yy' format."""
    return date.strftime("%d-%m-%y")