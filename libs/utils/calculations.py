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
