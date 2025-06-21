import pandas as pd

from pathlib import Path

from .config_variables import SEP_FORMAT


def read_df_on_time_from_csv(
    path: Path,
    set_index: bool = True,
    auto_convert: bool = False,
    num_decimals: int = 4,
) -> pd.DataFrame:
    """Lee un DataFrame desde un archivo CSV y configura la columna de tiempo."""

    # Leer CSV desde una ruta
    df = read_df_from_csv(path)

    # Convertir la columna time a datetime preservando milisegundos
    df = config_time_df(df, set_index)

    # Conversión automática de tipos de datos
    if auto_convert:
        # Convertir columnas
        df = df.convert_dtypes()

        # Reemplazar pd.NA con np.nan para evitar errores en gráficos
        df = df.apply(
            lambda col: col.astype(float) if col.dtype.name == "Float64" else col
        )

        # Redondear columnas de tipo float a num_decimals
        float_cols = df.select_dtypes(include=["floating"]).columns
        df[float_cols] = df[float_cols].round(num_decimals)

    return df


def read_df_from_csv(path: Path) -> pd.DataFrame:
    # Leer CSV desde una ruta
    df = pd.read_csv(path, sep=SEP_FORMAT)
    return df


def config_time_df(df: pd.DataFrame, set_index: bool = True) -> pd.DataFrame:
    # Convertir la columna time a datetime y establecer como índice
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="raise")
    if set_index:
        df = df.set_index("time")
    return df


def save_df_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """Guarda un DataFrame en un archivo CSV.

    Args:
        df: DataFrame a guardar
        file_path: Ruta donde guardar el archivo CSV
    """
    df.to_csv(file_path, index=False, sep=SEP_FORMAT)


def merge_new_records(df1, df2, match_columns=["time"], match_type="all"):
    # Convertir match_columns a lista si es string
    if isinstance(match_columns, str):
        match_columns = [match_columns]

    # Verificar que las columnas existan en ambos DataFrames
    for col in match_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"La columna {col} no existe en ambos DataFrames")

    # Crear máscara de coincidencia según el tipo de comparación
    if match_type.lower() == "all":
        # Todos los valores deben coincidir
        mask = ~df2[match_columns].apply(tuple, axis=1).isin(
            df1[match_columns].apply(tuple, axis=1)
        )
    elif match_type.lower() == "any":
        # Al menos un valor debe coincidir
        mask = ~df2[match_columns].isin(df1[match_columns]).any(axis=1)
    else:
        raise ValueError("match_type debe ser 'all' o 'any'")

    # Identificar registros en df2 que no están en df1
    missing_records = df2[mask]

    # Añadir estos registros a df1
    df1 = pd.concat([df1, missing_records], ignore_index=True)

    # Seleccionar solo las columnas en común
    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns]

    # Eliminar duplicados basados en las columnas de coincidencia
    df1 = df1.drop_duplicates(subset=match_columns, keep="first")

    return df1


def assign_params(df, **kwargs):
    """Asigna parámetros adicionales al DataFrame.

    Args:
        df: DataFrame al que se le asignarán los parámetros
        **kwargs: Parámetros a asignar como columnas en el DataFrame

    Returns:
        pd.DataFrame: DataFrame con los parámetros asignados
    """
    for key, value in kwargs.items():
        df[key] = value
    return df
