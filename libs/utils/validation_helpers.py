from pathlib import Path


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))  # Desciende un nivel
        else:
            flat_list.append(item)  # Elemento final
    return flat_list


def get_field_from_dict(fields: str | list[str], dictionary: dict):
    """
    Verifica si una clave (o ruta de claves anidadas) está definida en un diccionario,
    e informa en qué punto exacto ocurre un fallo si lo hay.

    Parameters
    ----------
    fields : str | list[str]
        Clave o lista de claves que representan un acceso anidado al diccionario.
    dictionary : dict
        Diccionario sobre el cual se realiza la validación.

    Returns
    -------
    value
        Valor extraído del diccionario si se encuentra correctamente.

    Raises
    ------
    ValueError
        Si alguna de las claves no está definida en el diccionario.
    """
    if isinstance(fields, str):
        fields = [fields]

    value = dictionary
    path_traversed = []

    for key in fields:
        path_traversed.append(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"Esperado un diccionario en la ruta {' -> '.join(path_traversed[:-1])}, "
                f"pero se encontró {type(value).__name__}."
            )
        if key not in value:
            raise ValueError(
                f"Clave '{key}' no definida en la ruta {' -> '.join(path_traversed[:-1])} "
                f"dentro del diccionario."
            )
        value = value[key]

    if value is None:
        raise ValueError(
            f"Valor en la ruta {' -> '.join(path_traversed)} está definido como None."
        )

    return value


def validate_folder(
    path: str | Path, create_if_missing: bool = False
) -> Path:
    """
    Valida si una carpeta existe. Opcionalmente, la crea si no existe.

    Parámetros:
    ----------
    path : str | Path
        Ruta a validar.
    create_if_missing : bool
        Si es True, crea la carpeta si no existe.

    Retorna:
    -------
    Path
        Objeto Path de la ruta validada o creada.

    Lanza:
    -----
    FileNotFoundError si la ruta no existe y `create_if_missing` es False.
    """
    path = Path(path)

    if path.is_dir():
        return path

    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
        return path
    else:
        raise FileNotFoundError(f"La ruta no existe: {path}")
    
def validate_file(path: str | Path) -> Path:
    """
    Valida si un archivo existe.

    Retorna:
        - Path: ruta validada.

    Lanza:
        - FileExistsError si la ruta existe pero no es un archivo.
        - FileNotFoundError si el archivo no existe.
    """
    path = Path(path)

    if path.is_file():
        return path
    elif path.exists():
        raise FileExistsError(f"La ruta existe pero no es un archivo: {path}")
    else:
        raise FileNotFoundError(f"El archivo no existe: {path}")
