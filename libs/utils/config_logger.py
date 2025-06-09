import os
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


def get_logger(module_name):
    # Crear un logger para el módulo específico
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Obtener la fecha actual para crear la carpeta correspondiente
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Crear un directorio de logs si no existe, agrupado por fecha
    log_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../logs", today_date)
    )
    os.makedirs(log_dir, exist_ok=True)

    # Configurar el archivo de log para este módulo dentro de la carpeta de la fecha actual
    log_file = os.path.join(log_dir, f"{module_name}.log")

    # Crear un formateador
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Crear un handler de rotación de logs
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Crear un handler para la consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Agregar los handlers al logger si no existen
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def log_execution_time(func=None, *, module=None):
    """
    Wrapper function to log the execution time of a function.

    Parameters
    ----------
    func : function
        The function to be wrapped.
    module : str, optional
        The module name for the logger.

    Returns
    -------
    function
        The wrapped function with execution time logging.
    """
    if func is None:
        return lambda f: log_execution_time(f, module=module)

    logger = get_logger(module) if module else get_logger("default")

    def wrapper(*args, **kwargs):
        start_time = time.time()
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            logger.info(f"Starting execution of {class_name}.{func.__name__}")
        else:
            logger.info(f"Starting execution of {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if args and hasattr(args[0], "__class__"):
            logger.info(
                f"Finished execution of {class_name}.{func.__name__} in {elapsed_time:.2f} seconds"
            )
        else:
            logger.info(
                f"Finished execution of {func.__name__} in {elapsed_time:.2f} seconds"
            )
        return result

    return wrapper
