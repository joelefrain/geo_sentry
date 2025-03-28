import os
import sys

# Add 'libs' path to sys.path
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_PATH)

import pandas as pd

from modules.calculations.excel_processor import ExcelProcessor
from modules.calculations.data_processor import DataProcessor
from libs.utils.logger_config import get_logger, log_execution_time
from libs.utils.df_helpers import (
    read_df_from_csv,
    read_df_on_time_from_csv,
    config_time_df,
    save_df_to_csv,
    merge_new_records,
)


logger = get_logger("scripts.sensor_process")


def get_work_path(company_code, project_code):
    """Obtiene la ruta de trabajo para un corte específico.

    Args:
        work_path: Ruta base de trabajo.
        cut_off: Fecha de corte para el procesamiento.

    Returns:
        Ruta de trabajo para el corte especificado.
    """
    return os.path.join(BASE_PATH, "var", company_code, project_code)


def iter_path_names(base_path, external_path, internal_path):
    """Genera un diccionario anidado de rutas combinando la ruta base con rutas externas e internas.

    Args:
        base_path: Ruta base del directorio.
        external_path: Diccionario de rutas externas.
        internal_path: Diccionario de rutas internas.

    Returns:
        Diccionario anidado con las rutas combinadas.
    """
    return {
        key: {
            k: os.path.join(base_path, v, internal_path[key])
            for k, v in external_path.items()
        }
        for key in internal_path
    }


def setup_seed_paths(cut_off, company_code, project_code, sensor_names):
    """Configura las rutas necesarias para el preprocesamiento.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        company_code: Código de la compañía.
        project_code: Código del proyecto.
        sensor_names: Diccionario con nombres de sensores.

    Returns:
        tuple: (seed_base_path, config_sensor_path)
    """
    seed_base_path = os.path.abspath(os.path.join(BASE_PATH, f"seed/{cut_off}/"))
    config_sensor_path = {
        key: os.path.join(
            BASE_PATH,
            f"data/config/{company_code}/{project_code}/excel_format/{key.lower()}.toml",
        )
        for key in sensor_names
    }
    return seed_base_path, config_sensor_path


def check_seed_paths(sensor_data_paths):
    """Verifica la existencia de las rutas de datos.

    Args:
        sensor_data_paths: Diccionario con las rutas de datos de sensores.
    """
    for key, subdict in sensor_data_paths.items():
        for k, path in subdict.items():
            if not os.path.exists(path):
                logger.info(f"{os.path.abspath(path)} -> No existe")


def preprocess_sensors(
    sensor_data_paths,
    config_sensor_path,
    cut_off,
    exclude_sheets,
    custom_functions,
    order_sensors,
    order_structure,
    work_path,
):
    """Procesa los datos de sensores para cada estructura.

    Args:
        sensor_data_paths: Diccionario con las rutas de datos de sensores.
        config_sensor_path: Diccionario con las rutas de configuración.
        cut_off: Fecha de corte para el procesamiento.
        exclude_sheets: Lista de nombres de hojas a excluir.
        custom_functions: Diccionario de funciones personalizadas.
        order_sensors: Orden de procesamiento de sensores.
        order_structure: Orden de procesamiento de estructuras.
        work_path: Ruta de trabajo.
    """
    for sensor_type in order_sensors:
        toml_path = config_sensor_path[sensor_type]
        if not os.path.exists(toml_path):
            logger.info(f"Config file not found: {toml_path}")
            continue

        processor = ExcelProcessor(toml_path)

        for structure in order_structure:
            input_folder = sensor_data_paths[sensor_type][structure]
            if os.path.exists(input_folder):
                output_folder_base = os.path.join(work_path, cut_off)
                custom_functions_for_sensor = custom_functions.copy()
                processor.process_excel_directory(
                    input_folder=input_folder,
                    output_folder_base=output_folder_base,
                    sensor_type=sensor_type,
                    code=structure,
                    exclude_sheets=exclude_sheets,
                    data_config=processor.config,
                    custom_functions=custom_functions_for_sensor,
                    selected_attr=processor.config["process"].get("selected_attr"),
                )


@log_execution_time(module="scripts.sensor_process")
def exec_preprocess(
    cut_off,
    sensor_raw_name,
    exclude_sheets,
    custom_functions,
    company_code,
    project_code,
    structure_names,
    sensor_names,
    order_structure,
    order_sensors,
    work_path,
):
    """Ejecuta el preprocesamiento de datos de sensores desde archivos Excel.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        sensor_raw_name: Diccionario con nombres de hojas para cada tipo de sensor.
        exclude_sheets: Lista de nombres de hojas a excluir.
        custom_functions: Diccionario de funciones personalizadas para el procesamiento.
        company_code: Código de la compañía.
        project_code: Código del proyecto.
        structure_names: Diccionario con nombres de estructuras.
        sensor_names: Diccionario con nombres de sensores.
        order_structure: Orden de procesamiento de estructuras.
        order_sensors: Orden de procesamiento de sensores.
        work_path: Ruta de trabajo.
    """
    seed_base_path, config_sensor_path = setup_seed_paths(
        cut_off, company_code, project_code, sensor_names
    )

    sensor_data_paths = iter_path_names(
        base_path=seed_base_path,
        external_path=structure_names,
        internal_path=sensor_raw_name,
    )

    check_seed_paths(sensor_data_paths)
    preprocess_sensors(
        sensor_data_paths,
        config_sensor_path,
        cut_off,
        exclude_sheets,
        custom_functions,
        order_sensors,
        order_structure,
        work_path,
    )


def process_sensor_files(sensor_type, data_path, output_path, process_func):
    """Procesa los archivos de un tipo de sensor específico.

    Args:
        sensor_type: Tipo de sensor a procesar.
        data_path: Ruta del directorio con los archivos a procesar.
        output_path: Ruta del directorio de salida.
        process_func: Función de procesamiento a aplicar.

    Returns:
        Lista con los resultados del procesamiento.
    """
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    results = []

    for csv_file in csv_files:
        structure, code = csv_file.split(".")[:2]
        input_path = os.path.join(data_path, csv_file)
        result = process_func(input_path, structure, code, sensor_type)
        if result is not None:
            results.append(result)

    return results


def create_empty_file(file_path):
    """Crea un archivo vacío si no existe y registra la acción."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
        logger.warning(f"Archivo creado: {file_path}")


def read_or_create_df(file_path: str, default_columns: list) -> pd.DataFrame:
    """Lee un archivo CSV o crea un DataFrame vacío con columnas predeterminadas.

    Args:
        file_path: Ruta del archivo CSV a leer.
        default_columns: Lista de columnas para el DataFrame vacío en caso de error.

    Returns:
        pd.DataFrame: DataFrame con los datos del CSV o vacío con las columnas especificadas.
    """
    try:
        df = read_df_from_csv(file_path)
        logger.info(f"Archivo leído: {file_path}")
        return df
    except Exception as e:
        logger.warning(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame(columns=default_columns)


def get_operativity(cut_off, location_data_folder_base, work_path, sensor_names):
    """Procesa y actualiza los datos de operatividad de los sensores.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        location_data_folder_base: Ruta base del directorio con datos de ubicación.
        work_path: Ruta de trabajo.
        sensor_names: Diccionario con nombres de sensores.
    """
    logger.info(f"Iniciando procesamiento de operatividad para {cut_off}")

    def process_location_file(file_path, structure, code, sensor_type):
        location_csv_df = read_df_from_csv(file_path)
        location_csv_df["structure"] = structure
        location_csv_df["sensor_type"] = sensor_type
        location_csv_df["code"] = code
        location_csv_df["operativiy"] = True
        return location_csv_df

    all_locations = []
    for sensor_type in sensor_names.keys():
        location_path = os.path.join(location_data_folder_base, sensor_type)
        sensor_locations = process_sensor_files(
            sensor_type, location_path, None, process_location_file
        )
        all_locations.extend(sensor_locations)

    location_df = (
        pd.concat(all_locations, ignore_index=True) if all_locations else pd.DataFrame()
    )
    print(location_df)
    operativity_path = os.path.join(work_path, "processed_data", "operativity.csv")
    os.makedirs(os.path.dirname(operativity_path), exist_ok=True)

    create_empty_file(operativity_path)

    operativity_df = read_or_create_df(
        operativity_path, default_columns=["structure", "sensor_type", "code"]
    )
    print(operativity_df)

    updated_df = merge_new_records(
        operativity_df, location_df, match_columns=["structure", "sensor_type", "code"], match_type='all'
    )
    updated_df.drop_duplicates(subset=["structure", "sensor_type", "code"])
    print(updated_df)
    save_df_to_csv(updated_df, operativity_path)


def get_processed_data(
    cut_off, preprocessed_data_folder_base, processed_data_folder_base, sensor_names
):
    """Procesa los datos de los sensores y guarda los resultados.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        preprocessed_data_folder_base: Ruta base de los datos preprocesados.
        processed_data_folder_base: Ruta base para guardar los datos procesados.
        sensor_names: Diccionario con nombres de sensores.
    """
    logger.info(f"Iniciando procesamiento de datos para {cut_off}")

    def process_data_file(preprocessed_csv_path, structure, code, sensor_type):
        processed_path = os.path.join(processed_data_folder_base, sensor_type)
        processed_csv_path = os.path.join(processed_path, f"{structure}.{code}.csv")
        os.makedirs(processed_path, exist_ok=True)

        logger.info(f"Procesando archivo: {preprocessed_csv_path}")

        create_empty_file(processed_csv_path)

        preprocess_df = read_df_on_time_from_csv(preprocessed_csv_path)

        process_df = read_or_create_df(processed_csv_path, default_columns=["time"])

        process_df = config_time_df(process_df)
        temp_df = merge_new_records(process_df, preprocess_df, match_columns=["time"])

        processor = DataProcessor(sensor_type.lower())
        df = processor.clean_data(temp_df)
        df = processor.process_raw_data(df)
        save_df_to_csv(df, processed_csv_path)

    for sensor_type in sensor_names.keys():
        preprocessed_path = os.path.join(preprocessed_data_folder_base, sensor_type)
        process_sensor_files(
            sensor_type,
            preprocessed_path,
            processed_data_folder_base,
            process_data_file,
        )


@log_execution_time(module="scripts.sensor_process")
def exec_process(cut_off, work_path, sensor_names):
    """Ejecuta el procesamiento completo de datos de sensores.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        work_path: Ruta de trabajo.
        sensor_names: Diccionario con nombres de sensores.
    """
    logger.info(f"Iniciando proceso completo para {cut_off}")
    processed_data_folder_base = os.path.join(work_path, "processed_data")
    preprocessed_data_folder_base = os.path.join(work_path, cut_off, "preprocess")
    location_data_folder_base = os.path.join(work_path, cut_off, "location")

    get_operativity(cut_off, location_data_folder_base, work_path, sensor_names)
    get_processed_data(
        cut_off, preprocessed_data_folder_base, processed_data_folder_base, sensor_names
    )


if __name__ == "__main__":
    client_keys = {
        "names": ["Shahuindo SAC", "Shahuindo"],
        "codes": ["Shahuindo_SAC", "Shahuindo"],
    }

    company_code = client_keys["codes"][0]
    project_code = client_keys["codes"][1]

    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }

    order_structure = structure_names.keys()
    order_sensors = sensor_names.keys()

    cut_off = "250228_Febrero"

    sensor_raw_name = {
        "PCV": "PZ CUERDA VIBRANTE",
        "PTA": "PZ CASAGRANDE",
        "PCT": "PRISMAS",
        "SACV": "CELDAS DE ASENTAMIENTO",
        "CPCV": "CELDAS DE PRESIÓN",
    }

    exclude_sheets = ["Hoja", "Kangatang", "X"]
    custom_functions = {"base_line": lambda row: False}

    work_path = get_work_path(company_code, project_code)

    exec_preprocess(
        cut_off,
        sensor_raw_name,
        exclude_sheets,
        custom_functions,
        company_code,
        project_code,
        structure_names,
        sensor_names,
        order_structure,
        order_sensors,
        work_path,
    )

    exec_process(cut_off, work_path, sensor_names)