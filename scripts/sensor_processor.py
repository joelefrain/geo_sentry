import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
from pathlib import Path

from modules.calculations.excel_processor import ExcelProcessor
from modules.calculations.data_processor import DataProcessor
from modules.calculations.gkn_processor import gkn_folder_to_csv
from libs.utils.config_variables import CALC_CONFIG_DIR, BASE_DIR, DATA_CONFIG
from libs.utils.config_loader import load_toml
from libs.utils.config_logger import get_logger, log_execution_time
from libs.utils.df_helpers import (
    read_df_from_csv,
    read_df_on_time_from_csv,
    config_time_df,
    save_df_to_csv,
    merge_new_records,
)


logger = get_logger("scripts.sensor_processor")


def get_work_path(client_code, project_code):
    """Obtiene la ruta de trabajo para un corte específico.

    Args:
        work_path: Ruta base de trabajo.
        cut_off: Fecha de corte para el procesamiento.

    Returns:
        Ruta de trabajo para el corte especificado.
    """
    return os.path.join(BASE_DIR, "var", client_code, project_code)


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


def setup_seed_paths(cut_off, client_code, project_code, sensor_names):
    """Configura las rutas necesarias para el preprocesamiento.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        client_code: Código de la compañía.
        project_code: Código del proyecto.
        sensor_names: Diccionario con nombres de sensores.

    Returns:
        tuple: (seed_base_path, config_sensor_path)
    """
    seed_base_path = os.path.join(
        BASE_DIR, f"seed/{client_code}/{project_code}/{cut_off}/"
    )
    config_sensor_path = os.path.join(
        DATA_CONFIG,
        f"{client_code}/{project_code}/reader_format",
    )
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
    sensor_codes,
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
    for sensor_code in sensor_codes:
        try:
            reader_config = load_toml(data_dir=config_sensor_path, toml_name=sensor_code.lower())
            logger.info(
                f"Configuración cargada para {sensor_code} desde {config_sensor_path}"
            )
        except Exception as e:
            logger.error(f"Error al cargar configuración para {sensor_code}: {e}")
            continue

        type_reader = reader_config["type"]
            
        if type_reader == "excel_processor":
            
            # Procesar archivos Excel
            processor = ExcelProcessor(reader_config)

            for structure in order_structure:
                input_folder = sensor_data_paths[sensor_code][structure]
                if os.path.exists(input_folder):
                    output_folder_base = os.path.join(work_path, cut_off)
                    custom_functions_for_sensor = custom_functions.copy()
                    processor.preprocess_excel_directory(
                        input_folder=input_folder,
                        output_folder_base=output_folder_base,
                        sensor_type=sensor_code,
                        code=structure,
                        exclude_sheets=exclude_sheets,
                        data_config=processor.config,
                        custom_functions=custom_functions_for_sensor,
                        selected_attr=processor.config["process"].get("selected_attr"),
                    )
        

        if type_reader == "gkn_processor":
            match_columns = reader_config["process_config"]["match_columns"]

            for structure in order_structure:
                try:
                    input_folder = sensor_data_paths[sensor_code][structure]
                except KeyError:
                    logger.error(f"No se encontró la ruta para '{structure}' en '{sensor_code}'")
                    continue

                if not os.path.exists(input_folder):
                    logger.error(f"Carpeta no encontrada: {input_folder}")
                    continue

                output_folder_base = os.path.join(work_path, cut_off, "preprocess", sensor_code)
                os.makedirs(output_folder_base, exist_ok=True)

                for subfolder_name in os.listdir(input_folder):
                    subfolder = os.path.join(input_folder, subfolder_name)
                    if os.path.isdir(subfolder):
                        sensor_name = subfolder_name

                        # Obtiene parámetros específicos o los por defecto
                        if sensor_name in reader_config.get(structure, {}):
                            params = reader_config[structure][sensor_name]
                        else:
                            params = reader_config["default_params"]

                        df = gkn_folder_to_csv(subfolder, match_columns, **params)

                        file_path = os.path.join(output_folder_base, f"{structure}.{sensor_name}.csv")
                        save_df_to_csv(df=df, file_path=file_path)

@log_execution_time(module="scripts.sensor_processor")
def exec_preprocess(
    cut_off,
    sensor_raw_name,
    exclude_sheets,
    client_code,
    project_code,
    structure_names,
    sensor_codes,
    work_path,
):
    """Ejecuta el preprocesamiento de datos de sensores desde archivos Excel.

    Args:
        cut_off: Fecha de corte para el procesamiento.
        sensor_raw_name: Diccionario con nombres de hojas para cada tipo de sensor.
        exclude_sheets: Lista de nombres de hojas a excluir.
        custom_functions: Diccionario de funciones personalizadas para el procesamiento.
        client_code: Código de la compañía.
        project_code: Código del proyecto.
        structure_names: Diccionario con nombres de estructuras.
        sensor_names: Diccionario con nombres de sensores.
        order_structure: Orden de procesamiento de estructuras.
        order_sensors: Orden de procesamiento de sensores.
        work_path: Ruta de trabajo.
    """
    custom_functions = {"base_line": lambda row: False}
    order_structure = structure_names.keys()

    seed_base_path, config_sensor_path = setup_seed_paths(
        cut_off, client_code, project_code, sensor_codes
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
        sensor_codes,
        order_structure,
        work_path,
    )


def process_sensor_files(sensor_type, data_path, process_func):
    """Procesa los archivos de un tipo de sensor específico.

    Args:
        sensor_type: Tipo de sensor a procesar.
        data_path: Ruta del directorio con los archivos a procesar.
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


def get_operativity(location_data_folder_base, work_path, sensor_codes):
    """Procesa y actualiza los datos de operatividad de los sensores.

    Args:
        location_data_folder_base: Ruta base del directorio con datos de ubicación.
        work_path: Ruta de trabajo.
        sensor_names: Diccionario con nombres de sensores.
    """
    logger.info("Iniciando procesamiento de operatividad")

    def process_location_file(file_path, structure, code, sensor_type):
        location_csv_df = read_df_from_csv(file_path)
        location_csv_df["structure"] = structure
        location_csv_df["sensor_type"] = sensor_type
        location_csv_df["code"] = code
        location_csv_df["operativiy"] = True
        return location_csv_df

    # Procesar nuevos datos de ubicación
    all_locations = []
    for sensor_code in sensor_codes:
        location_path = os.path.join(location_data_folder_base, sensor_code)
        if os.path.exists(location_path):
            sensor_locations = process_sensor_files(
                sensor_code, location_path, process_location_file
            )
            all_locations.extend(sensor_locations)

    new_location_df = (
        pd.concat(all_locations, ignore_index=True) if all_locations else pd.DataFrame()
    )

    # Configurar ruta y asegurar directorio
    operativity_path = os.path.join(work_path, "processed_data", "operativity.csv")
    os.makedirs(os.path.dirname(operativity_path), exist_ok=True)

    # Leer o crear archivo de operatividad
    if not os.path.exists(operativity_path):
        create_empty_file(operativity_path)
        existing_df = pd.DataFrame(
            columns=[
                "structure",
                "sensor_type",
                "code",
                "operativiy",
                "first_record",
                "first_value",
                "last_record",
                "last_value",
                "max_value",
                "max_record",
            ]
        )
    else:
        existing_df = read_df_from_csv(operativity_path)

    # Fusionar datos preservando información existente
    if not new_location_df.empty:
        # Identificar registros nuevos y existentes
        updated_df = merge_new_records(
            existing_df,
            new_location_df,
            match_columns=["structure", "sensor_type", "code"],
            match_type="all",
        )

        # Preservar valores no nulos existentes
        for col in existing_df.columns:
            if col not in ["structure", "sensor_type", "code"]:
                mask = (
                    updated_df["structure"].isin(existing_df["structure"])
                    & updated_df["sensor_type"].isin(existing_df["sensor_type"])
                    & updated_df["code"].isin(existing_df["code"])
                )
                updated_df.loc[mask, col] = existing_df.loc[mask, col]

        # Eliminar duplicados manteniendo la última actualización
        updated_df = updated_df.drop_duplicates(
            subset=["structure", "sensor_type", "code"], keep="last"
        )

        save_df_to_csv(updated_df, operativity_path)
    else:
        save_df_to_csv(existing_df, operativity_path)


def get_processed_data(
    cut_off, preprocessed_data_folder_base, processed_data_folder_base, sensor_codes
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
        processor = DataProcessor(sensor_type.lower())
        config = processor.config
        match_columns = config["process_config"]["match_columns"]

        processed_path = os.path.join(processed_data_folder_base, sensor_type)
        processed_csv_path = os.path.join(processed_path, f"{structure}.{code}.csv")
        os.makedirs(processed_path, exist_ok=True)

        logger.info(f"Procesando archivo: {preprocessed_csv_path}")

        create_empty_file(processed_csv_path)

        preprocess_df = read_df_on_time_from_csv(preprocessed_csv_path, set_index=False)
        process_df = read_or_create_df(
            processed_csv_path, default_columns=match_columns
        )
        process_df = config_time_df(process_df, set_index=False)
        temp_df = merge_new_records(
            process_df, preprocess_df, match_columns=match_columns, match_type="all"
        )

        df = processor.clean_data(temp_df, match_columns)
        df = processor.process_raw_data(df)
        save_df_to_csv(df, processed_csv_path)

    for sensor_code in sensor_codes:
        preprocessed_path = os.path.join(preprocessed_data_folder_base, sensor_code)
        logger.info(f"Procesando datos para el sensor: {sensor_code}")
        process_sensor_files(
            sensor_code,
            preprocessed_path,
            process_data_file,
        )


@log_execution_time(module="scripts.sensor_processor")
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

    get_operativity(location_data_folder_base, work_path, sensor_names)
    get_processed_data(
        cut_off, preprocessed_data_folder_base, processed_data_folder_base, sensor_names
    )


def get_main_records(work_path, sensor_codes):
    """Actualiza los primeros, últimos y máximos registros de cada instrumento."""
    logger.info("Actualizando registros en operativity.csv")

    processed_data_folder = os.path.join(work_path, "processed_data")
    operativity_path = os.path.join(processed_data_folder, "operativity.csv")

    operativity_df = read_df_from_csv(operativity_path)

    for sensor_code in sensor_codes:
        try:
            config = load_toml(CALC_CONFIG_DIR, f"{sensor_code.lower()}")
            target_column = config.get("target", {}).get("column")
        except Exception as e:
            logger.warning(f"Error leyendo TOML para {sensor_code}: {e}")
            continue

        sensor_folder = os.path.join(processed_data_folder, sensor_code)
        if not os.path.exists(sensor_folder):
            continue

        for csv_file in os.listdir(sensor_folder):
            if not csv_file.endswith(".csv"):
                continue

            structure, code = csv_file.split(".")[:2]
            csv_path = os.path.join(sensor_folder, csv_file)

            try:
                df = read_df_from_csv(csv_path)
                if "time" in df.columns and not df.empty:
                    last_record = df["time"].max()
                    first_record = df["time"].min()
                    first_value = last_value = max_value = None

                    if target_column and target_column in df.columns:
                        # Obtener primer valor
                        first_row = df[df["time"] == first_record]
                        if not first_row.empty:
                            first_value = f"{first_row[target_column].iloc[0]}"

                        # Obtener último valor
                        last_row = df[df["time"] == last_record]
                        if not last_row.empty:
                            last_value = f"{last_row[target_column].iloc[0]}"

                        # Obtener valor máximo
                        max_value = f"{df[target_column].max()}"

                    # Encontrar el registro del valor máximo
                    max_record = None
                    if target_column and target_column in df.columns:
                        max_idx = df[target_column].idxmax()
                        if max_idx is not None:
                            max_record = df.loc[max_idx, "time"]

                    mask = (
                        (operativity_df["structure"] == structure)
                        & (operativity_df["sensor_type"] == sensor_code)
                        & (operativity_df["code"] == code)
                    )

                    operativity_df.loc[mask, "first_record"] = first_record
                    operativity_df.loc[mask, "first_value"] = first_value
                    operativity_df.loc[mask, "last_record"] = last_record
                    operativity_df.loc[mask, "last_value"] = last_value
                    operativity_df.loc[mask, "max_record"] = max_record
                    operativity_df.loc[mask, "max_value"] = max_value

            except Exception as e:
                logger.warning(f"Error procesando {csv_file}: {str(e)}")

    save_df_to_csv(operativity_df, operativity_path)
    logger.info("Actualización de registros completada")


@log_execution_time(module="scripts.sensor_processor")
def exec_processor(
    client_code: str,
    project_code: str,
    cut_off: list,
    engineering_code: str,
    sensor_codes: list,
    methods: list,
) -> None:
    """Execute sensor data processing from Excel files.

    This function coordinates the entire processing workflow including preprocessing,
    main processing, and record maintenance for sensor data.

    Args:
        client_code (str): Client identification code
        project_code (str): Project identification code
        cut_off (list): List of processing cutoff dates
        engineering_code (str): Engineering configuration code
        sensor_codes (list): List of sensor types to process
        methods (list): List of processing methods to execute

    Raises:
        ValueError: If required parameters are missing or invalid
        FileNotFoundError: If configuration files cannot be found
    """
    logger.info(f"Starting processing workflow for {project_code}")

    try:
        # Validate input parameters
        if not all(
            [
                client_code,
                project_code,
                cut_off,
                engineering_code,
                sensor_codes,
                methods,
            ]
        ):
            raise ValueError("Missing required parameters")

        if not isinstance(cut_off, list) or not isinstance(methods, list):
            raise ValueError("cut_off and methods must be lists")

        # Load configuration
        config_dir = DATA_CONFIG / client_code / project_code / "processor"
        try:
            config = load_toml(config_dir, engineering_code)
            logger.debug(f"Configuration loaded from {config_dir}/{engineering_code}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise FileNotFoundError(f"Configuration not found: {engineering_code}")

        # Extract configuration
        structure_names = config.get("structures")
        sensor_raw_name = config.get("sensors", {}).get("raw_names")
        exclude_sheets = config.get("process", {}).get("exclude_sheets", [])

        if not all([structure_names, sensor_raw_name]):
            raise ValueError("Invalid configuration structure")

        # Set up work path
        work_path = get_work_path(client_code, project_code)
        logger.info(f"Work path set to: {work_path}")

        # Execute requested methods
        for method in methods:
            logger.info(f"Executing method: {method}")

            if method == "preprocess":
                for cut in cut_off:
                    try:
                        logger.info(f"Preprocessing data for cutoff: {cut}")
                        exec_preprocess(
                            cut,
                            sensor_raw_name,
                            exclude_sheets,
                            client_code,
                            project_code,
                            structure_names,
                            sensor_codes,
                            work_path,
                        )
                    except Exception as e:
                        logger.error(f"Preprocessing failed for {cut}: {e}")

            elif method == "process":
                for cut in cut_off:
                    try:
                        logger.info(f"Processing data for cutoff: {cut}")
                        exec_process(cut, work_path, sensor_codes)
                    except Exception as e:
                        logger.error(f"Processing failed for {cut}: {e}")

            elif method == "main_records":
                try:
                    logger.info("Updating main records")
                    get_main_records(work_path, sensor_codes)
                except Exception as e:
                    logger.error(f"Main records update failed: {e}")

            else:
                logger.warning(f"Unknown method: {method}")

        logger.info("Processing workflow completed successfully")

    except Exception as e:
        logger.error(f"Processing workflow failed: {e}")
        raise


if __name__ == "__main__":
    try:
        processor_params = {
            "client_code": "sample_client",
            "project_code": "sample_project",
            "engineering_code": "eor_2025",
            "cut_off": ["250430_Abril"],
            "methods": ["preprocess", "process", "main_records"],
            # "sensor_codes": ["PCV", "PTA", "PCT", "SACV", "CPCV", "INC"],
            "sensor_codes": ["INC"],

        }

        logger.info(
            "Starting sensor processor with parameters:", extra=processor_params
        )
        exec_processor(**processor_params)
        logger.info("Sensor processor completed successfully")

    except Exception as e:
        logger.error(f"Sensor processor failed: {e}")
        sys.exit(1)
