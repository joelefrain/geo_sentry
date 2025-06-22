import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd

from pathlib import Path
from typing import Dict, List, Optional, Callable

from modules.calculations.excel_parser import ExcelParser
from modules.calculations.data_processor import DataProcessor
from modules.calculations.text_processor import text_folder_to_csv
from libs.utils.config_variables import CALC_CONFIG_DIR, BASE_DIR, DATA_CONFIG
from libs.utils.config_loader import load_toml
from libs.utils.config_logger import get_logger, log_execution_time
from libs.utils.df_helpers import (
    read_df_from_csv,
    read_df_on_time_from_csv,
    config_time_df,
    save_df_to_csv,
    merge_new_records,
    assign_params,
)

logger = get_logger("scripts.sensor_processor")


class PathManager:
    """Handles all path-related operations using pathlib."""

    @staticmethod
    def get_work_path(client_code: str, project_code: str) -> Path:
        """Get the working path for a specific client and project."""
        return BASE_DIR / "var" / client_code / project_code

    @staticmethod
    def setup_seed_paths(
        cut_off: str, client_code: str, project_code: str
    ) -> tuple[Path, Path]:
        """Set up seed paths for preprocessing."""
        seed_base_path = BASE_DIR / "seed" / client_code / project_code / cut_off
        config_sensor_path = DATA_CONFIG / client_code / project_code / "reader_format"
        return seed_base_path, config_sensor_path

    @staticmethod
    def iter_path_names(
        base_path: Path, external_path: Dict, internal_path: Dict
    ) -> Dict:
        """Generate nested dictionary of paths combining base with external and internal paths."""
        return {
            key: {
                k: base_path / v / internal_path[key] for k, v in external_path.items()
            }
            for key in internal_path
        }

    @staticmethod
    def check_paths_existence(sensor_data_paths: Dict) -> None:
        """Check existence of sensor data paths."""
        for key, subdict in sensor_data_paths.items():
            for k, path in subdict.items():
                if not path.exists():
                    logger.info(f"{path.absolute()} -> No existe")


class SensorPreprocessor:
    """Handles preprocessing of sensor data from various sources with base_line filtering."""

    def __init__(self, config: Dict):
        self.config = config
        print("*" * 50)
        print(config)
        print("*" * 50)

    def preprocess_sensors(self, **kwargs) -> None:
        """Main preprocessing method that routes to specific processors with base_line filtering."""
        sensor_data_paths = kwargs.get("sensor_data_paths")
        config_sensor_path = kwargs.get("config_sensor_path")
        cut_off = kwargs.get("cut_off")
        exclude_sheets = kwargs.get("exclude_sheets", [])
        custom_functions = kwargs.get("custom_functions", {})
        sensor_codes = kwargs.get("sensor_codes")
        order_structure = kwargs.get("order_structure")
        work_path = kwargs.get("work_path")

        # Add base_line filter to custom functions if not present
        processing_functions = custom_functions.copy()
        if "base_line" not in processing_functions:
            processing_functions["base_line"] = lambda row: False

        for sensor_code in sensor_codes:
            try:
                # Cargar configuración específica para este sensor
                reader_config = load_toml(
                    data_dir=config_sensor_path, toml_name=sensor_code.lower()
                )
                logger.info(
                    f"Config loaded for {sensor_code} from {config_sensor_path}"
                )
                print("/" * 50)
                print(reader_config)
                print("/" * 50)

                # Verificar que la configuración sea válida
                if not reader_config or "type" not in reader_config:
                    logger.error(f"Invalid config for {sensor_code}")
                    continue

            except Exception as e:
                logger.exception(f"Error loading config for {sensor_code}: {e}")
                continue

            processor_method = getattr(
                self, f"_process_{reader_config['type']}", self._process_unknown_type
            )
            processor_method(
                sensor_code=sensor_code,
                reader_config=reader_config,
                sensor_data_paths=sensor_data_paths,
                cut_off=cut_off,
                exclude_sheets=exclude_sheets,
                custom_functions=processing_functions,
                order_structure=order_structure,
                work_path=work_path,
            )

    def _process_excel_processor(self, **kwargs) -> None:
        """Process Excel files with base_line filtering."""
        print("&" * 50) 
        print(kwargs["reader_config"])
        print("&" * 50) 

        processor = ExcelParser(kwargs["reader_config"])

        for structure in kwargs["order_structure"]:
            input_folder = kwargs["sensor_data_paths"][kwargs["sensor_code"]][structure]
            if input_folder.exists():
                output_folder_base = kwargs["work_path"] / kwargs["cut_off"]

                processor.preprocess_excel_directory(
                    input_folder=input_folder,
                    output_folder_base=output_folder_base,
                    sensor_type=kwargs["sensor_code"],
                    code=structure,
                    exclude_sheets=kwargs["exclude_sheets"],
                    data_config=processor.config,
                    custom_functions=kwargs["custom_functions"],
                    selected_attr=processor.config["process"].get("selected_attr"),
                )

    def _process_text_processor(self, **kwargs) -> None:
        """Process text files with base_line filtering."""
        match_columns = kwargs["reader_config"]["process_config"]["match_columns"]

        for structure in kwargs["order_structure"]:
            try:
                input_folder = kwargs["sensor_data_paths"][kwargs["sensor_code"]][
                    structure
                ]
            except KeyError:
                logger.exception(
                    f"Path not found for '{structure}' in '{kwargs['sensor_code']}'"
                )
                continue

            if not input_folder.exists():
                logger.error(f"Folder not found: {input_folder}")
                continue

            output_folder_base = (
                kwargs["work_path"]
                / kwargs["cut_off"]
                / "preprocess"
                / kwargs["sensor_code"]
            )
            output_folder_base.mkdir(parents=True, exist_ok=True)

            for subfolder_name in input_folder.iterdir():
                if subfolder_name.is_dir():
                    sensor_name = subfolder_name.name
                    params = self._get_processor_params(
                        kwargs["reader_config"], structure, sensor_name
                    )

                    df = text_folder_to_csv(subfolder_name, match_columns, **params)
                    df = assign_params(df, **params.get("processor", {}))

                    # Apply base_line filter if available
                    if "base_line" in df.columns:
                        df = df[df["base_line"] | (df["time"] == df["time"].max())]

                    self._save_sensor_data(
                        df, output_folder_base, structure, sensor_name
                    )

    def _process_match_csv_processor(self, **kwargs) -> None:
        """Process CSV files with matching columns and base_line filtering."""
        match_columns = kwargs["reader_config"]["process_config"]["match_columns"]

        for structure in kwargs["order_structure"]:
            try:
                input_folder = kwargs["sensor_data_paths"][kwargs["sensor_code"]][
                    structure
                ]
            except KeyError:
                logger.exception(
                    f"Path not found for '{structure}' in '{kwargs['sensor_code']}'"
                )
                continue

            if not input_folder.exists():
                logger.error(f"Folder not found: {input_folder}")
                continue

            output_folder_base = (
                kwargs["work_path"]
                / kwargs["cut_off"]
                / "preprocess"
                / kwargs["sensor_code"]
            )
            output_folder_base.mkdir(parents=True, exist_ok=True)

            for subfolder_name in input_folder.iterdir():
                if subfolder_name.is_dir():
                    try:
                        sensor_name = subfolder_name.name
                        params = self._get_processor_params(
                            kwargs["reader_config"], structure, sensor_name
                        )
                        folders = params.get("folders", [])
                        all_dfs = []

                        for folder in folders:
                            folder_path = subfolder_name / folder
                            if not folder_path.exists():
                                logger.warning(f"Folder not found: {folder_path}")
                                continue

                            folder_reader_config = params["folder"][folder]
                            df = text_folder_to_csv(
                                folder_path, match_columns, **folder_reader_config
                            )
                            all_dfs.append(df)

                        if all_dfs:
                            final_df = all_dfs[0]
                            for df in all_dfs[1:]:
                                final_df = pd.merge(
                                    final_df, df, on=match_columns, how="outer"
                                )

                            final_df.dropna(inplace=True)
                            final_df = assign_params(
                                final_df, **params.get("processor", {})
                            )

                            # Apply base_line filter if available
                            if "base_line" in final_df.columns:
                                final_df = final_df[
                                    final_df["base_line"]
                                    | (final_df["time"] == final_df["time"].max())
                                ]

                            self._save_sensor_data(
                                final_df, output_folder_base, structure, sensor_name
                            )

                    except Exception as e:
                        logger.exception(f"Error processing {subfolder_name.name}: {e}")

    def _process_unknown_type(self, **kwargs) -> None:
        """Handle unknown processor types."""
        logger.error(f"Unknown processor type: {kwargs['reader_config']['type']}")

    def _get_processor_params(
        self, reader_config: Dict, structure: str, sensor_name: str
    ) -> Dict:
        """Get processor parameters from config with fallback to defaults."""
        if sensor_name in reader_config.get(structure, {}):
            return reader_config[structure][sensor_name]
        return reader_config["default_params"]

    def _save_sensor_data(
        self, df: pd.DataFrame, output_folder: Path, structure: str, sensor_name: str
    ) -> None:
        """Save sensor data to CSV files, creating one file per unique 'code' value.

        Args:
            df: DataFrame with sensor data
            output_folder: Path to save the files
            structure: Structure name for filename
            sensor_name: Sensor name for filename
        """
        if "code" not in df.columns:
            # If no 'code' column, save single file
            file_path = output_folder / f"{structure}.{sensor_name}.csv"
            save_df_to_csv(df=df, file_path=file_path)
            return

        # For each unique code value
        for code_value in df["code"].unique():
            # Filter data for this code
            df_subset = df[df["code"] == code_value].copy()

            # Remove 'code' column before saving
            df_subset = df_subset.drop(columns=["code"])

            # Create filename with structure.sensor_name.code_value format
            file_path = output_folder / f"{structure}.{sensor_name}.{code_value}.csv"
            save_df_to_csv(df=df_subset, file_path=file_path)
            logger.debug(f"Saved data for code {code_value} to {file_path}")


class OperativityManager:
    """Manages operativity data including location information."""

    @staticmethod
    def get_operativity(
        location_data_folder_base: Path,
        work_path: Path,
        sensor_codes: List[str],
        preprocessed_data_folder_base: Path,
    ) -> None:
        """Process and update sensor operativity data.

        Args:
            location_data_folder_base: Path to location data folders
            work_path: Working directory path
            sensor_codes: List of sensor types to process
            preprocessed_data_folder_base: Path to preprocessed data to get sensor names
        """
        logger.info("Starting operativity processing")

        # First get all sensors from preprocessed data
        all_sensors = OperativityManager._get_all_sensors_from_preprocessed(
            preprocessed_data_folder_base, sensor_codes
        )

        # Then get location data for sensors that have it
        location_data = OperativityManager._get_location_data(
            location_data_folder_base, sensor_codes
        )

        # Combine both sources
        combined_df = OperativityManager._combine_sensor_data(
            all_sensors, location_data
        )

        # Setup operativity file path
        operativity_path = work_path / "processed_data" / "operativity.csv"
        operativity_path.parent.mkdir(parents=True, exist_ok=True)

        # Read or create operativity file
        existing_df = OperativityManager._read_or_create_operativity_file(
            operativity_path
        )

        # Merge with new data
        updated_df = merge_new_records(
            existing_df,
            combined_df,
            match_columns=["structure", "sensor_type", "code"],
            match_type="all",
        )

        # Preserve existing non-null values from previous records
        for col in existing_df.columns:
            if col not in ["structure", "sensor_type", "code", "operativiy"]:
                mask = (
                    updated_df["structure"].isin(existing_df["structure"])
                    & updated_df["sensor_type"].isin(existing_df["sensor_type"])
                    & updated_df["code"].isin(existing_df["code"])
                )
                updated_df.loc[mask, col] = existing_df.loc[mask, col]

        # Remove duplicates keeping the most recent
        updated_df = updated_df.drop_duplicates(
            subset=["structure", "sensor_type", "code"], keep="last"
        )

        save_df_to_csv(updated_df, operativity_path)

    @staticmethod
    def _get_all_sensors_from_preprocessed(
        preprocessed_data_folder_base: Path, sensor_codes: List[str]
    ) -> pd.DataFrame:
        """Get all sensors from preprocessed data folders."""
        sensor_records = []

        for sensor_code in sensor_codes:
            sensor_path = preprocessed_data_folder_base / sensor_code
            if not sensor_path.exists():
                continue

            for csv_file in sensor_path.glob("*.csv"):
                try:
                    # File name format: structure.code.csv or structure.code.extra.csv
                    parts = csv_file.stem.split(".")
                    structure = parts[0]
                    code = parts[1]

                    sensor_records.append(
                        {
                            "sensor_type": sensor_code,
                            "structure": structure,
                            "code": code,
                            "operativiy": True,  # Default to True if no location data
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not parse sensor info from {csv_file.name}: {e}"
                    )

        return pd.DataFrame(sensor_records)

    @staticmethod
    def _get_location_data(
        location_data_folder_base: Path, sensor_codes: List[str]
    ) -> pd.DataFrame:
        """Get location data for sensors that have it."""
        location_records = []

        for sensor_code in sensor_codes:
            location_path = location_data_folder_base / sensor_code
            if not location_path.exists():
                continue

            for csv_file in location_path.glob("*.csv"):
                try:
                    # File name format: structure.code.csv or structure.code.extra.csv
                    parts = csv_file.stem.split(".")
                    structure = parts[0]
                    code = parts[1]

                    location_df = read_df_from_csv(csv_file)
                    location_df["sensor_type"] = sensor_code
                    location_df["structure"] = structure
                    location_df["code"] = code
                    location_df["operativiy"] = True

                    location_records.append(location_df)
                except Exception as e:
                    logger.warning(
                        f"Could not process location file {csv_file.name}: {e}"
                    )

        return (
            pd.concat(location_records, ignore_index=True)
            if location_records
            else pd.DataFrame()
        )

    @staticmethod
    def _combine_sensor_data(
        sensors_df: pd.DataFrame, locations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine sensor data from preprocessed files with location data."""
        if locations_df.empty:
            return sensors_df

        # Merge to bring location data to known sensors
        combined_df = pd.merge(
            sensors_df,
            locations_df,
            on=["sensor_type", "structure", "code"],
            how="left",
            suffixes=("", "_loc"),
        )

        # Clean up merged columns
        for col in combined_df.columns:
            if col.endswith("_loc") and col.replace("_loc", "") in sensors_df.columns:
                combined_df[col.replace("_loc", "")] = combined_df[col]
                combined_df.drop(col, axis=1, inplace=True)

        return combined_df

    @staticmethod
    def _read_or_create_operativity_file(file_path: Path) -> pd.DataFrame:
        """Read or create operativity file with default columns."""
        default_columns = [
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

        if file_path.exists():
            df = read_df_from_csv(file_path)
            # Ensure all required columns exist
            for col in default_columns:
                if col not in df.columns:
                    df[col] = None
            return df
        else:
            create_empty_file(file_path)
            return pd.DataFrame(columns=default_columns)


class SensorDataProcessor:
    """Processes sensor data from preprocessed files with reprocessing capability."""

    @staticmethod
    def get_processed_data(
        cut_off: str,
        preprocessed_data_folder_base: Path,
        processed_data_folder_base: Path,
        sensor_codes: List[str],
        reprocess: bool = False,
    ) -> None:
        """Process sensor data and save results with reprocessing option."""
        logger.info(f"Starting data processing for {cut_off} (reprocess={reprocess})")

        def process_data_file(
            preprocessed_csv_path: Path, structure: str, code: str, sensor_type: str
        ) -> None:
            try:
                # Cargar configuración específica para este sensor
                processor = DataProcessor(sensor_type.lower())
                config = processor.config
                print("-" * 50)
                print(config)
                print("-" * 50)

                if not config or "process_config" not in config:
                    logger.error(f"Invalid config for {sensor_type}")
                    return

                match_columns = config["process_config"]["match_columns"]
                overall_columns = config["process_config"]["overall_columns"]

                processed_path = processed_data_folder_base / sensor_type
                processed_csv_path = processed_path / f"{structure}.{code}.csv"
                processed_path.mkdir(parents=True, exist_ok=True)

                logger.info(f"Processing file: {preprocessed_csv_path}")
                create_empty_file(processed_csv_path)

                preprocess_df = read_df_on_time_from_csv(
                    preprocessed_csv_path, set_index=False
                )

                if reprocess:
                    # For reprocessing, start fresh with just the preprocessed data
                    process_df = pd.DataFrame(columns=match_columns)
                else:
                    # For normal processing, read existing processed data
                    process_df = read_or_create_df(
                        processed_csv_path, default_columns=match_columns
                    )

                process_df = config_time_df(process_df, set_index=False)
                temp_df = merge_new_records(
                    process_df,
                    preprocess_df,
                    match_columns=match_columns,
                    match_type="all",
                )

                df = processor.prepare_data(temp_df, match_columns, overall_columns)
                df = processor.process_raw_data(df)
                save_df_to_csv(df, processed_csv_path)

            except Exception as e:
                logger.exception(
                    f"Error processing {sensor_type} file {preprocessed_csv_path}: {e}"
                )

        for sensor_code in sensor_codes:
            preprocessed_path = preprocessed_data_folder_base / sensor_code
            if not preprocessed_path.exists():
                logger.warning(f"No preprocessed data found for {sensor_code}")
                continue

            logger.info(f"Processing data for sensor: {sensor_code}")
            SensorDataProcessor._process_sensor_files(
                sensor_code, preprocessed_path, process_data_file
            )

    @staticmethod
    def _process_sensor_files(
        sensor_type: str, data_path: Path, process_func: Callable
    ) -> None:
        """Process files for a specific sensor type."""
        for csv_file in data_path.glob("*.csv"):
            structure, code = csv_file.stem.split(".")[:2]
            process_func(csv_file, structure, code, sensor_type)


class RecordsUpdater:
    """Updates first, last and max records for each instrument."""

    @staticmethod
    def get_main_records(work_path: Path, sensor_codes: List[str]) -> None:
        """Update instrument records in operativity.csv."""
        logger.info("Updating records in operativity.csv")

        processed_data_folder = work_path / "processed_data"
        operativity_path = processed_data_folder / "operativity.csv"
        operativity_df = read_df_from_csv(operativity_path)

        for sensor_code in sensor_codes:
            try:
                # Cargar configuración específica para este sensor
                config = load_toml(CALC_CONFIG_DIR, f"{sensor_code.lower()}")
                target_column = config.get("target", {}).get("column")

                logger.debug(
                    f"Processing records for {sensor_code} with target column: {target_column}"
                )

            except Exception as e:
                logger.warning(f"Error reading TOML for {sensor_code}: {e}")
                continue

            sensor_folder = processed_data_folder / sensor_code
            if not sensor_folder.exists():
                logger.warning(f"No processed data folder for {sensor_code}")
                continue

            for csv_file in sensor_folder.glob("*.csv"):
                try:
                    structure, code = csv_file.stem.split(".")[:2]
                    RecordsUpdater._update_records_for_file(
                        csv_file,
                        operativity_df,
                        structure,
                        sensor_code,
                        code,
                        target_column,
                    )
                except Exception as e:
                    logger.warning(f"Error processing {csv_file.name}: {str(e)}")

        save_df_to_csv(operativity_df, operativity_path)
        logger.info("Record update completed")

    @staticmethod
    def _update_records_for_file(
        csv_path: Path,
        operativity_df: pd.DataFrame,
        structure: str,
        sensor_code: str,
        code: str,
        target_column: Optional[str],
    ) -> None:
        """Update records for a specific sensor file."""
        df = read_df_from_csv(csv_path)
        if "time" not in df.columns or df.empty:
            return

        last_record = df["time"].max()
        first_record = df["time"].min()
        first_value = last_value = max_value = None

        if target_column and target_column in df.columns:
            # Get first value
            first_row = df[df["time"] == first_record]
            if not first_row.empty:
                first_value = f"{first_row[target_column].iloc[0]}"

            # Get last value
            last_row = df[df["time"] == last_record]
            if not last_row.empty:
                last_value = f"{last_row[target_column].iloc[0]}"

            # Get max value
            max_value = f"{df[target_column].max()}"

        # Find record of max value
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


class SensorProcessor:
    """Main class coordinating the sensor processing workflow."""

    @log_execution_time(module="scripts.sensor_processor")
    def exec_preprocess(self, **kwargs) -> None:
        """Execute sensor data preprocessing from various sources with base_line filtering."""
        custom_functions = kwargs.get("custom_functions", {})
        order_structure = kwargs.get("structure_names", {}).keys()

        seed_base_path, config_sensor_path = PathManager.setup_seed_paths(
            kwargs["cut_off"], kwargs["client_code"], kwargs["project_code"]
        )
        


        sensor_data_paths = PathManager.iter_path_names(
            base_path=seed_base_path,
            external_path=kwargs["structure_names"],
            internal_path=kwargs["sensor_raw_name"],
        )

        PathManager.check_paths_existence(sensor_data_paths)

        preprocessor = SensorPreprocessor(kwargs.get("config", {}))
        preprocessor.preprocess_sensors(
            sensor_data_paths=sensor_data_paths,
            config_sensor_path=config_sensor_path,
            cut_off=kwargs["cut_off"],
            exclude_sheets=kwargs.get("exclude_sheets", []),
            custom_functions=custom_functions,
            sensor_codes=kwargs["sensor_codes"],
            order_structure=order_structure,
            work_path=kwargs["work_path"],
        )

    @log_execution_time(module="scripts.sensor_processor")
    def exec_process(self, **kwargs) -> None:
        """Execute complete sensor data processing with reprocessing option."""
        logger.info(f"Starting complete process for {kwargs['cut_off']}")

        processed_data_folder_base = kwargs["work_path"] / "processed_data"
        preprocessed_data_folder_base = (
            kwargs["work_path"] / kwargs["cut_off"] / "preprocess"
        )
        location_data_folder_base = kwargs["work_path"] / kwargs["cut_off"] / "location"

        # Process operativity (will create entries even without location data)
        OperativityManager.get_operativity(
            location_data_folder_base=location_data_folder_base,
            work_path=kwargs["work_path"],
            sensor_codes=kwargs["sensor_codes"],
            preprocessed_data_folder_base=preprocessed_data_folder_base,
        )

        # Process sensor data with reprocessing option
        SensorDataProcessor.get_processed_data(
            cut_off=kwargs["cut_off"],
            preprocessed_data_folder_base=preprocessed_data_folder_base,
            processed_data_folder_base=processed_data_folder_base,
            sensor_codes=kwargs["sensor_codes"],
            reprocess=kwargs.get("reprocess", False),
        )

    @log_execution_time(module="scripts.sensor_processor")
    def exec_processor(self, **kwargs) -> None:
        """Main processing workflow executor with all requested features."""
        logger.info(f"Starting processing workflow for {kwargs.get('project_code')}")

        try:
            self._validate_input_parameters(**kwargs)
            config = self._load_configuration(**kwargs)

            # Set up work path
            work_path = PathManager.get_work_path(
                kwargs["client_code"], kwargs["project_code"]
            )
            logger.info(f"Work path set to: {work_path}")

            # Prepare common kwargs for processing methods
            process_kwargs = {
                "client_code": kwargs["client_code"],
                "project_code": kwargs["project_code"],
                "structure_names": config.get("structures"),
                "sensor_raw_name": config.get("sensors", {}).get("raw_names"),
                "exclude_sheets": config.get("process", {}).get("exclude_sheets", []),
                "sensor_codes": kwargs["sensor_codes"],
                "work_path": work_path,
                "config": config,
                "reprocess": kwargs.get("reprocess", False),
            }

            # Execute requested methods
            for method in kwargs["methods"]:
                logger.info(f"Executing method: {method}")

                if method == "preprocess":
                    for cut in kwargs["cut_off"]:
                        try:
                            logger.info(f"Preprocessing data for cutoff: {cut}")
                            self.exec_preprocess(cut_off=cut, **process_kwargs)
                        except Exception as e:
                            logger.exception(f"Preprocessing failed for {cut}: {e}")

                elif method == "process":
                    for cut in kwargs["cut_off"]:
                        try:
                            logger.info(f"Processing data for cutoff: {cut}")
                            self.exec_process(cut_off=cut, **process_kwargs)
                        except Exception as e:
                            logger.exception(f"Processing failed for {cut}: {e}")

                elif method == "main_records":
                    try:
                        logger.info("Updating main records")
                        RecordsUpdater.get_main_records(
                            work_path, kwargs["sensor_codes"]
                        )
                    except Exception as e:
                        logger.exception(f"Main records update failed: {e}")

                else:
                    logger.warning(f"Unknown method: {method}")

            logger.info("Processing workflow completed successfully")

        except Exception as e:
            logger.exception(f"Processing workflow failed: {e}")
            raise

    def _validate_input_parameters(self, **kwargs) -> None:
        """Validate input parameters."""
        required_params = [
            "client_code",
            "project_code",
            "cut_off",
            "engineering_code",
            "sensor_codes",
            "methods",
        ]

        if not all(kwargs.get(param) for param in required_params):
            raise ValueError("Missing required parameters")

        if not isinstance(kwargs["cut_off"], list) or not isinstance(
            kwargs["methods"], list
        ):
            raise ValueError("cut_off and methods must be lists")

    def _load_configuration(self, **kwargs) -> Dict:
        """Load and validate configuration."""
        config_dir = (
            DATA_CONFIG / kwargs["client_code"] / kwargs["project_code"] / "processor"
        )
        try:
            config = load_toml(config_dir, kwargs["engineering_code"])
            logger.debug(
                f"Configuration loaded from {config_dir}/{kwargs['engineering_code']}"
            )
        except Exception as e:
            logger.exception(f"Failed to load configuration: {e}")
            raise FileNotFoundError(
                f"Configuration not found: {kwargs['engineering_code']}"
            )

        if not all(
            [config.get("structures"), config.get("sensors", {}).get("raw_names")]
        ):
            raise ValueError("Invalid configuration structure")

        return config


def create_empty_file(file_path: Path) -> None:
    """Create an empty file if it doesn't exist."""
    if not file_path.exists():
        file_path.touch()
        logger.warning(f"File created: {file_path}")


def read_or_create_df(file_path: Path, default_columns: list) -> pd.DataFrame:
    """Read a CSV file or create an empty DataFrame with default columns."""
    try:
        df = read_df_from_csv(file_path)
        logger.info(f"File read: {file_path}")
        return df
    except Exception as e:
        logger.warning(f"Error reading CSV file: {e}")
        return pd.DataFrame(columns=default_columns)


if __name__ == "__main__":
    try:
        processor_params = {
            "client_code": "sample_client",
            "project_code": "sample_project",
            "engineering_code": "eor_2025",
            "cut_off": [
                "250430 Data Monitoreo Anddes ABRIL",
                "250530 Data Monitoreo Anddes MAYO",
            ],
            "methods": ["preprocess", "process", "main_records"],
            "sensor_codes": ["PCV", "PCT", "PTA"],
            "reprocess": False,  # Set to True to force reprocessing
        }

        logger.info(
            "Starting sensor processor with parameters:", extra=processor_params
        )

        processor = SensorProcessor()
        processor.exec_processor(**processor_params)

        logger.info("Sensor processor completed successfully")

    except Exception as e:
        logger.exception(f"Sensor processor failed: {e}")
        sys.exit(1)
