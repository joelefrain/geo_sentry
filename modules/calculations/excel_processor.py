import os
import sys

# Add 'libs' path to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)

from libs.utils.config_variables import SEP_FORMAT
from libs.utils.config_logger import get_logger

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from .excel_handler import ExcelReader
from .data_transformer import DataTransformer
from .config_handler import ConfigHandler
from .path_handler import PathHandler

logger = get_logger("modules.calculations")


class ExcelProcessor:
    """Clase principal para procesar archivos Excel y generar archivos CSV."""

    def __init__(self, config_name: str):
        self.config = ConfigHandler.load_config(config_name)
        self.data_transformer = DataTransformer()

    def process_excel_directory(
        self,
        input_folder: str,
        output_folder_base: str,
        sensor_type: str,
        code: str,
        exclude_sheets: List[str],
        data_config: Dict[str, Any],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Procesa todos los archivos Excel en un directorio."""
        # Validar ruta de entrada
        if not PathHandler.validate_input_path(input_folder):
            logger.info(f"❌ La ruta de entrada no existe: {input_folder}")
            return

        # Obtener rutas de salida usando PathHandler
        output_paths = PathHandler.create_output_paths(
            base_path=Path(output_folder_base), sensor_type=sensor_type
        )

        self._process_data_files(
            input_folder,
            output_paths["process"],
            data_config,
            code,
            exclude_sheets,
            custom_functions,
            selected_attr,
        )

        self._process_location_files(
            input_folder, output_paths["location"], data_config, code, exclude_sheets
        )

    def _process_data_files(
        self,
        input_folder: str,
        output_folder: str,
        data_config: Dict[str, Any],
        code: str,
        exclude_sheets: List[str],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Procesa los archivos Excel para datos de medición."""
        excel_files = self._get_excel_files(input_folder)
        if not excel_files:
            logger.info(f"❌ No se encontraron archivos Excel en {input_folder}")
            return

        for excel_file in excel_files:
            try:
                logger.info(f"\nProcesando {excel_file.name}...")
                reader = ExcelReader(str(excel_file))
                sheets = reader.get_filtered_sheets(exclude_sheets)

                for sheet_name in sheets:
                    df = reader.read_data_frame(
                        sheet_name,
                        data_config["process"]["start_row"],
                        data_config["process"]["columns"],
                        data_config["process"]["column_names"],
                    )

                    # Aplicar atributos seleccionados si están presentes
                    if not df.empty:
                        df = self._apply_transformations(
                            df,
                            excel_file,
                            sheet_name,
                            data_config,
                            custom_functions,
                            selected_attr,
                        )

                        output_path = Path(output_folder) / f"{code}.{sheet_name}.csv"
                        df.to_csv(output_path, sep=SEP_FORMAT, index=False)
                        logger.info(f"✅ Guardado: {output_path}")
                    else:
                        logger.info(
                            f"⚠️ Hoja '{sheet_name}' vacía después de eliminar nulos."
                        )

            except Exception as e:
                logger.warning(f"❌ Error procesando {excel_file.name}: {e}")

    def _process_location_files(
        self,
        input_folder: str,
        output_folder: str,
        data_config: Dict[str, Any],
        code: str,
        exclude_sheets: List[str],
    ) -> None:
        """Procesa los archivos Excel para datos de ubicación."""
        excel_files = self._get_excel_files(input_folder)
        if not excel_files:
            return

        for excel_file in excel_files:
            try:
                logger.info(f"\nProcesando ubicaciones de {excel_file.name}...")
                reader = ExcelReader(str(excel_file))
                sheets = reader.get_filtered_sheets(exclude_sheets)

                for sheet_name in sheets:
                    location_data = {}
                    for attr_name, cell_ref in data_config["location"][
                        "attributes"
                    ].items():
                        value = reader.read_cell_value(sheet_name, cell_ref)
                        if value is not None:
                            location_data[attr_name] = [value]

                    if location_data:
                        df = pd.DataFrame(location_data)
                        output_path = Path(output_folder) / f"{code}.{sheet_name}.csv"
                        df.to_csv(output_path, sep=SEP_FORMAT, index=False)
                        logger.info(f"✅ Guardado: {output_path}")

            except Exception as e:
                logger.warning(
                    f"❌ Error procesando ubicaciones de {excel_file.name}: {e}"
                )

    def _apply_transformations(
        self,
        df: pd.DataFrame,
        excel_file: Path,
        sheet_name: str,
        data_config: Dict[str, Any],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Aplica todas las transformaciones necesarias al DataFrame."""
        if selected_attr:
            df = self._apply_selected_attributes(
                df, excel_file, sheet_name, selected_attr
            )

        if data_config["process"].get("attributes"):
            reader = ExcelReader(str(excel_file))
            values = {
                attr_name: reader.read_cell_value(sheet_name, cell_ref)
                for attr_name, cell_ref in data_config["process"]["attributes"].items()
            }
            constant_functions = self.data_transformer.create_constant_value_functions(
                values
            )
            custom_functions.update(constant_functions)

        # Convert string lambdas to actual functions
        for func_name, func_str in data_config["process"]["custom_functions"].items():
            if isinstance(func_str, str) and func_str.startswith("lambda"):
                custom_functions[func_name] = eval(func_str)
            else:
                custom_functions[func_name] = func_str

        return self.data_transformer.apply_custom_transformations(df, custom_functions)

    def _apply_selected_attributes(
        self,
        df: pd.DataFrame,
        excel_file: Path,
        sheet_name: str,
        selected_attr: Dict[str, Any],
    ) -> pd.DataFrame:
        """Aplica atributos seleccionados al DataFrame y genera registros basados en celdas seleccionadas."""
        reader = ExcelReader(str(excel_file))
        cells = selected_attr.get("cell", [])
        cols = selected_attr.get("column", [])

        if len(cells) == len(cols):
            # Crear un nuevo registro con los valores seleccionados
            new_record = {}
            valid_values = True

            for cell_ref, col in zip(cells, cols):
                value = reader.read_cell_value(sheet_name, cell_ref)
                if value is None:
                    logger.info(f"⚠️ No se pudo obtener el valor de la celda {cell_ref}")
                    valid_values = False
                    break
                new_record[col] = value

            if valid_values and new_record:
                # Crear un DataFrame con el nuevo registro
                new_df = pd.DataFrame([new_record])
                
                # Asegurar que las columnas coincidan
                for col in df.columns:
                    if col not in new_df.columns:
                        new_df[col] = None

                # Asegurar que el nuevo registro tenga todas las columnas necesarias
                for col in new_df.columns:
                    if col not in df.columns:
                        df[col] = None

                # Concatenar el nuevo registro con el DataFrame existente
                df = pd.concat([df, new_df], ignore_index=True)
                
                # Ordenar por tiempo si existe la columna
                if 'time' in df.columns:
                    df = df.sort_values('time', ignore_index=True)
                
                logger.info(f"✅ Nuevo registro agregado con valores de celdas seleccionadas")

        return df

    @staticmethod
    def _get_excel_files(folder_path: str) -> List[Path]:
        """Obtiene la lista de archivos Excel en un directorio."""
        return list(Path(folder_path).glob("**/*.xls*"))