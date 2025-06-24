import gc

import pandas as pd

from pathlib import Path
from typing import Dict, List, Any, Optional

from .data_transformer import DataTransformer

from libs.utils.df_helpers import merge_new_records
from libs.utils.validation_helpers import validate_folder

from libs.utils.config_logger import get_logger

logger = get_logger("modules.calculations.excel_parser")


class ExcelParser:
    """Clase principal para procesar archivos Excel y generar archivos CSV."""

    def __init__(self, config: dict):
        self.config = config
        self.data_transformer = DataTransformer()
        self.sheet_data = {}
        self.location_data = pd.DataFrame()

    def parse_excel_dir(
        self,
        input_folder: str,
        exclude_sheets: List[str],
        data_config: Dict[str, Any],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Procesa todos los archivos Excel en un directorio."""
        # Validar ruta de entrada
        try:
            validate_folder(input_folder, create_if_missing=False)
        except FileNotFoundError as e:
            logger.warning(str(e))

        self._preprocess_data_files(
            input_folder,
            data_config,
            exclude_sheets,
            custom_functions,
            selected_attr,
        )

        self._preprocess_location_files(
            input_folder,
            data_config,
            exclude_sheets,
        )

    def _preprocess_data_files(
        self,
        input_folder: str,
        data_config: Dict[str, Any],
        exclude_sheets: List[str],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Procesa los archivos Excel para datos de medición."""

        excel_files = self._get_excel_files(input_folder)
        if not excel_files:
            logger.warning(f"No se encontraron archivos Excel en {input_folder}")
            return

        for excel_file in excel_files:
            try:
                logger.info(f"Procesando {excel_file.name}...")
                reader = ExcelHandler(str(excel_file))
                sheets = reader.get_filtered_sheets(exclude_sheets)

                codes = []
                dfs = []

                for sheet_name in sheets:
                    sheet_transformed_df = reader.read_data_frame(
                        sheet_name,
                        data_config["process"]["start_row"],
                        data_config["process"]["columns"],
                        data_config["process"]["column_names"],
                    )

                    # Aplicar atributos seleccionados si están presentes
                    if not sheet_transformed_df.empty:
                        sheet_transformed_df = self._apply_transformations(
                            sheet_transformed_df,
                            excel_file,
                            sheet_name,
                            data_config,
                            custom_functions,
                            selected_attr,
                        )

                        codes.append(sheet_name)
                        dfs.append(sheet_transformed_df)

                    else:
                        logger.warning(
                            f"Hoja '{sheet_name}' vacía después de eliminar nulos."
                        )

                self.sheet_data.update(dict(zip(codes, dfs)))

            except Exception as e:
                logger.exception(f"Error procesando {excel_file.name}: {e}")

    def _preprocess_location_files(
        self,
        input_folder: str,
        data_config: Dict[str, Any],
        exclude_sheets: List[str],
    ) -> pd.DataFrame:
        """Procesa los archivos Excel para datos de ubicación y los une en un solo DataFrame."""
        excel_files = self._get_excel_files(input_folder)
        if not excel_files:
            raise f"No se encontraron archivos en {input_folder}"

        location_rows = []

        for excel_file in excel_files:
            try:
                logger.info(f"Procesando ubicaciones de {excel_file.name}...")
                reader = ExcelHandler(str(excel_file))
                sheets = reader.get_filtered_sheets(exclude_sheets)

                for sheet_name in sheets:
                    row = {
                        "sheet": sheet_name,
                    }

                    for attr_name, cell_ref in data_config["location"][
                        "attributes"
                    ].items():
                        value = reader.read_cell_value(sheet_name, cell_ref)
                        if value is not None:
                            row[attr_name] = value

                    # Si solo tiene 'sheet', se considera vacía
                    if len(row) > 1:
                        location_rows.append(row)
                    else:
                        logger.warning(
                            f"Hoja '{sheet_name}' vacía después de validación."
                        )

            except Exception as e:
                logger.exception(
                    f"Error procesando ubicaciones de {excel_file.name}: {e}"
                )

        # Crear DataFrame final
        self.location_data = pd.DataFrame(location_rows)

    def _apply_transformations(
        self,
        sheet_read_df: pd.DataFrame,
        excel_file: Path,
        sheet_name: str,
        data_config: Dict[str, Any],
        custom_functions: Dict[str, Any],
        selected_attr: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Aplica todas las transformaciones necesarias al DataFrame."""
        if selected_attr:
            sheet_read_df = self._apply_selected_attributes(
                sheet_read_df, excel_file, sheet_name, selected_attr
            )

        if data_config["process"].get("attributes"):
            reader = ExcelHandler(str(excel_file))
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

        return self.data_transformer.apply_custom_transformations(
            sheet_read_df, custom_functions
        )

    def _apply_selected_attributes(
        self,
        df: pd.DataFrame,
        excel_file: Path,
        sheet_name: str,
        selected_attr: Dict[str, Any],
    ) -> pd.DataFrame:
        """Aplica atributos seleccionados al DataFrame y genera registros basados en celdas seleccionadas."""

        reader = ExcelHandler(str(excel_file))
        cells = selected_attr.get("cell", [])
        cols = selected_attr.get("column", [])

        if len(cells) == len(cols):
            # Crear un nuevo registro con los valores seleccionados
            new_record = {}
            valid_values = True

            for cell_ref, col in zip(cells, cols):
                value = reader.read_cell_value(sheet_name, cell_ref)
                if value is None:
                    logger.warning(
                        f"No se pudo obtener el valor de la celda {cell_ref}"
                    )
                    valid_values = False
                    break
                new_record[col] = value

            if valid_values and new_record:
                # Crear un DataFrame con el nuevo registro
                new_df = pd.DataFrame([new_record])

                # Unir el nuevo registro con el DataFrame existente evitando duplicados
                try:
                    df = merge_new_records(
                        df, new_df, match_columns=["time"], match_type="all"
                    )

                    logger.info(
                        f"Nuevo registro agregado con valores de celdas seleccionadas para {sheet_name}"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error en merge de registros {excel_file.name}: {e}"
                    )

        return df

    @staticmethod
    def _get_excel_files(folder_path: str) -> List[Path]:
        """Obtiene la lista de archivos Excel en un directorio."""
        return list(Path(folder_path).glob("**/*.xls*"))

    def clear_memory(self) -> None:
        """Elimina todos los atributos del objeto para liberar memoria."""
        attributes = list(self.__dict__.keys())
        for attr in attributes:
            try:
                delattr(self, attr)
            except Exception as e:
                logger.exception(
                    f"Advertencia: No se pudo eliminar el atributo {attr}: {e}"
                )
        gc.collect()


class ExcelHandler:
    """Clase para manejar la lectura de archivos Excel de manera eficiente."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.excel_file = pd.ExcelFile(file_path)

    def get_filtered_sheets(self, exclude_sheets: List[str] = []) -> List[str]:
        """Obtiene la lista de hojas filtradas excluyendo las especificadas."""
        return [
            sheet
            for sheet in self.excel_file.sheet_names
            if not any(exclude_text in sheet for exclude_text in exclude_sheets)
        ]

    def read_cell_value(self, sheet_name: str, cell_ref: str) -> Any:
        """Lee el valor de una celda específica de una hoja."""
        col_idx = ord(cell_ref[0].upper()) - ord("A")
        row_idx = int(cell_ref[1:]) - 1

        try:
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=None,
                nrows=row_idx + 1,
                usecols=[col_idx],
            )
            return df.iat[row_idx, 0]
        except Exception:
            return None

    def read_data_frame(
        self,
        sheet_name: str,
        start_row: int,
        columns: List[int],
        column_names: List[str],
    ) -> pd.DataFrame:
        """Lee un DataFrame de una hoja específica con los parámetros dados."""
        try:
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                usecols=columns,
                skiprows=start_row - 1,
                names=column_names,
            )
            return df.dropna()
        except Exception as e:
            logger.exception(f"Error leyendo datos de '{sheet_name}': {e}")
            return pd.DataFrame()
