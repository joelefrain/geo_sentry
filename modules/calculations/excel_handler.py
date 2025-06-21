import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.utils.config_variables import SEP_FORMAT
from libs.utils.config_logger import get_logger

import pandas as pd
from typing import Dict, List, Tuple, Any

logger = get_logger("modules.calculations")

class ExcelReader:
    """Clase para manejar la lectura de archivos Excel de manera eficiente."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.excel_file = pd.ExcelFile(file_path)
        
    def get_filtered_sheets(self, exclude_sheets: List[str] = []) -> List[str]:
        """Obtiene la lista de hojas filtradas excluyendo las especificadas."""
        return [sheet for sheet in self.excel_file.sheet_names 
                if not any(exclude_text in sheet for exclude_text in exclude_sheets)]
    
    def read_cell_value(self, sheet_name: str, cell_ref: str) -> Any:
        """Lee el valor de una celda específica de una hoja."""
        col_idx = ord(cell_ref[0].upper()) - ord('A')
        row_idx = int(cell_ref[1:]) - 1
        
        try:
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=None,
                nrows=row_idx + 1,
                usecols=[col_idx]
            )
            return df.iat[row_idx, 0]
        except Exception:
            return None
    
    def read_data_frame(self, sheet_name: str, 
                       start_row: int, columns: List[int], 
                       column_names: List[str]) -> pd.DataFrame:
        """Lee un DataFrame de una hoja específica con los parámetros dados."""
        try:
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                usecols=columns,
                skiprows=start_row - 1,
                names=column_names
            )
            return df.dropna()
        except Exception as e:
            logger.warning(f"⚠️ Error leyendo datos de '{sheet_name}': {e}")
            return pd.DataFrame()

def get_cell_indices(attributes: Dict[str, str]) -> Dict[str, Tuple[int, int]]:
    """Convierte referencias de celdas a índices (col, row)."""
    return {attr_name: (ord(cell_ref[0].upper()) - ord('A'), int(cell_ref[1:]) - 1)
            for attr_name, cell_ref in attributes.items()}

def create_output_folder(folder_path: str) -> None:
    """Crea el directorio de salida si no existe."""
    os.makedirs(folder_path, exist_ok=True)

def save_dataframe_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Guarda un DataFrame a un archivo CSV con el separador especificado."""
    try:
        df.to_csv(output_path, sep=SEP_FORMAT, index=False)
        logger.info(f"Guardado: {output_path}")
    except Exception as e:
        logger.warning(f"❌ Error guardando CSV: {e}")