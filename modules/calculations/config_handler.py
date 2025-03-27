import tomli
from pathlib import Path
from typing import Dict, Any, List

class ConfigHandler:
    """Clase para manejar la configuración y validación de datos."""
    
    @staticmethod
    def load_config(config_name: str) -> Dict[str, Any]:
        """Carga la configuración desde un archivo TOML.
        
        Args:
            config_name: Nombre de la configuración (e.g., 'prisms') o ruta completa al archivo TOML.
            
        Returns:
            Diccionario con la configuración.
            
        Raises:
            ValueError: Si el archivo de configuración no existe.
        """
        # Si config_name es una ruta completa, la usamos directamente
        if Path(config_name).suffix == '.toml':
            toml_path = Path(config_name)
        else:
            # Si es solo un nombre, buscamos en el directorio config
            toml_path = Path(__file__).parent / 'config' / f'{config_name}.toml'
            
        if not toml_path.is_file():
            raise ValueError(f"Archivo de configuración '{toml_path}' no encontrado.")
        
        with open(toml_path, "rb") as f:
            return tomli.load(f)
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_sections: List[str]) -> bool:
        """Valida que la configuración contenga las secciones requeridas.
        
        Args:
            config: Diccionario con la configuración.
            required_sections: Lista de secciones requeridas.
            
        Returns:
            True si la configuración es válida, False en caso contrario.
        """
        return all(section in config for section in required_sections)
    
    @staticmethod
    def validate_data_config(data_config: Dict[str, Any]) -> bool:
        """Valida la configuración específica para procesamiento de datos.
        
        Args:
            data_config: Diccionario con la configuración de datos.
            
        Returns:
            True si la configuración es válida, False en caso contrario.
        """
        required_process_fields = ['start_row', 'columns', 'column_names', 'attributes']
        required_location_fields = ['attributes']
        
        process_config = data_config.get('process', {})
        location_config = data_config.get('location', {})
        
        process_valid = all(field in process_config for field in required_process_fields)
        location_valid = all(field in location_config for field in required_location_fields)
        
        return process_valid and location_valid
    
    @staticmethod
    def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Obtiene un valor de la configuración usando una ruta de acceso.
        
        Args:
            config: Diccionario con la configuración.
            path: Ruta de acceso al valor (e.g., 'process.start_row').
            default: Valor por defecto si no se encuentra la ruta.
            
        Returns:
            Valor encontrado o valor por defecto.
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
                
        return value if value is not None else default