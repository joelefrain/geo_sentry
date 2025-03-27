import os
from pathlib import Path
from typing import Dict, Any


class PathHandler:
    """Clase para manejar la creación y validación de rutas del proyecto."""

    @staticmethod
    def create_output_paths(base_path: str, sensor_type: str) -> Dict[str, str]:
        """Crea y retorna las rutas de salida para los archivos procesados.

        Args:
            base_path: Ruta base del proyecto.
            company_name: Nombre de la compañía.
            project_name: Nombre del proyecto.
            sensor_type: Tipo de sensor.

        Returns:
            Diccionario con las rutas de salida para process y location.
        """

        # Crear rutas específicas para process y location
        output_paths = {
            "process": os.path.join(base_path, "preprocess", sensor_type),
            "location": os.path.join(base_path, "location", sensor_type),
        }

        # Asegurar que los directorios existan
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)

        return output_paths

    @staticmethod
    def validate_input_path(input_path: str) -> bool:
        """Valida que la ruta de entrada exista.

        Args:
            input_path: Ruta a validar.

        Returns:
            True si la ruta existe, False en caso contrario.
        """
        return os.path.exists(input_path)
