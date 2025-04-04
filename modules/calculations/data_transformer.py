import pandas as pd
from typing import Dict, Any, List, Callable

class DataTransformer:
    """Clase para manejar transformaciones de datos y funciones personalizadas."""
    
    @staticmethod
    def apply_custom_transformations(df: pd.DataFrame, custom_functions: Dict[str, Callable]) -> pd.DataFrame:
        """Aplica transformaciones personalizadas al DataFrame."""
        for col_name, func in custom_functions.items():
            df[col_name] = df.apply(func, axis=1)
        return df
    
    @staticmethod
    def create_constant_value_functions(values_dict: Dict[str, Any]) -> Dict[str, Callable]:
        """Crea funciones lambda que devuelven valores constantes."""
        return {attr_name: lambda row, val=value: val 
                for attr_name, value in values_dict.items()}
    
    @staticmethod
    def apply_conditional_values(df: pd.DataFrame, 
                               conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Aplica valores basados en condiciones específicas.
        
        Args:
            df: DataFrame a modificar
            conditions: Lista de diccionarios con las condiciones y valores a aplicar
                       Cada diccionario debe contener:
                       - column: Nombre de la columna a modificar
                       - condition: Función lambda con la condición
                       - value: Valor a aplicar cuando se cumple la condición
        """
        for condition_dict in conditions:
            column = condition_dict.get('column')
            condition_func = condition_dict.get('condition')
            value = condition_dict.get('value')
            
            if all([column, condition_func, value is not None]):
                mask = df.apply(lambda row: condition_func(row), axis=1)
                if mask.any():
                    df.loc[mask, column] = value
        return df
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Valida que el DataFrame contenga las columnas requeridas."""
        return all(col in df.columns for col in required_columns)
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el DataFrame eliminando duplicados y valores nulos."""
        return df.drop_duplicates().dropna()
