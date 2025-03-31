import os
import tomli
from typing import Dict, Any

def load_toml(data_dir: str, toml_name: str) -> Dict[str, Any]:
    """
    Load a TOML configuration file from a specific directory.
    
    Args:
        data_dir (str): Base directory path where TOML files are stored
        toml_name (str): Name of the TOML file (with or without extension)
        
    Returns:
        Dict[str, Any]: Parsed TOML content as a dictionary
        
    Raises:
        FileNotFoundError: If the TOML file doesn't exist
        tomli.TOMLDecodeError: If the TOML file is invalid
    """
    # Ensure toml_name has .toml extension
    if not toml_name.endswith('.toml'):
        toml_name += '.toml'
    
    # Build full path
    toml_path = os.path.join(data_dir, toml_name)
    
    try:
        with open(toml_path, 'rb') as f:
            return tomli.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {toml_path}")
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Invalid TOML file {toml_path}: {str(e)}")
