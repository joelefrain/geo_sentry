# This file contains configuration variables for the application.
# ---------------------------------------------------------------
from pathlib import Path


# Constants for formats
# ---------------------------------------------------------------
SEP_FORMAT = ";"

# Constants for number formats
DECIMAL_CHAR = ","
THOUSAND_CHAR = " "

# Defaults for the report
# ---------------------------------------------------------------
DOC_TITLE = "SIG-AND"
THEME_COLOR = "#0069AA"
THEME_COLOR_FONT = "white"
COLOR_PALETTE = "RdYlGn"

# Paths to data files
# ---------------------------------------------------------------
LOGO_SVG = Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main_anddes.svg"
LOGO_PDF = Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main_anddes.pdf"

# Paths to configuration directories
# ---------------------------------------------------------------
CALC_CONFIG_DIR = Path(__file__).parent.parent.parent / "modules" / "calculations" / "data"
REPORT_CONFIG_DIR = Path(__file__).parent.parent.parent / "modules" / "reporter" / "data"

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
PROCESS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "process"
PREPROCESS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "preprocess"
DASHBOARD_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "dashboard"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "analysis"

# Visual configuration for sensors in dashboard
# ---------------------------------------------------------------
SENSOR_VISUAL_CONFIG = {
    "PCV": {'marker': 'circle', 'color': 'skyblue'},        # Piezómetro de cuerda vibrante
    "PTA": {'marker': 'circle_hollow', 'color': 'blue'},    # Piezómetro de tubo abierto
    "PCT": {'marker': 'diamond', 'color': 'purple'},        # Punto de control topográfico
    "SACV": {'marker': 'triangle', 'color': 'orange'},      # Celda de asentamiento de cuerda vibrante
    "CPCV": {'marker': 'ellipse', 'color': 'pink'},         # Celda de presión de cuerda vibrante
}