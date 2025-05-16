# This file contains configuration variables for the application.
# ---------------------------------------------------------------
from pathlib import Path


# Constants for formats
# ---------------------------------------------------------------
SEP_FORMAT = ";"

# Font for the report
DEFAULT_FONT = "sans-serif"

# Constants for number formats
DECIMAL_CHAR = ","
THOUSAND_CHAR = " "
DATE_FORMAT = "%d-%m-%y"

# Language settings
LANG_DEFAULT = "es"  # Default language for the application

# Defaults for the report
# ---------------------------------------------------------------
DOC_TITLE = "SIG-AND"
THEME_COLOR = "#0069AA"
THEME_COLOR_FONT = "white"

# Paths to data files
# ---------------------------------------------------------------
LOGO_SVG = (
    Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main_anddes.svg"
)
LOGO_PDF = (
    Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main_anddes.pdf"
)
DATA_CONFIG = Path(__file__).parent.parent.parent / "data" / "config"

# Paths to configuration directories
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
CALC_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "modules" / "calculations" / "data"
)
REPORT_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "modules" / "reporter" / "data" / "reports"
)
CHART_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "modules" / "reporter" / "data" / "charts"
)
NOTE_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "modules" / "reporter" / "data" / "notes"
)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
PROCESS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "process"
DASHBOARD_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "dashboard"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "analysis"

# Configuration for sensors
# ---------------------------------------------------------------
SENSOR_CONFIG = {
    "PCV": {
        "marker": "circle",
        "color": "skyblue",
        "name": {"es": "Piezómetro de cuerda vibrante"},
    },
    "PTA": {
        "marker": "circle_hollow",
        "color": "blue",
        "name": {"es": "Piezómetro de tubo abierto"},
    },
    "PCT": {
        "marker": "diamond",
        "color": "purple",
        "name": {"es": "Punto de control topográfico"},
    },
    "SACV": {
        "marker": "triangle",
        "color": "orange",
        "name": {"es": "Celda de asentamiento de cuerda vibrante"},
    },
    "CPCV": {
        "marker": "ellipse",
        "color": "pink",
        "name": {"es": "Celda de presión de cuerda vibrante"},
    },
}
