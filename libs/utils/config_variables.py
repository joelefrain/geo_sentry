# This file contains configuration variables for the application.
# ---------------------------------------------------------------
from pathlib import Path


# Constants for formats
# ---------------------------------------------------------------
SEP_FORMAT = ";"

# Font for the report
DEFAULT_FONT = "Arial"

# Constants for number formats
DECIMAL_CHAR = ","
THOUSAND_CHAR = " "
DATE_FORMAT = "%d-%m-%y"

# Language settings
LANG_DEFAULT = "es"  # Default language for the application

# Defaults for the report
# ---------------------------------------------------------------
DOC_TITLE = "@joelefrain"
THEME_COLOR = "#0d879c"
THEME_COLOR_FONT = "white"

# Paths to data files
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

LOGO_SVG = BASE_DIR / "data" / "logo" / "logo_main_109x50.svg"
LOGO_PDF = BASE_DIR / "data" / "logo" / "logo_main_109x50.pdf"
DATA_CONFIG = BASE_DIR / "data" / "config"

# Paths to configuration directories
# ---------------------------------------------------------------
CALC_CONFIG_DIR = BASE_DIR / "modules" / "calculations" / "data"
REPORT_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "reports"
CHART_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "charts"
NOTE_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "notes"

OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESS_OUTPUT_DIR = BASE_DIR / "outputs" / "process"
DASHBOARD_OUTPUT_DIR = BASE_DIR / "outputs" / "dashboard"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "outputs" / "analysis"
APPENDIX_OUTPUT_DIR = BASE_DIR / "outputs" / "appendix"

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
    "INC": {
        "marker": "square",
        "color": "green",
        "name": {"es": "Inclinómetro vertical"},
    },
}

# Extensions for file types
# ---------------------------------------------------------------
VALID_TEXT_EXTENSIONS = (".gkn", ".txt", ".csv")

# Allowed characters in formats
# ---------------------------------------------------------------
ALLOWED_TIME_CHARS = r"[^\d\s:/\-.]"
ALLOWED_SEP_CHARS = r"[;,]"
ALLOWD_HEADER_CHARS = r"[^\w()+\-]"

# Minimum records required for processing & plotting
# ---------------------------------------------------------------
MINIMUN_RECORDS = 2