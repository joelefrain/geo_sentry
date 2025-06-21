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
DOC_TITLE = "SIG-AND"
THEME_COLOR = "#0069AA"
THEME_COLOR_FONT = "white"

# Paths to data files
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent

LOG_DIR = BASE_DIR / "logs"
ENV_FILE_PATH = BASE_DIR / "config" / ".env"

LOGO_SVG = BASE_DIR / "data" / "logo" / "logo_main_anddes.svg"
LOGO_PDF = BASE_DIR / "data" / "logo" / "logo_main_anddes.pdf"
DATA_CONFIG = BASE_DIR / "data" / "config"
DXF_COLORS_PATH = BASE_DIR / "data" / "styles" / "dxf_colors.json"
DXF_LINETYPES_PATH = BASE_DIR / "data" / "styles" / "dxf_linetypes.json"

# Paths to configuration directories
# ---------------------------------------------------------------
CALC_CONFIG_DIR = BASE_DIR / "modules" / "calculations" / "data"
REPORT_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "reports"
CHART_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "charts"
NOTE_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "notes"
TABLE_CONFIG_DIR = BASE_DIR / "modules" / "reporter" / "data" / "tables"

OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESS_OUTPUT_DIR = BASE_DIR / "outputs" / "process"
DASHBOARD_OUTPUT_DIR = BASE_DIR / "outputs" / "dashboard"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "outputs" / "analysis"
APPENDIX_OUTPUT_DIR = BASE_DIR / "outputs" / "appendix"

# Configuration for sensors
# ---------------------------------------------------------------
SENSOR_VISUAL_CONFIG = {
    "pcv": {
        "bokeh_marker": "circle_x",
        "mpl_marker": r"$\circ$",
        "color": "skyblue",
    },
    "pta": {
        "bokeh_marker": "circle",
        "mpl_marker": "o",
        "color": "blue",
    },
    "pct": {
        "bokeh_marker": "diamond",
        "mpl_marker": "D",
        "color": "purple",
    },
    "sacv": {
        "bokeh_marker": "triangle",
        "mpl_marker": "^",
        "color": "orange",
    },
    "cpcv": {
        "bokeh_marker": "circle_dot",
        "mpl_marker": r"$\odot$",
        "color": "pink",
    },
    "inc": {
        "bokeh_marker": "square",
        "mpl_marker": "s",
        "color": "green",
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

# Styles for matplotlib
# ---------------------------------------------------------------
UNIQUE_MARKERS = ["o", "s", "D", "v", "^", "<", ">", "p", "h"]
