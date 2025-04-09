# This file contains configuration variables for the application.
# ---------------------------------------------------------------
from pathlib import Path


# Constants for formats
# ---------------------------------------------------------------
SEP_FORMAT = ";"
DECIMAL_CHAR = ","

# Defaults for the report
# ---------------------------------------------------------------
DOC_TITLE = "@joelefrain"
THEME_COLOR = "#006D77"
THEME_COLOR_FONT = "white"
COLOR_PALETTE = "cool"

# Paths to data files
# ---------------------------------------------------------------
LOGO_SVG = Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main.svg"
LOGO_PDF = Path(__file__).parent.parent.parent / "data" / "logo" / "logo_main.pdf"

# Paths to configuration directories
# ---------------------------------------------------------------
CALC_CONFIG_DIR = Path(__file__).parent.parent.parent / "modules" / "calculations" / "data"
REPORT_CONFIG_DIR = Path(__file__).parent.parent.parent / "modules" / "reporter" / "data"

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
PROCESS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "process"
PREPROCESS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "preprocess"
DASHBOARD_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "dashboard"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "analysis"