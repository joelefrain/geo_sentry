import os
import sys
import pandas as pd
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.utils.config_loader import load_toml
from libs.utils.config_variables import TABLE_CONFIG_DIR


class TableHandler:
    def __init__(self, style_name="default"):
        """
        Inicializa el manejador de tablas con un estilo específico
        :param style_name: Nombre del archivo de estilo (sin extensión .toml)
        """
        self.style_name = style_name
        self.style_config = load_toml(TABLE_CONFIG_DIR, style_name)

    def create_table(self, df: pd.DataFrame, width=None, height=None):
        """
        Crea una tabla estilizada a partir de un DataFrame y la configuración de estilo
        :param df: DataFrame de pandas
        :param width: Ancho opcional de la tabla
        :param height: Alto opcional de la tabla
        :return: Objeto Table de reportlab
        """
        data = [list(df.columns)] + df.values.tolist()

        table_cfg = self.style_config.get("table", {})
        total_width = width if width is not None else table_cfg.get("width")
        total_height = height if height is not None else table_cfg.get("height")

        ncols = len(df.columns)
        nrows = len(df) + 1
        colWidths = None
        rowHeights = None

        if total_width:
            colWidths = [float(total_width) / ncols] * ncols
        if total_height:
            rowHeights = [float(total_height) / nrows] * nrows

        table = Table(data, colWidths=colWidths, rowHeights=rowHeights)
        style = self._build_table_style(len(df), len(df.columns))
        table.setStyle(style)
        return table

    def _build_table_style(self, nrows, ncols):
        """
        Construye el TableStyle de reportlab a partir de la configuración
        """
        cfg = self.style_config
        ts = []
        # Encabezado
        head_cfg = cfg.get("header", {})
        if head_cfg:
            ts.append(
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    self._parse_color(head_cfg.get("background", "#CCCCCC")),
                )
            )
            ts.append(
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    self._parse_color(head_cfg.get("textColor", "#000000")),
                )
            )
            if "fontName" in head_cfg:
                ts.append(("FONTNAME", (0, 0), (-1, 0), head_cfg["fontName"]))
            if "fontSize" in head_cfg:
                ts.append(("FONTSIZE", (0, 0), (-1, 0), head_cfg["fontSize"]))
        # Cuerpo
        body_cfg = cfg.get("body", {})
        if body_cfg:
            ts.append(
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    self._parse_color(body_cfg.get("background", "#FFFFFF")),
                )
            )
            ts.append(
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, -1),
                    self._parse_color(body_cfg.get("textColor", "#000000")),
                )
            )
            if "fontName" in body_cfg:
                ts.append(("FONTNAME", (0, 1), (-1, -1), body_cfg["fontName"]))
            if "fontSize" in body_cfg:
                ts.append(("FONTSIZE", (0, 1), (-1, -1), body_cfg["fontSize"]))
        # Bordes
        border_cfg = cfg.get("borders", {})
        if border_cfg:
            color = self._parse_color(border_cfg.get("color", "#000000"))
            width = border_cfg.get("width", 1)
            ts.append(("GRID", (0, 0), (-1, -1), width, color))
        # Otros estilos
        for key, value in cfg.get("table", {}).items():
            ts.append((key, (0, 0), (-1, -1), value))
        return TableStyle(ts)

    def _parse_color(self, color_str):
        """
        Convierte un string de color hexadecimal a un objeto de color de reportlab
        """
        if isinstance(color_str, tuple):
            return color_str
        if color_str.startswith("#"):
            color_str = color_str.lstrip("#")
            return colors.HexColor("#" + color_str)
        return getattr(colors, color_str, colors.black)
