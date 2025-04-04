import os
import sys


# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import locale
import random
import numpy as np
import pandas as pd

from datetime import datetime
from matplotlib import pyplot as plt

from libs.utils.config_loader import load_toml
from libs.utils.logger_config import get_logger
from modules.reporter.plot_builder import PlotBuilder, PlotMerger
from modules.reporter.report_builder import load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.calculations import round_decimal
from libs.utils.df_helpers import read_df_on_time_from_csv

logger = get_logger("scripts.chart_plotter")
locale.setlocale(locale.LC_TIME, "es_ES.utf8")


def generate_random_color():
    """Generate a random hex color."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def get_palette_colors(n_colors, palette_name="viridis"):
    """Generate n colors from a matplotlib colormap."""
    cmap = plt.colormaps[palette_name]
    colors = [
        f"#{int(x[0]*255):02x}{int(x[1]*255):02x}{int(x[2]*255):02x}"
        for x in cmap(np.linspace(0, 1, n_colors))
    ]
    return colors


def process_dataframes(dfs, df_names, config, start_query, end_query):
    """Process dataframes and generate charts and legends.

    This function is now a wrapper around the ChartProcessor class.

    Parameters
    ----------
    dfs : list
        List of dataframes to process
    df_names : list
        List of dataframe names
    config : dict
        Configuration dictionary from TOML file
    start_query : str
        Start date for filtering
    end_query : str
        End date for filtering

    Returns
    -------
    dict
        Dictionary containing 'charts' and 'legends' lists
    """
    # Use the new ChartProcessor class to handle the chart generation
    from modules.reporter.chart_processor import ChartProcessor

    processor = ChartProcessor(config, start_query, end_query)
    return processor.generate_charts(dfs, df_names)


def generate_pdfs(charts_and_legends, report_params):
    """Generate PDFs for each chart with its corresponding legend.

    This function is now a wrapper around the PDFGenerator class.

    Parameters
    ----------
    charts_and_legends : dict
        Dictionary containing 'charts' and 'legends' lists
    report_params : dict
        Dictionary containing parameters for the report
    """
    # Use the new PDFGenerator class to handle PDF generation
    from modules.reporter.chart_processor import PDFGenerator

    pdf_generator = PDFGenerator(report_params)
    pdf_generator.generate_pdfs(charts_and_legends)


if __name__ == "__main__":
    client_keys = {
        "names": ["Shahuindo SAC", "Shahuindo"],
        "codes": ["Shahuindo_SAC", "Shahuindo"],
    }

    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }

    config = load_toml(
        data_dir=r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\config\Shahuindo_SAC\Shahuindo\charts",
        toml_name="pta",
    )

    # Remove hardcoded configurations and use those from the TOML file
    company_code = client_keys["codes"][0]
    project_code = client_keys["codes"][1]
    company_name = client_keys["names"][0]
    project_name = client_keys["names"][1]
    engineer_code = "1408.10.0050-0000"
    engineer_name = "Ingeniería de registro, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque"
    date_chart = "23-03-25"
    revision = "B"
    appendix_num = "4"
    elaborated_by = "J.A."
    approved_by = "R.L."
    doc_title = "@joelefrain"
    theme_color = "#006D77"
    theme_color_font = "white"
    logo_path = (
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\logo\logo_main.svg"
    )

    sensor_type_name = sensor_names["PTA"]
    sensor_code = "PTA-SH23-101"

    start_query = "2024-01-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    east = 808779.55
    north = 9157518.99

    dxf_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf"

    start_formatted = datetime.strptime(start_query, "%Y-%m-%d %H:%M:%S")
    start_formatted = start_formatted.strftime("%B %Y")
    end_formatted = datetime.strptime(end_query, "%Y-%m-%d %H:%M:%S")
    end_formatted = end_formatted.strftime("%B %Y")

    target = config["target"]["column"]
    unit = config["target"]["unit"]
    target_phrase = config["inline"]["es"]["target_phrase"]

    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\Shahuindo_SAC\Shahuindo\processed_data\PTA\DME_CHO.PH-SH23-103A.csv"
    )
    df_name = "PH-SH23-103A"
    df_2 = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\Shahuindo_SAC\Shahuindo\processed_data\PTA\DME_CHO.PH-SH23-103B.csv"
    )
    df2_name = "PH-SH23-103B"

    # Filtrar datos según el rango de tiempo
    mask = (df["time"] >= pd.to_datetime(start_query)) & (
        df["time"] <= pd.to_datetime(end_query)
    )
    df_filtered = df[mask]

    mask_2 = (df_2["time"] >= pd.to_datetime(start_query)) & (
        df_2["time"] <= pd.to_datetime(end_query)
    )
    df2_filtered = df_2[mask_2]

    # Usar la configuración de notas desde el archivo TOML
    notes_handler = NotesHandler(config["notes"].get("style", "default"))

    # Obtener las secciones de notas desde la configuración TOML
    if (
        "notes" in config
        and "sections" in config["notes"]
        and "items" in config["notes"]["sections"]
    ):
        sections = []

        for item in config["notes"]["sections"]["items"]:
            section = {"title": item["title"], "format_type": item["format_type"]}

            # Procesar el contenido según su tipo
            if isinstance(item["content"], dict) and "data" in item["content"]:
                # Contenido referenciado desde otra parte de la configuración
                if item["content"]["data"] == "config" and "name" in item["content"]:
                    section["content"] = [
                        config["data_config"][item["content"]["name"]]
                    ]
            elif isinstance(item["content"], list):
                # Lista de contenidos con posibles variables a reemplazar
                content_list = []
                for content_item in item["content"]:
                    # Reemplazar variables en el texto
                    formatted_content = content_item.format(
                        total_records=len(df_filtered),
                        start_formatted=start_formatted,
                        end_formatted=end_formatted,
                        avg_diff_time_rel=round_decimal(
                            df_filtered["diff_time_rel"].mean(), 2
                        ),
                        target_phrase=target_phrase,
                        max_value=df_filtered[target].max(),
                        max_time=df_filtered.loc[df_filtered[target].idxmax(), "time"],
                        last_value=df_filtered.iloc[-1][target],
                        last_time=df_filtered.iloc[-1]["time"],
                        unit=unit,
                    )
                    content_list.append(formatted_content)
                section["content"] = content_list

            sections.append(section)
    else:
        # Configuración de notas por defecto si no está en el TOML
        sections = [
            {
                "title": "Ubicación:",
                "content": ["Talud izquierdo"],
                "format_type": "paragraph",
            },
            {
                "title": "Material:",
                "content": ["Desmonte de mina"],
                "format_type": "paragraph",
            },
        ]

    # Crear las notas usando el manejador
    note_paragraph = notes_handler.create_notes(sections)

    map_plotter_args = {
        "data": [
            {
                "x": [east],
                "y": [north],
                "color": "red",
                "linestyle": "",
                "linewidth": 0,
                "marker": "o",
                "secondary_y": False,
                "label": "",
                "note": [sensor_code],
                "fontsize": 15,
                "markersize": 25,
            },
        ],
        "dxf_path": dxf_path,
        "size": (5, 5),
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": True,
        "dxf_params": {
            "color": "black",
            "linestyle": "-",
            "linewidth": 0.02,
        },
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    map_plotter = PlotBuilder(style_file="default")
    map_plotter.plot_series(**map_plotter_args)
    map_plotter.add_arrow(
        data=map_plotter_args["data"],
        position="first",
        angle=0,
        radius=0,
        color="red",
    )
    map_draw = map_plotter.get_drawing()
    map_plotter.close()

    map_draw = PlotMerger.scale_figure(map_draw, size=(1.5, 1.5))

    sample = "chart_landscape_a4_type_01"
    logo_cell = load_svg(logo_path, 0.08)

    # Create report parameters dictionary
    report_params = {
        "sample": sample,
        "theme_color": theme_color,
        "theme_color_font": theme_color_font,
        "logo_cell": logo_cell,
        "note_paragraph": note_paragraph,
        "map_draw": map_draw,
        "engineer_code": engineer_code,
        "company_name": company_name,
        "engineer_name": engineer_name,
        "date_chart": date_chart,
        "revision": revision,
        "elaborated_by": elaborated_by,
        "approved_by": approved_by,
        "doc_title": doc_title,
        "appendix_num": appendix_num,
    }

    # Process dataframes and generate charts and legends
    charts_and_legends = process_dataframes(
        dfs=[df, df_2],
        df_names=[df_name, df2_name],
        config=config,
        start_query=start_query,
        end_query=end_query,
    )

    # Generate PDFs with charts and legends
    generate_pdfs(charts_and_legends, report_params)
