import os
import sys
import locale

# Add 'libs' path to sys.path
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_PATH)

import pandas as pd
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet

from modules.calculations.data_processor import DataProcessor
from libs.utils.logger_config import get_logger, log_execution_time
from modules.reporter.plot_builder import PlotBuilder, PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.calculations import round_decimal
from libs.utils.df_helpers import read_df_on_time_from_csv

logger = get_logger("scripts.chart_plotter")
locale.setlocale(locale.LC_TIME, "es_ES.utf8")

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
    logo_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\logo\logo_main.svg"

    sensor_type_name = sensor_names["PCV"]
    sensor_code = "PCV-SH23-101"
    
    config = DataProcessor("pcv").config
    start_query = "2024-01-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    east = 808779.55
    north = 9157518.99

    dxf_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf"

    start_formatted = datetime.strptime(start_query, "%Y-%m-%d %H:%M:%S")
    start_formatted = start_formatted.strftime("%B %Y")
    end_formatted = datetime.strptime(end_query, "%Y-%m-%d %H:%M:%S")
    end_formatted = end_formatted.strftime("%B %Y")

    column = config["target"]["column"]

    color = config["plot"]["color"]
    linetype = config["plot"]["linetype"]
    lineweight = config["plot"]["lineweight"]
    marker = config["plot"]["marker"]

    label = config["names"]["es"][column]
    title_x = config["names"]["es"]["time"]

    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\Shahuindo_SAC\Shahuindo\processed_data\PCV\DME_CHO.PCV-SH23-101.csv"
    )

    # Filtrar datos según el rango de tiempo
    mask = (df["time"] >= pd.to_datetime(start_query)) & (
        df["time"] <= pd.to_datetime(end_query)
    )
    df_filtered = df[mask]

    total_records = len(df_filtered)
    avg_diff_time_rel = df_filtered["diff_time_rel"].mean()
    avg_diff_time_rel = round_decimal(avg_diff_time_rel, 2)
    last_value, last_time = df_filtered.iloc[-1][[column, "time"]]
    max_value, max_time = df_filtered.loc[df_filtered[column].idxmax(), [column, "time"]]

    note_number_records = f"Se registraron {total_records} lecturas durante el periodo entre {start_formatted} y {end_formatted}."
    note_freq_monit = f"La frecuencia de registro promedio es de {avg_diff_time_rel} días durante el periodo entre {start_formatted} y {end_formatted}."
    note_max_record = f"El valor máximo registrado fue {round_decimal(max_value, 2)} en el día {max_time.strftime('%d-%m-%Y')}."
    note_last_record = f"El último valor registrado fue {round_decimal(last_value, 2)} en el día {last_time.strftime('%d-%m-%Y')}."
    
    notes = [note_number_records, note_freq_monit, note_max_record, note_last_record]
    sections = [
        {'title': 'Notas:', 'content': notes, 'format_type': 'alphabet'},
        {'title': 'Ubicación', 'content': ['Talud izquierdo', 'Esquina derecha'], 'format_type': 'paragraph'}
    ]

    # Crear instancia del manejador de notas con estilo de bullets
    notes_handler = NotesHandler("default")  # Usar el nuevo estilo

    # Donde antes se usaba create_notes directamente, ahora usar:
    note_paragraph = notes_handler.create_notes(sections)

    upper_title = f"Registro de {sensor_type_name.lower()} {sensor_code}"
    lower_title = f"{upper_title} desde {start_formatted} hasta {end_formatted}"

    upper_data = [
        {
            "x": df["time"].tolist(),
            "y": df[column].tolist(),
            "color": color,
            "linetype": linetype,
            "lineweight": lineweight,
            "marker": marker,
            "secondary_y": False,
            "label": label,
        },
    ]

    lower_data = [
        {
            "x": df_filtered["time"].tolist(),
            "y": df_filtered[column].tolist(),
            "color": color,
            "linetype": linetype,
            "lineweight": lineweight,
            "marker": marker,
            "secondary_y": False,
            "label": label,
        },
    ]

    map_plotter_args = {
        "data": [
            {
                "x": [east],
                "y": [north],
                "color": "red",
                "linetype": "",
                "lineweight": 0,
                "marker": "o",
                "secondary_y": False,
                "label": "",
                "note": sensor_code,
                "fontsize": 10,
            },
        ],
        "dxf_path": dxf_path,
        "size": (6, 4),
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
        "markersize": 10,
    }

    upper_plotter_args = {
        "data": upper_data,
        "size": (7.5, 2.8),
        "title_x": "",
        "title_y": label,
        "title_chart": upper_title,
        "show_legend": True,
    }

    lower_plotter_args = {
        "data": lower_data,
        "size": (7.5, 2.8),
        "title_x": title_x,
        "title_y": label,
        "title_chart": lower_title,
    }

    map_plotter = PlotBuilder()
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

    upper_plotter = PlotBuilder()
    upper_plotter.plot_series(**upper_plotter_args)
    upper_draw = upper_plotter.get_drawing()
    upper_plotter.close()

    lower_plotter = PlotBuilder()
    lower_plotter.plot_series(**lower_plotter_args)
    lower_draw = lower_plotter.get_drawing()
    lower_plotter.close()

    grid = PlotMerger(fig_size=(7.5, 6))
    grid.create_grid(2, 1)

    # Añadir objetos con sus posiciones
    grid.add_object(upper_draw, (0, 0))
    grid.add_object(lower_draw, (1, 0))

    # Construir y obtener el objeto svg2rlg final
    chart_draw = grid.build(color_border="white")

    grid = PlotMerger(fig_size=(2, 2))
    grid.create_grid(1, 1)
    grid.add_object(map_draw, (0, 0))
    map_draw = grid.build(color_border="white")

    sample = "chart_landscape_a4_type_01"
    logo_cell = load_svg(logo_path, 0.10)

    num_item = f"{appendix_num}.200"
    pdf_generator = ReportBuilder(
        sample=sample,
        theme_color=theme_color,
        theme_color_font=theme_color_font,
        logo_cell=logo_cell,
        upper_cell=note_paragraph,
        lower_cell=map_draw,
        chart_cell=chart_draw,
        chart_title="PIEZOMETRO DE PRUEBA",
        num_item=num_item,
        project_code=engineer_code,
        company_name=company_name,
        project_name=engineer_name,
        date=date_chart,
        revision=revision,
        elaborated_by=elaborated_by,
        approved_by=approved_by,
        doc_title=doc_title,
    )
    pdf_generator.generate_pdf(pdf_path="PCV_PRUEBA.pdf")
