import os
import sys
from glob import glob
from pathlib import Path

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from modules.reporter.plot_builder import PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.plot_builder import PlotBuilder

from libs.utils.df_helpers import read_df_on_time_from_csv

def get_map(dxf_path, east, north, sensor_code, size=(2, 2)):
    
    # hace falta convertir east, north y sensor_code en listas separadas si vienen como numeros o string simples
        
    map_plotter_args = {
        "data": [
            {
                "x": east,
                "y": north,
                "color": "red",
                "linestyle": "",
                "linewidth": 0,
                "marker": "o",
                "secondary_y": False,
                "label": "",
                "note": sensor_code,
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

    return PlotMerger.scale_figure(map_draw, size=size)
    
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
    
    dxf_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf"
    
    logo_path = (
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\logo\logo_main.svg"
    )

    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\sample_client\sample_project\processed_data\PTA\DME_CHO.PH-SH23-103A.csv"
    )
    df_2 = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\sample_client\sample_project\processed_data\PTA\DME_CHO.PH-SH23-103B.csv"
    )
    
    sensor_type_name = sensor_names["PTA"]

    start_query = "2024-01-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    east = [808779.55, 808779.55]
    north = [9157518.99, 9157518.99]

    # Este es un grupo de sensores
    data_to_plot = {
        "PTA-SH23-101": df,
        "PTA-SH23-102": df_2,
    }
    
    group_name = "Talud izquierdo"

    # Variables externas que se pasan como kwargs
    context_var = {"location": "Dique sur", "material": "Desmonte"}
    query_var = {"start": start_query, "end": end_query}