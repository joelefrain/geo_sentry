import os
import sys
import importlib

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_variables import (
    DOC_TITLE,
    THEME_COLOR,
    THEME_COLOR_FONT,
)

if __name__ == "__main__":
    # Define the module name dynamically

    static_report_params = {
        "project_code": "1410.28.0050",
        "company_name": "Shahuindo SAC",
        "project_name": "Ingeniería de registro, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque",
        "date": "04-04-25",
        "revision": "B",
        "elaborated_by": "J.A.",
        "approved_by": "R.L.",
        "doc_title": DOC_TITLE,
        "theme_color": THEME_COLOR,
        "theme_color_font": THEME_COLOR_FONT,
    }

    appendix = "A"
    start_item = 201

    dxf_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf"

    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\sample_client\sample_project\processed_data\PCV\PAD_2B_2C.PCV-SH23-110-A.csv"
    )
    df_2 = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\sample_client\sample_project\processed_data\PCV\PAD_2B_2C.PCV-SH23-110-B.csv"
    )

    start_query = "2024-05-01 00:00:00"
    end_query = "2025-03-23 00:00:00"
    query_var = {"start": start_query, "end": end_query}

    geo_structure = "DME Choloque"

    data_sensors = {
        "names": ["PCV-SH23-110-A", "PCV-SH23-110-B"],
        "east": [808779.55, 808779.55],
        "north": [9157518.99, 9157518.99],
        "df": [df, df_2],
    }

    group_args = {
        "name": "Talud izquierdo",
        "location": "Dique sur",
        "material": "Desmonte",
    }
    
    output_dir = "./outputs"

    sensor_type = "pcv"
    plot_template = "ts_plot_type_01"


    column_config = {
        "target_column": "hydraulic_load_kpa",
        "unit_target": "kPa",
        "primary_column": "piezometric_level",
        "primary_title_y": "Elevación (m s. n. m.)",
        "secondary_column": "hydraulic_load_kpa",
        "top_reference_column": "terrain_level",
        "bottom_reference_column": "sensor_level",
        "serie_x": "time",
        "sensor_type_name": "piezómetro de cuerda vibrante",
        "sensor_aka": "El piezómetro",
    }

    plotter_module = importlib.import_module(f"modules.reporter.data.plotters.{plot_template}")
    generate_report = plotter_module.generate_report
    
    generated_pdf = generate_report(
        data_sensors=data_sensors,
        group_args=group_args,
        dxf_path=dxf_path,
        start_query=start_query,
        end_query=end_query,
        appendix=appendix,
        start_item=start_item,
        geo_structure=geo_structure,
        sensor_type=sensor_type,
        output_dir=output_dir,
        static_report_params=static_report_params,
        column_config=column_config,
    )
    print(f"Generated PDF: {generated_pdf}")
