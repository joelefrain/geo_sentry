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
    import pandas as pd

    static_report_params = {
        "project_code": "1410.28.0054-0000",
        "company_name": "Shahuindo SAC",
        "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
        "date": "09-04-25",
        "revision": "B",
        "elaborated_by": "J.A.",
        "approved_by": "R.L.",
        "doc_title": DOC_TITLE,
        "theme_color": THEME_COLOR,
        "theme_color_font": THEME_COLOR_FONT,
    }

    sensor_df = pd.read_csv(
        "var\sample_client\sample_project\processed_data\operativity.csv", sep=";"
    )
    sensor_df = sensor_df[(sensor_df["sensor_type"] == "PCT") & (sensor_df["operativiy"] == True)]

    start_item = 1
    appendix = "A"
    output_dir = "./outputs"
    sensor_type = "pct"
    plot_template = "merged_plot_type_01"
    plotter_module = importlib.import_module(f"modules.reporter.data.plotters.{plot_template}")
    generate_report = plotter_module.generate_report

    column_config = {
        "plots": [
            {
                "target_column": "diff_disp_total_abs",
                "unit_target": "cm",
                "ts_serie_flag": True,
                "series_x": "time",
            },
            {
                "target_column": "diff_vert_abs",
                "unit_target": "cm",
                "ts_serie_flag": True,
                "series_x": "time",
            },
            {
                "target_column": "diff_disp_total_abs",
                "unit_target": "cm",
                "ts_serie_flag": True,
                "series_x": "time",
            },
            {
                "target_column": "mean_vel_rel",
                "unit_target": "cm/día",
                "ts_serie_flag": True,
                "series_x": "time",
            },
            {
                "target_column": "inv_mean_vel_rel",
                "unit_target": "día/cm",
                "ts_serie_flag": True,
                "series_x": "time",
            },
            {
                "target_column": "north",
                "ts_serie_flag": False,
                "series_x": "east",
                "serie_aka": "Trayectoria",
            },
        ],
        "sensor_type_name": "punto de control topográfico",
    }

    start_query = "2024-08-01 00:00:00"
    end_query = "2025-04-09 00:00:00"

    # structures = ["PAD_2A", "PAD_2B_2C", "DME_SUR", "DME_CHO"]
    structures = ["DME_CHO"]

    for structure in structures:
        df_structure = sensor_df.groupby("structure").get_group(structure)
        df_structure.dropna(subset=["first_record", "last_record"], inplace=True)
        dxf_path = f"data\\config\\sample_client\\sample_project\\dxf\\{structure}.dxf"
        geo_structure = structure

        for group, df_group in df_structure.groupby("group"):
            
            # Generar listas para data_sensors
            names = df_group["code"].tolist()
            east = df_group["east"].tolist()
            north = df_group["north"].tolist()

            dfs = []
            for code in names:
                csv_path = f"var\\sample_client\\sample_project\\processed_data\\PCT\\{structure}.{code}.csv"
                df_sensor = read_df_on_time_from_csv(csv_path)
                dfs.append(df_sensor)

            print(f"[{structure} - {group} - {names}] Generating PDF...")
            data_sensors = {
                "names": names,
                "east": east,
                "north": north,
                "df": dfs,
            }

            group_args = {
                "name": group,
                "location": structure,
            }

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
            print(f"[{structure} - {group}] Generated PDF: {generated_pdf}")
            n_pdf = len(generated_pdf)
            start_item += n_pdf
