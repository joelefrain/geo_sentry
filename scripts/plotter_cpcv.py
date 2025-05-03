<<<<<<< HEAD
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

    sensor_df = pd.read_csv(
        "var\sample_client\sample_project\processed_data\operativity.csv", sep=";"
    )
    sensor_df = sensor_df[
        (sensor_df["sensor_type"] == "CPCV") & (sensor_df["operativiy"] == True)
    ]

    start_item = 1
    appendix = "E"
    output_dir = "./outputs/processing"
    sensor_type = "cpcv"
    plot_template = "ts_plot_type_02"
    plotter_module = importlib.import_module(
        f"modules.reporter.data.plotters.{plot_template}"
    )
    generate_report = plotter_module.generate_report

    start_query = "2024-05-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    column_config = {
        "target_column": "pressure_kpa",
        "unit_target": "kPa",
        "primary_column": "",
        "primary_title_y": "Elevación (m s. n. m.)",
        "secondary_column": "pressure_kpa",
        "top_reference_column": "terrain_level",
        "bottom_reference_column": "sensor_level",
        "serie_x": "time",
        "sensor_type_name": "celda de presión",
        "sensor_aka": "La celda de presión",
    }

    for structure_code, structure_name in structure_names.items():

        try:
            df_structure = sensor_df.groupby("structure").get_group(structure_code)
        except KeyError as e:
            print(f"KeyError: {e} not found in the DataFrame. Skipping this entry.")
            continue

        df_structure.dropna(subset=["first_record", "last_record"], inplace=True)
        dxf_path = (
            f"data\\config\\sample_client\\sample_project\\dxf\\{structure_code}.dxf"
        )

        # If group is not assigned, use the sensor code as group
        df_structure["group"] = df_structure["group"].fillna(df_structure["code"])
        for group, df_group in df_structure.groupby("group"):

            # Generar listas para data_sensors
            names = df_group["code"].tolist()
            east = df_group["east"].tolist()
            north = df_group["north"].tolist()
            material = df_group["material"].tolist()[0]

            dfs = []
            for code in names:
                csv_path = f"var\\sample_client\\sample_project\\processed_data\\{sensor_type.upper()}\\{structure_code}.{code}.csv"
                df_sensor = read_df_on_time_from_csv(csv_path, set_index=False)
                dfs.append(df_sensor)

            data_sensors = {
                "names": names,
                "east": east,
                "north": north,
                "df": dfs,
            }

            group_args = {
                "name": group,
                "location": structure_name,
                "material": material,
            }

            generated_pdf = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                dxf_path=dxf_path,
                start_query=start_query,
                end_query=end_query,
                appendix=appendix,
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type,
                output_dir=f"{output_dir}/{structure_code}/{sensor_type}",
                static_report_params=static_report_params,
                column_config=column_config,
            )
            print(f"Generated PDF: {generated_pdf}")
            n_pdf = len(generated_pdf)
            start_item += n_pdf
=======
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

    sensor_df = pd.read_csv(
        "var\sample_client\sample_project\processed_data\operativity.csv", sep=";"
    )
    sensor_df = sensor_df[
        (sensor_df["sensor_type"] == "CPCV") & (sensor_df["operativiy"] == True)
    ]

    start_item = 1
    appendix = "E"
    output_dir = "./outputs/processing"
    sensor_type = "cpcv"
    plot_template = "ts_plot_type_02"
    plotter_module = importlib.import_module(
        f"modules.reporter.data.plotters.{plot_template}"
    )
    generate_report = plotter_module.generate_report

    start_query = "2024-05-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    column_config = {
        "target_column": "pressure_kpa",
        "unit_target": "kPa",
        "primary_column": "",
        "primary_title_y": "Elevación (m s. n. m.)",
        "secondary_column": "pressure_kpa",
        "top_reference_column": "terrain_level",
        "bottom_reference_column": "sensor_level",
        "serie_x": "time",
        "sensor_type_name": "celda de presión",
        "sensor_aka": "La celda de presión",
    }

    for structure_code, structure_name in structure_names.items():

        try:
            df_structure = sensor_df.groupby("structure").get_group(structure_code)
        except KeyError as e:
            print(f"KeyError: {e} not found in the DataFrame. Skipping this entry.")
            continue

        df_structure.dropna(subset=["first_record", "last_record"], inplace=True)
        dxf_path = (
            f"data\\config\\sample_client\\sample_project\\dxf\\{structure_code}.dxf"
        )

        # If group is not assigned, use the sensor code as group
        df_structure["group"] = df_structure["group"].fillna(df_structure["code"])
        for group, df_group in df_structure.groupby("group"):

            # Generar listas para data_sensors
            names = df_group["code"].tolist()
            east = df_group["east"].tolist()
            north = df_group["north"].tolist()
            material = df_group["material"].tolist()[0]

            dfs = []
            for code in names:
                csv_path = f"var\\sample_client\\sample_project\\processed_data\\{sensor_type.upper()}\\{structure_code}.{code}.csv"
                df_sensor = read_df_on_time_from_csv(csv_path, set_index=False)
                dfs.append(df_sensor)

            data_sensors = {
                "names": names,
                "east": east,
                "north": north,
                "df": dfs,
            }

            group_args = {
                "name": group,
                "location": structure_name,
                "material": material,
            }

            generated_pdf = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                dxf_path=dxf_path,
                start_query=start_query,
                end_query=end_query,
                appendix=appendix,
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type,
                output_dir=f"{output_dir}/{structure_code}/{sensor_type}",
                static_report_params=static_report_params,
                column_config=column_config,
            )
            print(f"Generated PDF: {generated_pdf}")
            n_pdf = len(generated_pdf)
            start_item += n_pdf
>>>>>>> 118aabc (update | Independizacion del locale del sistema operativo)
