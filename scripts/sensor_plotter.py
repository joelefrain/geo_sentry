<<<<<<< HEAD
import os
import sys
import importlib

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd

from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_variables import (
    DOC_TITLE,
    THEME_COLOR,
    THEME_COLOR_FONT,
)


def get_sensor_config(sensor_type):
    """Get the configuration for a specific sensor type.

    Args:
        sensor_type (str): The sensor type code (PCV, PTA, PCT, SACV, CPCV)

    Returns:
        dict: Configuration dictionary for the sensor type
    """
    appendix_chapter = "5"
    configs = {
        "pcv": {
            "plot_template": "ts_plot_type_01",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "hydraulic_load_kpa",
                "unit_target": "kPa",
                "serie_x": "time",
                "sensor_type_name": "piezómetro de cuerda vibrante",
                "sensor_aka": "El piezómetro",
            },
        },
        "pta": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "piezometric_level",
                "unit_target": "m s. n. m.",
                "primary_column": "piezometric_level",
                "primary_title_y": "Elevación (m s. n. m.)",
                "secondary_column": "water_height",
                "top_reference_column": "terrain_level",
                "bottom_reference_column": "bottom_well_elevation",
                "serie_x": "time",
                "sensor_type_name": "piezómetro de tubo abierto",
                "sensor_aka": "El piezómetro",
            },
        },
        "pct": {
            "plot_template": "merged_plot_type_01",
            "appendix": appendix_chapter,
            "column_config": {
                "plots": [
                    {
                        "target_column": "diff_horz_abs",
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
            },
        },
        "sacv": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "settlement_m",
                "unit_target": "m",
                "primary_column": "",
                "primary_title_y": "Elevación (m s. n. m.)",
                "secondary_column": "settlement_m",
                "top_reference_column": "terrain_level",
                "bottom_reference_column": "sensor_level",
                "serie_x": "time",
                "sensor_type_name": "celda de asentamiento",
                "sensor_aka": "La celda de asentamiento",
            },
        },
        "cpcv": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
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
            },
        },
    }

    return configs.get(sensor_type.lower(), {})


def generate_structure_plots(
    structure_code,
    structure_name,
    sensors,
    start_item=1,
    output_dir="./outputs/processing",
    start_query="2024-05-01 00:00:00",
    end_query="2025-03-23 00:00:00",
    static_report_params=None,
):
    """Generate plots for a specific structure with all requested sensor types.

    Args:
        structure_code (str): The structure code (PAD_1A, PAD_2A, etc.)
        structure_name (str): The structure name (Pad 1A, Pad 2A, etc.)
        sensors (list): List of sensor types to process
        start_item (int): Starting item number for the report
        output_dir (str): Output directory for the generated PDFs
        start_query (str): Start date for the query
        end_query (str): End date for the query
        static_report_params (dict): Static report parameters

    Returns:
        int: The next start item number
    """
    # Set default static report parameters if not provided
    if static_report_params is None:
        static_report_params = {
            "project_code": "1410.28.0054-0000",
            "company_name": "Shahuindo SAC",
            "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
            "date": "15-04-25",
            "revision": "B",
            "elaborated_by": "J.A.",
            "approved_by": "R.L.",
            "doc_title": DOC_TITLE,
            "theme_color": THEME_COLOR,
            "theme_color_font": THEME_COLOR_FONT,
        }

    # Load operativity data once
    operativity_df = pd.read_csv(
        "var/sample_client/sample_project/processed_data/operativity.csv", sep=";"
    )
    
    # DXF path for the structure
    dxf_path = f"data/config/sample_client/sample_project/dxf/{structure_code}.dxf"
    
    print(f"\nProcessing structure: {structure_name} ({structure_code})...")
    
    # Process each sensor type for this structure
    for sensor_type in sensors:
        print(f"  Processing sensor type: {sensor_type}...")
        
        # Get sensor configuration
        sensor_config = get_sensor_config(sensor_type)
        if not sensor_config:
            print(f"  Error: Configuration for sensor type '{sensor_type}' not found.")
            continue
            
        # Filter operativity data for this structure and sensor type
        sensor_df = operativity_df[
            (operativity_df["structure"] == structure_code) &
            (operativity_df["sensor_type"] == sensor_type.upper()) &
            (operativity_df["operativiy"] == True)
        ]
        
        # Skip if no sensors of this type for this structure
        if sensor_df.empty:
            print(f"  No {sensor_type} sensors found for {structure_name}.")
            continue
            
        # Drop sensors with missing records
        sensor_df.dropna(subset=["first_record", "last_record"], inplace=True)
        
        # Import plotter module
        plot_template = sensor_config["plot_template"]
        try:
            plotter_module = importlib.import_module(
                f"modules.reporter.data.plotters.{plot_template}"
            )
            generate_report = plotter_module.generate_report
        except ImportError as e:
            print(f"  Error importing plotter module: {e}")
            continue
        
        # If group is not assigned, use the sensor code as group
        sensor_df["group"] = sensor_df["group"].fillna(sensor_df["code"])
        
        # Process each group
        for group, df_group in sensor_df.groupby("group"):
            # Generate lists for data_sensors
            names = df_group["code"].tolist()
            east = df_group["east"].tolist()
            north = df_group["north"].tolist()
            
            # Get material if available
            material = None
            if "material" in df_group.columns:
                material = df_group["material"].tolist()[0]
                
            # Load data for each sensor
            dfs = []
            for code in names:
                csv_path = f"var/sample_client/sample_project/processed_data/{sensor_type.upper()}/{structure_code}.{code}.csv"
                try:
                    df_sensor = read_df_on_time_from_csv(csv_path, set_index=False)
                    dfs.append(df_sensor)
                except Exception as e:
                    print(f"  Error reading data for sensor {code}: {e}")
                    continue
                    
            # Skip if no data loaded
            if not dfs:
                print(f"  No data loaded for group {group}. Skipping.")
                continue
                
            # Prepare data_sensors dictionary
            data_sensors = {
                "names": names,
                "east": east,
                "north": north,
                "df": dfs,
            }
            
            # Prepare group_args dictionary
            group_args = {
                "name": group,
                "location": structure_name,
            }
            
            # Generate report
            print(f"  [{structure_code} - {sensor_type} - {group}] Generating PDF...")
            
            # Ensure output directory exists
            os.makedirs(f"{output_dir}/{structure_code}/{sensor_type.lower()}", exist_ok=True)
            
            generated_pdf = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                dxf_path=dxf_path,
                start_query=start_query,
                end_query=end_query,
                appendix=sensor_config["appendix"],
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type.lower(),
                output_dir=f"{output_dir}/{structure_code}/{sensor_type.lower()}",
                static_report_params=static_report_params,
                column_config=sensor_config["column_config"],
            )
            
            print(f"  [{structure_code} - {sensor_type} - {group}] Generated PDF: {generated_pdf}")
            n_pdf = len(generated_pdf)
            start_item += n_pdf
            
    return start_item


def main(
    sensors=None,
    output_dir="./outputs/processing",
    start_date="2024-05-01 00:00:00",
    end_date="2025-03-23 00:00:00",
):
    """Main function to run the sensor plotter.

    Args:
        sensors (list): List of sensor types to process. If None, all sensors will be processed.
        output_dir (str): Output directory for generated PDFs.
        start_date (str): Start date for the query (YYYY-MM-DD HH:MM:SS).
        end_date (str): End date for the query (YYYY-MM-DD HH:MM:SS).
    """
    # Define sensor names and their descriptions
    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }

    # If no sensors specified, use all available sensors
    if sensors is None:
        sensors = list(sensor_names.keys())

    # Structure names mapping
    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    # Set static report parameters
    static_report_params = {
        "project_code": "1410.28.0054-0000",
        "company_name": "Shahuindo SAC",
        "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
        "date": "15-04-25",
        "revision": "B",
        "elaborated_by": "J.A.",
        "approved_by": "R.L.",
        "doc_title": DOC_TITLE,
        "theme_color": THEME_COLOR,
        "theme_color_font": THEME_COLOR_FONT,
    }

    # Process each structure first, then each sensor type within the structure
    start_item = 1
    for structure_code, structure_name in structure_names.items():
        start_item = generate_structure_plots(
            structure_code=structure_code,
            structure_name=structure_name,
            sensors=sensors,
            start_item=start_item,
            output_dir=output_dir,
            start_query=start_date,
            end_query=end_date,
            static_report_params=static_report_params,
        )

    print(f"\nAll structures and sensors processed. Final item number: {start_item-1}")


if __name__ == "__main__":
    # You can directly call main with specific parameters
    # Example: main(sensors=["PCV", "PTA"], output_dir="./custom_output", start_date="2024-06-01 00:00:00", end_date="2025-01-01 00:00:00")
    # Or use default parameters by calling without arguments
=======
import os
import sys
import importlib

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd

from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_variables import (
    DOC_TITLE,
    THEME_COLOR,
    THEME_COLOR_FONT,
)


def get_sensor_config(sensor_type):
    """Get the configuration for a specific sensor type.

    Args:
        sensor_type (str): The sensor type code (PCV, PTA, PCT, SACV, CPCV)

    Returns:
        dict: Configuration dictionary for the sensor type
    """
    appendix_chapter = "5"
    configs = {
        "pcv": {
            "plot_template": "ts_plot_type_01",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "hydraulic_load_kpa",
                "unit_target": "kPa",
                "serie_x": "time",
                "sensor_type_name": "piezómetro de cuerda vibrante",
                "sensor_aka": "El piezómetro",
            },
        },
        "pta": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "piezometric_level",
                "unit_target": "m s. n. m.",
                "primary_column": "piezometric_level",
                "primary_title_y": "Elevación (m s. n. m.)",
                "secondary_column": "water_height",
                "top_reference_column": "terrain_level",
                "bottom_reference_column": "bottom_well_elevation",
                "serie_x": "time",
                "sensor_type_name": "piezómetro de tubo abierto",
                "sensor_aka": "El piezómetro",
            },
        },
        "pct": {
            "plot_template": "merged_plot_type_01",
            "appendix": appendix_chapter,
            "column_config": {
                "plots": [
                    {
                        "target_column": "diff_horz_abs",
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
            },
        },
        "sacv": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
                "target_column": "settlement_m",
                "unit_target": "m",
                "primary_column": "",
                "primary_title_y": "Elevación (m s. n. m.)",
                "secondary_column": "settlement_m",
                "top_reference_column": "terrain_level",
                "bottom_reference_column": "sensor_level",
                "serie_x": "time",
                "sensor_type_name": "celda de asentamiento",
                "sensor_aka": "La celda de asentamiento",
            },
        },
        "cpcv": {
            "plot_template": "ts_plot_type_02",
            "appendix": appendix_chapter,
            "column_config": {
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
            },
        },
    }

    return configs.get(sensor_type.lower(), {})


def generate_structure_plots(
    structure_code,
    structure_name,
    sensors,
    start_item=1,
    output_dir="./outputs/processing",
    start_query="2024-05-01 00:00:00",
    end_query="2025-03-23 00:00:00",
    static_report_params=None,
):
    """Generate plots for a specific structure with all requested sensor types.

    Args:
        structure_code (str): The structure code (PAD_1A, PAD_2A, etc.)
        structure_name (str): The structure name (Pad 1A, Pad 2A, etc.)
        sensors (list): List of sensor types to process
        start_item (int): Starting item number for the report
        output_dir (str): Output directory for the generated PDFs
        start_query (str): Start date for the query
        end_query (str): End date for the query
        static_report_params (dict): Static report parameters

    Returns:
        int: The next start item number
    """
    # Set default static report parameters if not provided
    if static_report_params is None:
        static_report_params = {
            "project_code": "1410.28.0054-0000",
            "company_name": "Shahuindo SAC",
            "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
            "date": "15-04-25",
            "revision": "B",
            "elaborated_by": "J.A.",
            "approved_by": "R.L.",
            "doc_title": DOC_TITLE,
            "theme_color": THEME_COLOR,
            "theme_color_font": THEME_COLOR_FONT,
        }

    # Load operativity data once
    operativity_df = pd.read_csv(
        "var/sample_client/sample_project/processed_data/operativity.csv", sep=";"
    )
    
    # DXF path for the structure
    dxf_path = f"data/config/sample_client/sample_project/dxf/{structure_code}.dxf"
    
    print(f"\nProcessing structure: {structure_name} ({structure_code})...")
    
    # Process each sensor type for this structure
    for sensor_type in sensors:
        print(f"  Processing sensor type: {sensor_type}...")
        
        # Get sensor configuration
        sensor_config = get_sensor_config(sensor_type)
        if not sensor_config:
            print(f"  Error: Configuration for sensor type '{sensor_type}' not found.")
            continue
            
        # Filter operativity data for this structure and sensor type
        sensor_df = operativity_df[
            (operativity_df["structure"] == structure_code) &
            (operativity_df["sensor_type"] == sensor_type.upper()) &
            (operativity_df["operativiy"] == True)
        ]
        
        # Skip if no sensors of this type for this structure
        if sensor_df.empty:
            print(f"  No {sensor_type} sensors found for {structure_name}.")
            continue
            
        # Drop sensors with missing records
        sensor_df.dropna(subset=["first_record", "last_record"], inplace=True)
        
        # Import plotter module
        plot_template = sensor_config["plot_template"]
        try:
            plotter_module = importlib.import_module(
                f"modules.reporter.data.plotters.{plot_template}"
            )
            generate_report = plotter_module.generate_report
        except ImportError as e:
            print(f"  Error importing plotter module: {e}")
            continue
        
        # If group is not assigned, use the sensor code as group
        sensor_df["group"] = sensor_df["group"].fillna(sensor_df["code"])
        
        # Process each group
        for group, df_group in sensor_df.groupby("group"):
            # Generate lists for data_sensors
            names = df_group["code"].tolist()
            east = df_group["east"].tolist()
            north = df_group["north"].tolist()
            
            # Get material if available
            material = None
            if "material" in df_group.columns:
                material = df_group["material"].tolist()[0]
                
            # Load data for each sensor
            dfs = []
            for code in names:
                csv_path = f"var/sample_client/sample_project/processed_data/{sensor_type.upper()}/{structure_code}.{code}.csv"
                try:
                    df_sensor = read_df_on_time_from_csv(csv_path, set_index=False)
                    dfs.append(df_sensor)
                except Exception as e:
                    print(f"  Error reading data for sensor {code}: {e}")
                    continue
                    
            # Skip if no data loaded
            if not dfs:
                print(f"  No data loaded for group {group}. Skipping.")
                continue
                
            # Prepare data_sensors dictionary
            data_sensors = {
                "names": names,
                "east": east,
                "north": north,
                "df": dfs,
            }
            
            # Prepare group_args dictionary
            group_args = {
                "name": group,
                "location": structure_name,
            }
            
            # Generate report
            print(f"  [{structure_code} - {sensor_type} - {group}] Generating PDF...")
            
            # Ensure output directory exists
            os.makedirs(f"{output_dir}/{structure_code}/{sensor_type.lower()}", exist_ok=True)
            
            generated_pdf = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                dxf_path=dxf_path,
                start_query=start_query,
                end_query=end_query,
                appendix=sensor_config["appendix"],
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type.lower(),
                output_dir=f"{output_dir}/{structure_code}/{sensor_type.lower()}",
                static_report_params=static_report_params,
                column_config=sensor_config["column_config"],
            )
            
            print(f"  [{structure_code} - {sensor_type} - {group}] Generated PDF: {generated_pdf}")
            n_pdf = len(generated_pdf)
            start_item += n_pdf
            
    return start_item


def main(
    sensors=None,
    output_dir="./outputs/processing",
    start_date="2024-05-01 00:00:00",
    end_date="2025-03-23 00:00:00",
):
    """Main function to run the sensor plotter.

    Args:
        sensors (list): List of sensor types to process. If None, all sensors will be processed.
        output_dir (str): Output directory for generated PDFs.
        start_date (str): Start date for the query (YYYY-MM-DD HH:MM:SS).
        end_date (str): End date for the query (YYYY-MM-DD HH:MM:SS).
    """
    # Define sensor names and their descriptions
    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }

    # If no sensors specified, use all available sensors
    if sensors is None:
        sensors = list(sensor_names.keys())

    # Structure names mapping
    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    # Set static report parameters
    static_report_params = {
        "project_code": "1410.28.0054-0000",
        "company_name": "Shahuindo SAC",
        "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
        "date": "15-04-25",
        "revision": "B",
        "elaborated_by": "J.A.",
        "approved_by": "R.L.",
        "doc_title": DOC_TITLE,
        "theme_color": THEME_COLOR,
        "theme_color_font": THEME_COLOR_FONT,
    }

    # Process each structure first, then each sensor type within the structure
    start_item = 1
    for structure_code, structure_name in structure_names.items():
        start_item = generate_structure_plots(
            structure_code=structure_code,
            structure_name=structure_name,
            sensors=sensors,
            start_item=start_item,
            output_dir=output_dir,
            start_query=start_date,
            end_query=end_date,
            static_report_params=static_report_params,
        )

    print(f"\nAll structures and sensors processed. Final item number: {start_item-1}")


if __name__ == "__main__":
    # You can directly call main with specific parameters
    # Example: main(sensors=["PCV", "PTA"], output_dir="./custom_output", start_date="2024-06-01 00:00:00", end_date="2025-01-01 00:00:00")
    # Or use default parameters by calling without arguments
>>>>>>> 118aabc (update | Independizacion del locale del sistema operativo)
    main()