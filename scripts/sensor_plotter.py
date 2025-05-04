import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import importlib
import pandas as pd

from libs.utils.config_loader import load_toml
from libs.utils.config_logger import get_logger, log_execution_time
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_variables import (
    DOC_TITLE,
    THEME_COLOR,
    THEME_COLOR_FONT,
    DATA_CONFIG,
    PROCESS_OUTPUT_DIR,
)

logger = get_logger("scripts.sensor_plotter")

def get_sensor_config(config, sensor_type):
    """Get the configuration for a specific sensor type from TOML file."""
    return config["sensors"].get(sensor_type.lower(), {})


def generate_structure_plots(
    config,
    structure_code,
    structure_name,
    client_code,
    project_code,
    sensors,
    start_item,
    appendix_chapter,
    output_dir,
    start_query,
    end_query,
    static_report_params,
):
    """Generate plots for a specific structure with all requested sensor types.

    Args:
        structure_code (str): The structure code (PAD_1A, PAD_2A, etc.)
        structure_name (str): The structure name (Pad 1A, Pad 2A, etc.)
        sensors (list): List of sensor types to process
        start_item (int): Starting item number for the report
        appendix_chapter (str): Appendix chapter for the report
        output_dir (str): Output directory for the generated PDFs
        start_query (str): Start date for the query
        end_query (str): End date for the query
        static_report_params (dict): Static report parameters

    Returns:
        int: The next start item number
    """

    # Load operativity data once
    operativity_df = pd.read_csv(
        f"var/{client_code}/{project_code}/processed_data/operativity.csv", sep=";"
    )

    # DXF path for the structure
    dxf_path = f"data/config/{client_code}/{project_code}/dxf/{structure_code}.dxf"

    logger.info(f"\nProcessing structure: {structure_name} ({structure_code})...")

    # Process each sensor type for this structure
    for sensor_type in sensors:
        logger.info(f"  Processing sensor type: {sensor_type}...")

        # Get sensor configuration
        sensor_config = get_sensor_config(config, sensor_type)
        if not sensor_config:
            logger.error(f"  Error: Configuration for sensor type '{sensor_type}' not found.")
            continue

        # Filter operativity data for this structure and sensor type
        sensor_df = operativity_df[
            (operativity_df["structure"] == structure_code)
            & (operativity_df["sensor_type"] == sensor_type.upper())
            & (operativity_df["operativiy"] == True)
        ]

        # Skip if no sensors of this type for this structure
        if sensor_df.empty:
            logger.warning(f"  No {sensor_type} sensors found for {structure_name}.")
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
            logger.error(f"  Error importing plotter module: {e}")
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
            material = (
                df_group["material"].iloc[0] if "material" in df_group.columns else None
            )

            # Load data for each sensor
            dfs = []
            for code in names:
                csv_path = f"var/{client_code}/{project_code}/processed_data/{sensor_type.upper()}/{structure_code}.{code}.csv"
                try:
                    df_sensor = read_df_on_time_from_csv(
                        csv_path, set_index=False, auto_convert=True, num_decimals=3
                    )
                    dfs.append(df_sensor)
                except Exception as e:
                    logger.error(f"  Error reading data for sensor {code}: {e}")
                    continue

            # Skip if no data loaded
            if not dfs:
                logger.warning(f"  No data loaded for group {group}. Skipping.")
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
                "material": material,
            }

            # Generate report
            logger.info(f"  [{structure_code} - {sensor_type} - {group}] Generating PDF...")

            # Ensure output directory exists
            os.makedirs(
                f"{output_dir}/{structure_code}/{sensor_type.lower()}", exist_ok=True
            )

            generated_pdf = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                dxf_path=dxf_path,
                start_query=start_query,
                end_query=end_query,
                appendix=appendix_chapter,
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type.lower(),
                output_dir=f"{output_dir}/{structure_code}/{sensor_type.lower()}",
                static_report_params=static_report_params,
                column_config=sensor_config["column_config"],
            )

            logger.info(
                f"  [{structure_code} - {sensor_type} - {group}] Generated PDF: {generated_pdf}"
            )
            n_pdf = len(generated_pdf)
            start_item += n_pdf

    return start_item

@log_execution_time(module="scripts.sensor_plotter")
def exec_plotter(
    client_code,
    project_code,
    engineering_code,
    elaborated_by,
    approved_by,
    start_date,
    end_date,
    report_date,
    start_item,
    appendix_chapter,
    revsion,
    sensors,
    output_dir=PROCESS_OUTPUT_DIR,
):
    # Load configuration from TOML
    config_dir = DATA_CONFIG / client_code / project_code / "plot_formats"
    config = load_toml(config_dir, engineering_code)

    # Extract structure names and project parameters
    structure_names = config["structures"]
    project_params = config["project"]

    # Set static report parameters
    static_report_params = {
        **project_params,
        "elaborated_by": elaborated_by,
        "approved_by": approved_by,
        "revision": revsion,
        "date": report_date,
        "doc_title": DOC_TITLE,
        "theme_color": THEME_COLOR,
        "theme_color_font": THEME_COLOR_FONT,
    }

    # Process each structure first, then each sensor type within the structure
    for structure_code, structure_name in structure_names.items():
        start_item = generate_structure_plots(
            config=config,
            structure_code=structure_code,
            structure_name=structure_name,
            client_code=client_code,
            project_code=project_code,
            sensors=sensors,
            start_item=start_item,
            appendix_chapter=appendix_chapter,
            output_dir=output_dir,
            start_query=start_date,
            end_query=end_date,
            static_report_params=static_report_params,
        )

    logger.info(
        f"\nAll structures and sensors processed. Final item number: {start_item - 1}"
    )


if __name__ == "__main__":
    plotter_params = {
        "client_code": "sample_client",
        "project_code": "sample_project",
        "engineering_code": "eor_2025",
        "elaborated_by": "J.A.",
        "approved_by": "R.L.",
        "start_date": "2024-05-01 00:00:00",
        "end_date": "2025-03-23 00:00:00",
        "report_date": "15-04-25",
        "start_item": 1,
        "appendix_chapter": "5",
        "revsion": "B",
        "sensors": ["PCV", "PTA", "PCT", "SACV", "CPCV"],
    }

    exec_plotter(**plotter_params)
