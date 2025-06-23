import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import importlib

import pandas as pd

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from libs.utils.config_variables import (
    DOC_TITLE,
    THEME_COLOR,
    THEME_COLOR_FONT,
    DATA_CONFIG,
    OUTPUTS_DIR,
    MINIMUN_RECORDS,
)

from libs.utils.config_loader import load_toml
from libs.utils.text_helpers import write_lines
from libs.utils.calc_helpers import calc_epsg_from_utm
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.helpers.pdf_merger import find_pdf_files, merge_pdfs
from libs.utils.validation_helpers import flatten, get_field_from_dict

from libs.utils.config_logger import get_logger, log_execution_time

logger = get_logger("scripts.sensor_plotter")


class SensorDataLoader:
    """Handles loading and processing of sensor data."""

    def __init__(self, client_code: str, project_code: str):
        self.client_code = client_code
        self.project_code = project_code
        self.operativity_df = self._load_operativity_data()

    def _load_operativity_data(self) -> pd.DataFrame:
        """Load operativity data once for reuse."""
        operativity_path = (
            f"var/{self.client_code}/{self.project_code}/processed_data/operativity.csv"
        )
        return pd.read_csv(operativity_path, sep=";")

    def get_sensor_data(self, structure_code: str, sensor_type: str) -> pd.DataFrame:
        """Get filtered sensor data for a specific structure and sensor type."""
        return self.operativity_df[
            (self.operativity_df["structure"] == structure_code)
            & (self.operativity_df["sensor_type"] == sensor_type.upper())
            & (self.operativity_df["operativiy"] == True)
        ].dropna(subset=["first_record", "last_record"])

    def load_sensor_readings(
        self, structure_code: str, sensor_type: str, sensor_codes: List[str]
    ) -> List[pd.DataFrame]:
        """Load actual sensor readings data."""
        dfs = []
        for code in sensor_codes:
            csv_path = f"var/{self.client_code}/{self.project_code}/processed_data/{sensor_type.upper()}/{structure_code}.{code}.csv"
            try:
                df_sensor = read_df_on_time_from_csv(
                    csv_path, set_index=False, auto_convert=True, num_decimals=3
                )
                # df_sensor = df_sensor[~df_sensor["base_line"]]

                if len(df_sensor) > MINIMUN_RECORDS:
                    dfs.append(df_sensor)
                else:
                    logger.warning(
                        f"Sensor {code} has less than {MINIMUN_RECORDS + 1} records. Skipping."
                    )
            except Exception as e:
                logger.exception(f"Error reading data for sensor {code}: {e}")
                continue
        return dfs


class SensorGroupManager:
    """Manages sensor grouping strategies."""

    @staticmethod
    def create_groups(
        sensor_df: pd.DataFrame,
        structure_code: str,
        sensor_type: str,
        by_groups: bool = True,
        all_sensor_type: bool = False,
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Create sensor groups based on grouping strategy."""
        if all_sensor_type:
            return SensorGroupManager._create_all_sensor_type_groups(
                sensor_df, structure_code
            )
        elif by_groups:
            return SensorGroupManager._create_attribute_groups(sensor_df)
        else:
            return SensorGroupManager._create_single_group(
                sensor_df, structure_code, sensor_type
            )

    @staticmethod
    def _create_all_sensor_type_groups(
        sensor_df: pd.DataFrame, structure_code: str
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Group all sensor types together by structure."""
        group_name = f"{structure_code} - All Sensors"
        sensor_df["group"] = group_name
        return [(group_name, sensor_df)]

    @staticmethod
    def _create_attribute_groups(
        sensor_df: pd.DataFrame,
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Group sensors by their 'group' attribute."""
        sensor_df["group"] = sensor_df["group"].fillna(sensor_df["code"])
        return list(sensor_df.groupby("group"))

    @staticmethod
    def _create_single_group(
        sensor_df: pd.DataFrame, structure_code: str, sensor_type: str
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Create single group with structure and sensor type."""
        group_name = f"{structure_code} - {sensor_type}"
        sensor_df["group"] = group_name
        return [(group_name, sensor_df)]


class PlotGenerator:
    """Handles plot generation for sensor groups."""

    def __init__(self, config: Dict[str, Any], data_loader: SensorDataLoader):
        self.config = config
        self.data_loader = data_loader

    def get_sensor_config(self, sensor_type: str) -> Dict[str, Any]:
        """Get configuration for a specific sensor type."""
        try:
            return get_field_from_dict(["sensors", sensor_type.lower()], self.config)
        except ValueError as e:
            logger.error(f"Configuration validation error: {e}")
            return {}

    def generate_plots_for_group(
        self,
        group_name: str,
        df_group: pd.DataFrame,
        structure_code: str,
        structure_name: str,
        sensor_type: str,
        plotter_params: Dict[str, Any],
        static_report_params: Dict[str, Any],
        start_item: int,
        output_dir: str,
        appendix_chapter: str = None,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """Generate plots for a specific sensor group."""

        # Get sensor configuration
        sensor_config = self.get_sensor_config(sensor_type)
        if not sensor_config:
            logger.error(f"Configuration for sensor type '{sensor_type}' not found.")
            return None, None

        # Read sensors to use during plotting
        codes = df_group["code"].tolist()

        # Load sensor readings
        dfs = self.data_loader.load_sensor_readings(structure_code, sensor_type, codes)
        if not dfs:
            logger.warning(f"No data loaded for group {group_name}. Skipping.")
            return None, None

        # Convert all columns of df_group to lists
        columns_as_lists = {col: df_group[col].tolist() for col in df_group.columns}

        # Add any other data
        columns_as_lists.update({"df": dfs})

        data_sensors = columns_as_lists

        # Add any additional columns from df_group as lists
        for col in df_group.columns:
            if col not in ["code", "east", "north", "material"]:
                data_sensors[col] = df_group[col].tolist()

        group_args = {
            "name": group_name,
            "location": structure_name,
        }

        # Import and execute plotter
        return self._execute_plotter(
            sensor_config,
            data_sensors,
            group_args,
            structure_code,
            structure_name,
            sensor_type,
            plotter_params,
            static_report_params,
            start_item,
            output_dir,
            appendix_chapter,
        )

    def _execute_plotter(
        self,
        sensor_config: Dict[str, Any],
        data_sensors: Dict[str, Any],
        group_args: Dict[str, Any],
        structure_code: str,
        structure_name: str,
        sensor_type: str,
        plotter_params: Dict[str, Any],
        static_report_params: Dict[str, Any],
        start_item: int,
        output_dir: str,
        appendix_chapter: str = None,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """Execute the plotting module."""

        try:
            plot_template = get_field_from_dict("plot_template", sensor_config)
        except ValueError as e:
            logger.error(f"Plot template validation error: {e}")
            return None, None

        try:
            plotter_module = importlib.import_module(f"modules.plotter.{plot_template}")
            generate_report = plotter_module.generate_report
        except ImportError as e:
            logger.exception(f"Error importing plotter module: {e}")
            return None, None

        try:
            column_config = get_field_from_dict("column_config", sensor_config)
        except ValueError as e:
            logger.error(f"Column config validation error: {e}")
            return None, None

        try:
            generated_pdf, chart_title = generate_report(
                data_sensors=data_sensors,
                group_args=group_args,
                appendix=appendix_chapter,
                start_item=start_item,
                structure_code=structure_code,
                structure_name=structure_name,
                sensor_type=sensor_type.lower(),
                output_dir=f"{output_dir}/{structure_code}/{sensor_type.lower()}",
                static_report_params=static_report_params,
                column_config=column_config,
                **plotter_params,
            )

            if not generated_pdf:
                logger.warning(
                    f"No PDF generated for group {group_args['name']} in {structure_code} - {sensor_type}"
                )
                return None, None

            return generated_pdf, chart_title

        except Exception as e:
            logger.exception(
                f"Error generating PDF for group {group_args['name']}: {e}"
            )
            return None, None


class StructureProcessor:
    """Processes all sensors for a single structure."""

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader: SensorDataLoader,
        plot_generator: PlotGenerator,
    ):
        self.config = config
        self.data_loader = data_loader
        self.plot_generator = plot_generator

    def process_structure(
        self,
        structure_code: str,
        structure_name: str,
        sensors: List[str],
        plotter_params: Dict[str, Any],
        static_report_params: Dict[str, Any],
        start_item: int,
        output_dir: str,
        appendix_chapter: str = None,
        by_groups: bool = True,
        all_sensor_type: bool = False,
    ) -> Tuple[int, List[str]]:
        """Process all sensor types for a single structure."""

        logger.info(f"\nProcessing structure: {structure_name} ({structure_code})...")

        chart_titles = []

        # Get DXF and TIF paths
        base_path = (
            Path("data/config")
            / self.data_loader.client_code
            / self.data_loader.project_code
        )
        structure_paths = {
            ext: base_path / ext / f"{structure_code}.{ext}" for ext in ["dxf", "tif"]
        }

        plotter_params.update(
            {f"{ext}_path": path for ext, path in structure_paths.items()}
        )

        # Process each sensor type
        for sensor_type in sensors:
            start_item, sensor_chart_titles = self._process_sensor_type(
                structure_code,
                structure_name,
                sensor_type,
                plotter_params,
                static_report_params,
                start_item,
                output_dir,
                appendix_chapter,
                by_groups,
                all_sensor_type,
            )
            if sensor_chart_titles:
                chart_titles.extend(sensor_chart_titles)

        return start_item, chart_titles

    def _process_sensor_type(
        self,
        structure_code: str,
        structure_name: str,
        sensor_type: str,
        plotter_params: Dict[str, Any],
        static_report_params: Dict[str, Any],
        start_item: int,
        output_dir: str,
        appendix_chapter: str = None,
        by_groups: bool = True,
        all_sensor_type: bool = False,
    ) -> Tuple[int, List[str]]:
        """Process a single sensor type for a structure."""

        logger.info(f"Processing sensor type: {sensor_type}...")

        # Get sensor data
        sensor_df = self.data_loader.get_sensor_data(structure_code, sensor_type)
        if sensor_df.empty:
            logger.warning(f"No {sensor_type} sensors found for {structure_name}.")
            return start_item, []

        # Create groups
        groups = SensorGroupManager.create_groups(
            sensor_df, structure_code, sensor_type, by_groups, all_sensor_type
        )

        chart_titles = []

        # Process each group
        for group_name, df_group in groups:
            generated_pdf, chart_title = self.plot_generator.generate_plots_for_group(
                group_name,
                df_group,
                structure_code,
                structure_name,
                sensor_type,
                plotter_params,
                static_report_params,
                start_item,
                output_dir,
                appendix_chapter,
            )

            if generated_pdf and chart_title:
                chart_titles.append(chart_title)
                n_pdf = len(generated_pdf)
                start_item += n_pdf

                logger.info(
                    f"[{structure_code} - {sensor_type} - {group_name}] Generated PDF: {generated_pdf} "
                    f"(Total: {len(generated_pdf)}) Starting item: {start_item}"
                )

        return start_item, chart_titles


class OutputManager:
    """Manages output file operations."""

    @staticmethod
    def merge_outputs(
        plot_type: str, client_code: str, project_code: str, output_dir: Path
    ) -> None:
        """Merge all generated PDFs into a single output file."""
        output_file = (
            OUTPUTS_DIR / "appendix" / client_code / project_code / f"{plot_type}.pdf"
        )

        if not os.path.isdir(output_dir):
            logger.exception(f"Input directory '{output_dir}' does not exist")
            return

        pdf_files = find_pdf_files(output_dir)
        if not pdf_files:
            logger.warning(f"No PDF files found in '{output_dir}'")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")

        try:
            merge_pdfs(pdf_files, output_file)
            logger.info(f"PDFs successfully merged into '{output_file}'")
        except Exception as e:
            logger.exception(f"Error merging PDFs: {str(e)}")

    @staticmethod
    def save_chart_titles(chart_titles: List[str], output_dir: Path) -> None:
        """Save all chart titles to a text file."""
        flattened_titles = flatten(chart_titles)
        write_lines(flattened_titles, output_dir / "charts.txt")


class SensorPlotterOrchestrator:
    """Main orchestrator for the sensor plotting process."""

    def __init__(self, client_code: str, project_code: str, engineering_code: str):
        self.client_code = client_code
        self.project_code = project_code
        self.engineering_code = engineering_code
        self.data_loader = SensorDataLoader(client_code, project_code)

    def load_configurations(self, plot_type: str) -> Tuple[Dict[str, Any], int, bool]:
        """Load plot and processor configurations."""
        # Load plot configuration
        config_dir = DATA_CONFIG / self.client_code / self.project_code / plot_type
        plotter_config = load_toml(config_dir, self.engineering_code)

        # Load processor configuration for UTM zone
        processor_config_dir = (
            DATA_CONFIG / self.client_code / self.project_code / "processor"
        )
        processor_config = load_toml(processor_config_dir, self.engineering_code)

        try:
            utm_zone = get_field_from_dict(["location", "utm_zone"], processor_config)
        except ValueError as e:
            logger.error(
                f"UTM zone validation error for {self.client_code}/{self.project_code}: {e}"
            )
            sys.exit(1)

        try:
            northern_hemisphere = get_field_from_dict(
                ["location", "northern_hemisphere"], processor_config
            )
        except ValueError as e:
            logger.error(
                f"Hemisphere validation error for {self.client_code}/{self.project_code}: {e}"
            )
            sys.exit(1)

        return plotter_config, utm_zone, northern_hemisphere

    def create_static_report_params(
        self, config: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Create static report parameters from config and kwargs."""
        try:
            project_params = get_field_from_dict("project", config)
        except ValueError as e:
            logger.error(f"Project config validation error: {e}")
            project_params = {}

        return {
            **project_params,
            "elaborated_by": kwargs.get("elaborated_by"),
            "approved_by": kwargs.get("approved_by"),
            "revision": kwargs.get("revision"),
            "date": kwargs.get("report_date"),
            "doc_title": DOC_TITLE,
            "theme_color": THEME_COLOR,
            "theme_color_font": THEME_COLOR_FONT,
        }

    def execute_plotting(self, plot_type: str, sensors: List[str], **kwargs) -> None:
        """Execute the complete sensor plotting process."""
        # Load configurations
        plotter_config, utm_zone, northern_hemisphere = self.load_configurations(
            plot_type
        )

        # Create components
        plot_generator = PlotGenerator(plotter_config, self.data_loader)
        structure_processor = StructureProcessor(
            plotter_config, self.data_loader, plot_generator
        )

        # Setup parameters
        output_dir = str(OUTPUTS_DIR / plot_type / self.client_code / self.project_code)

        try:
            structure_names = get_field_from_dict("structures", plotter_config)
        except ValueError as e:
            logger.error(f"Structures config validation error: {e}")
            return

        static_report_params = self.create_static_report_params(
            plotter_config, **kwargs
        )

        plotter_params = {
            "project_epsg": calc_epsg_from_utm(
                utm_zone=utm_zone, northern_hemisphere=northern_hemisphere
            ),
            "start_query": kwargs.get("start_date"),
            "end_query": kwargs.get("end_date"),
        }

        # Process all structures
        start_item = kwargs.get("start_item", 1)
        all_chart_titles = []

        # Extract configuration parameters with validation
        try:
            format_config = get_field_from_dict("format", plotter_config)
            by_groups = format_config.get("by_groups", True)
            all_sensor_type = format_config.get("all_sensor_type", False)
        except ValueError as e:
            logger.warning(f"Format config validation warning: {e}. Using defaults.")
            by_groups = True
            all_sensor_type = False

        appendix_chapter = kwargs.get("appendix_chapter")

        for structure_code, structure_name in structure_names.items():
            try:
                start_item, chart_titles = structure_processor.process_structure(
                    structure_code=structure_code,
                    structure_name=structure_name,
                    sensors=sensors,
                    plotter_params=plotter_params,
                    static_report_params=static_report_params,
                    start_item=start_item,
                    output_dir=output_dir,
                    appendix_chapter=appendix_chapter,
                    by_groups=by_groups,
                    all_sensor_type=all_sensor_type,
                )

                if chart_titles:
                    all_chart_titles.extend(chart_titles)
                    logger.info(
                        f"Processed structure {structure_code} with {len(chart_titles)} charts."
                    )

            except Exception as e:
                logger.exception(f"Error processing structure {structure_code}: {e}")
                continue

        logger.info(
            f"All structures and sensors processed. Final item number: {start_item - 1}"
        )

        # Generate outputs
        OutputManager.merge_outputs(
            plot_type, self.client_code, self.project_code, Path(output_dir)
        )
        OutputManager.save_chart_titles(all_chart_titles, Path(output_dir))


@log_execution_time(module="scripts.sensor_plotter")
def exec_plotter(
    plot_type: str | list,
    client_code: str,
    project_code: str,
    engineering_code: str,
    sensors: List[str],
    **kwargs,
) -> None:
    if isinstance(plot_type, list):
        start_item = kwargs.get("start_item", 1)
        for pt in plot_type:
            orchestrator = SensorPlotterOrchestrator(
                client_code, project_code, engineering_code
            )
            orchestrator.execute_plotting(
                pt, sensors, **{**kwargs, "start_item": start_item}
            )
            # Update start_item based on the number of charts generated (read from charts.txt)
            output_dir = str(OUTPUTS_DIR / pt / client_code / project_code)
            charts_file = Path(output_dir) / "charts.txt"
            if charts_file.exists():
                with open(charts_file, "r") as f:
                    n_charts = len([line for line in f if line.strip()])
                start_item += n_charts
    else:
        orchestrator = SensorPlotterOrchestrator(
            client_code, project_code, engineering_code
        )
        orchestrator.execute_plotting(plot_type, sensors, **kwargs)


if __name__ == "__main__":
    try:
        plotter_params = {
            "plot_type": ["sensor_plotter", "map_creator"],
            "client_code": "sample_client",
            "project_code": "sample_project",
            "engineering_code": "eor_2025",
            "sensors": ["PCV", "PTA", "PCT", "SACV", "CPCV", "INC"],
            # "sensors": ["PCT"],
            # Required parameters
            "elaborated_by": "J.A.",
            "approved_by": "R.L.",
            "start_date": "2024-12-01 00:00:00",
            "end_date": "2025-06-15 00:00:00",
            "report_date": "15-06-25",
            "start_item": 1,
            "appendix_chapter": "4",
            "revision": "B",
        }

        logger.info("Starting sensor processor with parameters:", extra=plotter_params)
        exec_plotter(**plotter_params)
        logger.info("Sensor processor completed successfully")

    except Exception as e:
        logger.exception(f"Sensor processor failed: {e}")
        sys.exit(1)
