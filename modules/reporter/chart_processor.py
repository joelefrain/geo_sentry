import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from modules.reporter.plot_builder import PlotBuilder, PlotMerger
from libs.utils.logger_config import get_logger

logger = get_logger("modules.reporter.chart_processor")


class ChartProcessor:
    """Class for processing dataframes and generating charts.

    This class follows SOLID principles to handle the processing of dataframes
    and generation of charts in a modular and extensible way.

    Attributes
    ----------
    config : dict
        Configuration dictionary from TOML file
    start_query : str
        Start date for filtering
    end_query : str
        End date for filtering
    """

    def __init__(self, config: Dict[str, Any], start_query: str, end_query: str):
        """Initialize the ChartProcessor.

        Parameters
        ----------
        config : dict
            Configuration dictionary from TOML file
        start_query : str
            Start date for filtering
        end_query : str
            End date for filtering
        """
        self.config = config
        self.start_query = start_query
        self.end_query = end_query

    def _generate_df_colors(
        self, df_names: List[str], palette_name: str = "viridis"
    ) -> Dict[str, str]:
        """Generate colors from palette for each DataFrame.

        Parameters
        ----------
        df_names : list
            List of dataframe names
        palette_name : str, optional
            Name of the matplotlib colormap, by default 'viridis'

        Returns
        -------
        dict
            Dictionary mapping df_names to colors
        """
        from matplotlib import pyplot as plt

        cmap = plt.colormaps[palette_name]
        colors = [
            f"#{int(x[0]*255):02x}{int(x[1]*255):02x}{int(x[2]*255):02x}"
            for x in cmap(np.linspace(0, 1, len(df_names)))
        ]
        return dict(zip(df_names, colors))

    def _filter_dataframe(self, df: pd.DataFrame, query_type: str) -> pd.DataFrame:
        """Filter dataframe based on query type.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to filter
        query_type : str
            Type of query to apply ('all' or 'query')

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame
        """
        if query_type == "all":
            return df.copy()
        else:  # query_type == "query"
            mask = (df["time"] >= pd.to_datetime(self.start_query)) & (
                df["time"] <= pd.to_datetime(self.end_query)
            )
            return df[mask]

    def _create_plot_data(
        self,
        df: pd.DataFrame,
        df_name: str,
        cell_config: Dict[str, Any],
        to_combine: bool,
        df_colors: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Create plot data dictionary for a single dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data to plot
        df_name : str
            Name of the dataframe
        cell_config : dict
            Configuration for the cell
        to_combine : bool
            Whether plots are being combined
        df_colors : dict
            Dictionary mapping df_names to colors

        Returns
        -------
        list
            List of dictionaries containing plot data
        """
        plot_series = cell_config["serie"]
        plot_colors = cell_config["color"]
        plot_linestyles = cell_config["linestyle"]
        plot_linewidths = cell_config["linewidth"]
        plot_markers = cell_config["marker"]
        plot_markersizes = cell_config["markersize"]

        return [
            {
                "x": df[cell_config["title_x"]].tolist(),
                "y": df[s].tolist(),
                "color": (
                    df_colors[df_name]
                    if (to_combine and s == cell_config.get("common_title"))
                    else c
                ),
                "linestyle": lt,
                "linewidth": lw,
                "marker": m,
                "markersize": ms,
                "secondary_y": False,
                "label": (
                    df_name
                    if (to_combine and s == cell_config.get("common_title"))
                    else self.config["names"]["es"][s]
                ),
            }
            for s, c, lt, lw, m, ms in zip(
                plot_series,
                plot_colors,
                plot_linestyles,
                plot_linewidths,
                plot_markers,
                plot_markersizes,
            )
        ]

    def _create_plotter(
        self,
        plot_data: List[Dict[str, Any]],
        cell_config: Dict[str, Any],
        plot_config: Dict[str, Any],
        df_name: Optional[str] = None,
    ) -> Tuple[PlotBuilder, Dict[str, Any]]:
        """Create a PlotBuilder instance with the given plot data.

        Parameters
        ----------
        plot_data : list
            List of dictionaries containing plot data
        cell_config : dict
            Configuration for the cell
        plot_config : dict
            Configuration for the plot
        df_name : str, optional
            Name of the dataframe, by default None

        Returns
        -------
        tuple
            Tuple containing the PlotBuilder instance and the plotter arguments
        """
        plotter = PlotBuilder()

        title_chart = plot_config["title_chart"]
        if df_name is not None:
            title_chart = f"{title_chart} - {df_name}"

        plotter_args = {
            "data": plot_data,
            "size": tuple(cell_config["size"]),
            "title_x": self.config["names"]["es"][cell_config["title_x"]],
            "title_y": self.config["names"]["es"][cell_config["title_y"]],
            "title_chart": title_chart,
            "show_legend": cell_config["show_legend"],
        }

        return plotter, plotter_args

    def _process_combined_plot(
        self,
        plot_key: str,
        plot_config: Dict[str, Any],
        dfs: List[pd.DataFrame],
        df_names: List[str],
        df_colors: Dict[str, str],
    ) -> Dict[str, Any]:
        """Process a combined plot with multiple dataframes.

        Parameters
        ----------
        plot_key : str
            Key of the plot in the configuration
        plot_config : dict
            Configuration for the plot
        dfs : list
            List of dataframes to process
        df_names : list
            List of dataframe names
        df_colors : dict
            Dictionary mapping df_names to colors

        Returns
        -------
        dict
            Dictionary containing chart information and legends
        """
        cells_draw = {}
        legends = []

        for cell_key, cell_config in plot_config["cells"].items():
            plot_data = []

            for df_idx, (df_data, df_name) in enumerate(zip(dfs, df_names)):
                # Apply query filter per cell
                df_filtered = self._filter_dataframe(
                    df_data, cell_config.get("query_type", "query")
                )

                # Create plot data for this dataframe
                plot_data.extend(
                    self._create_plot_data(
                        df_filtered,
                        df_name,
                        cell_config,
                        plot_config["to_combine"],
                        df_colors,
                    )
                )

            # Create plotter and generate drawing
            plotter, plotter_args = self._create_plotter(
                plot_data, cell_config, plot_config
            )
            plotter.plot_series(**plotter_args)
            cells_draw[tuple(cell_config["position"])] = plotter.get_drawing()

            if cell_config["show_legend"]:
                legends.append(plotter.get_legend(4, 2))

            plotter.close()

        # Create grid and add drawings
        grid = PlotMerger(fig_size=tuple(plot_config["fig_size"]))
        grid.create_grid(*plot_config["grid_size"])

        for position, draw in cells_draw.items():
            grid.add_object(draw, position)

        # Verificar si las notas están habilitadas para este plot
        show_notes = plot_config.get("notes", True) != False
        if "show_note" in plot_config:
            show_notes = plot_config["show_note"]

        return {
            "chart": {
                "key": plot_key,
                "draw": grid.build(color_border="white"),
                "title": plot_config["title_chart"],
                "combined": True,
                "show_notes": show_notes,
            },
            "legends": legends,
        }

    def _process_individual_plots(
        self,
        plot_key: str,
        plot_config: Dict[str, Any],
        dfs: List[pd.DataFrame],
        df_names: List[str],
        df_colors: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Process individual plots for each dataframe.

        Parameters
        ----------
        plot_key : str
            Key of the plot in the configuration
        plot_config : dict
            Configuration for the plot
        dfs : list
            List of dataframes to process
        df_names : list
            List of dataframe names
        df_colors : dict
            Dictionary mapping df_names to colors

        Returns
        -------
        list
            List of dictionaries containing chart information and legends
        """
        results = []

        for df_idx, (df_data, df_name) in enumerate(zip(dfs, df_names)):
            chart_result = self._process_single_dataframe_plot(
                plot_key, plot_config, df_data, df_name, df_colors
            )
            if chart_result:
                results.append(chart_result)

        return results

    def _process_single_dataframe_plot(
        self,
        plot_key: str,
        plot_config: Dict[str, Any],
        df_data: pd.DataFrame,
        df_name: str,
        df_colors: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Process plot for a single dataframe.

        Parameters
        ----------
        plot_key : str
            Key of the plot in the configuration
        plot_config : dict
            Configuration for the plot
        df_data : pandas.DataFrame
            DataFrame to process
        df_name : str
            Name of the dataframe
        df_colors : dict
            Dictionary mapping df_names to colors

        Returns
        -------
        dict or None
            Dictionary containing chart information and legends, or None if no cells were drawn
        """
        cells_draw = {}
        legends = []

        # Process each cell in the plot configuration
        for cell_key, cell_config in plot_config["cells"].items():
            cell_result = self._process_cell_for_dataframe(
                df_data, df_name, cell_config, plot_config, df_colors
            )

            if cell_result:
                cells_draw[tuple(cell_config["position"])] = cell_result["drawing"]
                if cell_result["legend"]:
                    legends.append(cell_result["legend"])

        if not cells_draw:
            return None

        # Create grid with all cell drawings
        grid_result = self._create_grid_from_cells(plot_config, cells_draw)

        # Determine if notes should be shown
        show_notes = self._determine_show_notes(plot_config)

        return {
            "chart": {
                "key": plot_key,
                "draw": grid_result,
                "title": f"{plot_config['title_chart']} - {df_name}",
                "combined": False,
                "df_name": df_name,
                "show_notes": show_notes,
            },
            "legends": legends,
        }

    def _process_cell_for_dataframe(
        self,
        df_data: pd.DataFrame,
        df_name: str,
        cell_config: Dict[str, Any],
        plot_config: Dict[str, Any],
        df_colors: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Process a single cell for a dataframe.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame to process
        df_name : str
            Name of the dataframe
        cell_config : dict
            Configuration for the cell
        plot_config : dict
            Configuration for the plot
        df_colors : dict
            Dictionary mapping df_names to colors

        Returns
        -------
        dict or None
            Dictionary containing drawing and legend, or None if processing failed
        """
        # Apply query filter per cell
        df_filtered = self._filter_dataframe(
            df_data, cell_config.get("query_type", "query")
        )

        # Create plot data for this dataframe
        plot_data = self._create_plot_data(
            df_filtered,
            df_name,
            cell_config,
            plot_config["to_combine"],
            df_colors,
        )

        # Create plotter and generate drawing
        plotter, plotter_args = self._create_plotter(
            plot_data, cell_config, plot_config, df_name
        )
        plotter.plot_series(**plotter_args)
        drawing = plotter.get_drawing()

        # Get legend if needed
        legend = plotter.get_legend(4, 2) if cell_config["show_legend"] else None

        plotter.close()

        return {"drawing": drawing, "legend": legend}

    def _create_grid_from_cells(
        self,
        plot_config: Dict[str, Any],
        cells_draw: Dict[Tuple[int, int], Any],
    ) -> Any:
        """Create a grid from cell drawings.

        Parameters
        ----------
        plot_config : dict
            Configuration for the plot
        cells_draw : dict
            Dictionary mapping positions to drawings

        Returns
        -------
        Any
            The built grid
        """
        grid = PlotMerger(fig_size=tuple(plot_config["fig_size"]))
        grid.create_grid(*plot_config["grid_size"])

        for position, draw in cells_draw.items():
            grid.add_object(draw, position)

        return grid.build(color_border="white")

    def _determine_show_notes(
        self,
        plot_config: Dict[str, Any],
    ) -> bool:
        """Determine if notes should be shown for a plot.

        Parameters
        ----------
        plot_config : dict
            Configuration for the plot

        Returns
        -------
        bool
            Whether notes should be shown
        """
        show_notes = plot_config.get("notes", True) != False
        if "show_note" in plot_config:
            show_notes = plot_config["show_note"]
        return show_notes

    def generate_charts(
        self, dfs: List[pd.DataFrame], df_names: List[str]
    ) -> Dict[str, List]:
        """Process dataframes and generate charts and legends.

        Parameters
        ----------
        dfs : list
            List of dataframes to process
        df_names : list
            List of dataframe names

        Returns
        -------
        dict
            Dictionary containing 'charts' and 'legends' lists
        """
        # Generate colors from palette for each DataFrame
        df_colors = self._generate_df_colors(df_names, "viridis")

        results = {"charts": [], "legends": []}

        # Process each plot configuration
        for plot_key, plot_config in self.config["plots"].items():
            if plot_config["to_combine"]:
                # Process combined plot
                plot_result = self._process_combined_plot(
                    plot_key, plot_config, dfs, df_names, df_colors
                )
                results["charts"].append(plot_result["chart"])
                results["legends"].extend(plot_result["legends"])
            else:
                # Process individual plots
                plot_results = self._process_individual_plots(
                    plot_key, plot_config, dfs, df_names, df_colors
                )
                for result in plot_results:
                    results["charts"].append(result["chart"])
                    results["legends"].extend(result["legends"])

        return results


class PDFGenerator:
    """Class for generating PDFs from charts and legends.

    This class follows the Single Responsibility Principle to handle
    the generation of PDFs from charts and legends.

    Attributes
    ----------
    report_params : dict
        Dictionary containing parameters for the report
    """

    def __init__(self, report_params: Dict[str, Any]):
        """Initialize the PDFGenerator.

        Parameters
        ----------
        report_params : dict
            Dictionary containing parameters for the report
        """
        self.report_params = report_params

    def generate_pdfs(self, charts_and_legends: Dict[str, List]):
        """Generate PDFs for each chart with its corresponding legend.

        Parameters
        ----------
        charts_and_legends : dict
            Dictionary containing 'charts' and 'legends' lists
        """
        from modules.reporter.report_builder import ReportBuilder

        item = 200

        for chart in charts_and_legends["charts"]:
            appendix_item = f"{self.report_params['appendix_num']}.{item}"

            # Verificar si las notas deben mostrarse para este gráfico
            show_notes = chart.get("show_notes", True)
            upper_cell = self.report_params["note_paragraph"] if show_notes else None

            pdf_generator = ReportBuilder(
                sample=self.report_params["sample"],
                theme_color=self.report_params["theme_color"],
                theme_color_font=self.report_params["theme_color_font"],
                logo_cell=self.report_params["logo_cell"],
                upper_cell=upper_cell,
                lower_cell=self.report_params["map_draw"],
                chart_cell=chart["draw"],
                chart_title=chart["title"],
                num_item=appendix_item,
                project_code=self.report_params["engineer_code"],
                company_name=self.report_params["company_name"],
                project_name=self.report_params["engineer_name"],
                date=self.report_params["date_chart"],
                revision=self.report_params["revision"],
                elaborated_by=self.report_params["elaborated_by"],
                approved_by=self.report_params["approved_by"],
                doc_title=self.report_params["doc_title"],
            )

            filename = (
                f"{self.report_params['appendix_num']}.{item}_{chart['key']}"
                f"{'_' + chart['df_name'] if not chart['combined'] else ''}.pdf"
            )
            pdf_generator.generate_pdf(pdf_path=filename)
            item += 1
