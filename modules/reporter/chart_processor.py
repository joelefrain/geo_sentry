import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# Add 'libs' path to sys.path
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_PATH)

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
        
    def _generate_df_colors(self, df_names: List[str], palette_name: str = 'viridis') -> Dict[str, str]:
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
        colors = [f'#{int(x[0]*255):02x}{int(x[1]*255):02x}{int(x[2]*255):02x}' 
                for x in cmap(np.linspace(0, 1, len(df_names)))]
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
            mask = (df["time"] >= pd.to_datetime(self.start_query)) & \
                   (df["time"] <= pd.to_datetime(self.end_query))
            return df[mask]
    
    def _create_plot_data(self, df: pd.DataFrame, df_name: str, cell_config: Dict[str, Any], 
                         to_combine: bool, df_colors: Dict[str, str]) -> List[Dict[str, Any]]:
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
                "color": df_colors[df_name] if (to_combine and s == cell_config.get("common_title")) else c,
                "linestyle": lt,
                "linewidth": lw,
                "marker": m,
                "markersize": ms,
                "secondary_y": False,
                "label": df_name if (to_combine and s == cell_config.get("common_title")) 
                        else self.config["names"]["es"][s],
            }
            for s, c, lt, lw, m, ms in zip(
                plot_series, plot_colors, plot_linestyles, plot_linewidths,
                plot_markers, plot_markersizes
            )
        ]
    
    def _create_plotter(self, plot_data: List[Dict[str, Any]], cell_config: Dict[str, Any], 
                       plot_config: Dict[str, Any], df_name: Optional[str] = None) -> Tuple[PlotBuilder, Dict[str, Any]]:
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
    
    def _process_combined_plot(self, plot_key: str, plot_config: Dict[str, Any], 
                             dfs: List[pd.DataFrame], df_names: List[str], 
                             df_colors: Dict[str, str]) -> Dict[str, Any]:
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
                df_filtered = self._filter_dataframe(df_data, cell_config.get("query_type", "query"))
                
                # Create plot data for this dataframe
                plot_data.extend(self._create_plot_data(
                    df_filtered, df_name, cell_config,
                    plot_config["to_combine"], df_colors
                ))
            
            # Create plotter and generate drawing
            plotter, plotter_args = self._create_plotter(plot_data, cell_config, plot_config)
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
            'chart': {
                'key': plot_key,
                'draw': grid.build(color_border="white"),
                'title': plot_config["title_chart"],
                'combined': True,
                'show_notes': show_notes
            },
            'legends': legends
        }
    
    def _process_individual_plots(self, plot_key: str, plot_config: Dict[str, Any], 
                               dfs: List[pd.DataFrame], df_names: List[str], 
                               df_colors: Dict[str, str]) -> List[Dict[str, Any]]:
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
            cells_draw = {}
            legends = []
            
            for cell_key, cell_config in plot_config["cells"].items():
                # Apply query filter per cell
                df_filtered = self._filter_dataframe(df_data, cell_config.get("query_type", "query"))
                
                # Create plot data for this dataframe
                plot_data = self._create_plot_data(
                    df_filtered, df_name, cell_config,
                    plot_config["to_combine"], df_colors
                )
                
                # Create plotter and generate drawing
                plotter, plotter_args = self._create_plotter(plot_data, cell_config, plot_config, df_name)
                plotter.plot_series(**plotter_args)
                cells_draw[tuple(cell_config["position"])] = plotter.get_drawing()
                
                if cell_config["show_legend"]:
                    legends.append(plotter.get_legend(4, 2))
                    
                plotter.close()
            
            if cells_draw:
                # Create grid and add drawings
                grid = PlotMerger(fig_size=tuple(plot_config["fig_size"]))
                grid.create_grid(*plot_config["grid_size"])
                
                for position, draw in cells_draw.items():
                    grid.add_object(draw, position)
                
                # Verificar si las notas están habilitadas para este plot
                show_notes = plot_config.get("notes", True) != False
                if "show_note" in plot_config:
                    show_notes = plot_config["show_note"]
                
                results.append({
                    'chart': {
                        'key': plot_key,
                        'draw': grid.build(color_border="white"),
                        'title': f"{plot_config['title_chart']} - {df_name}",
                        'combined': False,
                        'df_name': df_name,
                        'show_notes': show_notes
                    },
                    'legends': legends
                })
        
        return results
    
    def generate_charts(self, dfs: List[pd.DataFrame], df_names: List[str]) -> Dict[str, List]:
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
        df_colors = self._generate_df_colors(df_names, 'viridis')
        
        results = {
            'charts': [],
            'legends': []
        }
        
        # Process each plot configuration
        for plot_key, plot_config in self.config["plots"].items():
            if plot_config["to_combine"]:
                # Process combined plot
                plot_result = self._process_combined_plot(plot_key, plot_config, dfs, df_names, df_colors)
                results['charts'].append(plot_result['chart'])
                results['legends'].extend(plot_result['legends'])
            else:
                # Process individual plots
                plot_results = self._process_individual_plots(plot_key, plot_config, dfs, df_names, df_colors)
                for result in plot_results:
                    results['charts'].append(result['chart'])
                    results['legends'].extend(result['legends'])
        
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
        
        for chart in charts_and_legends['charts']:
            appendix_item = f"{self.report_params['appendix_num']}.{item}"
            
            # Verificar si las notas deben mostrarse para este gráfico
            show_notes = chart.get('show_notes', True)
            upper_cell = self.report_params['note_paragraph'] if show_notes else None
            
            pdf_generator = ReportBuilder(
                sample=self.report_params['sample'],
                theme_color=self.report_params['theme_color'],
                theme_color_font=self.report_params['theme_color_font'],
                logo_cell=self.report_params['logo_cell'],
                upper_cell=upper_cell,
                lower_cell=self.report_params['map_draw'],
                chart_cell=chart['draw'],
                chart_title=chart['title'],
                num_item=appendix_item,
                project_code=self.report_params['engineer_code'],
                company_name=self.report_params['company_name'],
                project_name=self.report_params['engineer_name'],
                date=self.report_params['date_chart'],
                revision=self.report_params['revision'],
                elaborated_by=self.report_params['elaborated_by'],
                approved_by=self.report_params['approved_by'],
                doc_title=self.report_params['doc_title'],
            )
            
            filename = (f"{self.report_params['appendix_num']}.{item}_{chart['key']}"
                      f"{'_' + chart['df_name'] if not chart['combined'] else ''}.pdf")
            pdf_generator.generate_pdf(pdf_path=filename)
            item += 1