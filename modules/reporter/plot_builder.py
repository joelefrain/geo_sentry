import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import ezdxf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from io import BytesIO
from adjustText import adjust_text
from svglib.svglib import svg2rlg
from reportlab.graphics.shapes import Drawing, Group, Rect
from typing import Tuple, List

from libs.utils.config_plot import PlotConfig
from libs.utils.config_loader import load_toml
from libs.utils.config_variables import CHART_CONFIG_DIR


class PlotBuilder:
    """A versatile class for creating and managing publication-quality plots.

    This class provides a comprehensive interface for generating professional plots
    with support for multiple data visualization features and automatic resource management.

    Key Features
    -----------
    * Multiple data series plotting with customizable styles
    * DXF file visualization and overlay support
    * Color bands and region highlighting
    * Automatic arrow annotations with customizable positioning
    * Smart text annotation placement with collision avoidance
    * Legend management and customization

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Main figure container for the plot
    ax1 : matplotlib.axes.Axes
        Primary plotting axes
    ax2 : matplotlib.axes.Axes, optional
        Secondary axes for dual-axis plots
    show_legend : bool
        Controls legend visibility
    default_styles : dict
        Plot styles loaded from TOML configuration

    Examples
    --------
    Basic line plot:
    >>> plotter = PlotBuilder()
    >>> plotter.plot_series([{
    ...     'x': [1, 2, 3],
    ...     'y': [4, 5, 6],
    ...     'label': 'Data'
    ... }])

    Plot with DXF overlay:
    >>> plotter.plot_series(
    ...     data=[{'x': [0, 1], 'y': [0, 1]}],
    ...     dxf_path='drawing.dxf',
    ...     title_chart='Plot with DXF'
    ... )

    Notes
    -----
    - Manages plot lifecycle and resource cleanup automatically
    - Uses TOML files for style configuration
    - Thread-safe plot generation and cleanup
    - Memory-efficient resource management
    """

    def __init__(
        self, style_file: str = "default", ts_serie: bool = True, ymargin: float = 0.20
    ):
        """
        Initialize the Plotter class.

        Parameters
        ----------
        style_file : str, optional
            Name of the TOML file containing plot styles, by default "default.toml"
            The file should be located in data/charts/ directory.
        """
        self.fig: plt.Figure = None
        self.ax1: plt.Axes = None
        self.ax2: plt.Axes = None
        self.show_legend: bool = False
        self._is_closed: bool = False

        PlotConfig.setup_matplotlib(ts_serie, ymargin)

        # Load default styles from TOML
        try:
            self.default_styles = load_toml(CHART_CONFIG_DIR, style_file)
        except Exception as e:
            self.default_styles = {}
            raise RuntimeError(
                f"Could not load styles from {CHART_CONFIG_DIR / style_file}: {e}"
            )

    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()

    def close(self):
        """Explicitly close and cleanup matplotlib objects."""
        if not self._is_closed:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
            self.ax1 = None
            self.ax2 = None
            self._is_closed = True

    def plot_series(
        self,
        data: list = [],
        dxf_path: str = None,
        size: tuple = (4, 3),
        title_x: str = "",
        title_y: str = "",
        title_chart: str = "",
        show_legend: bool = False,
        xlim: tuple = None,
        ylim: tuple = None,
        invert_y: bool = False,
        dxf_params: dict = {},
        format_params: dict = None,
    ) -> None:
        """Create a sophisticated multi-series plot with optional overlays.

        Provides a high-level interface for creating complex plots with multiple
        data series, optional DXF overlays, and extensive customization options.

        Parameters
        ----------
        data : list[dict]
            List of data series to plot. Each dict supports:

            * x : array-like
                X-coordinates for the series
            * y : array-like
                Y-coordinates for the series
            * color : str, optional
                Line/marker color (default: 'blue')
            * linestyle : str, optional
                Line style (default: '-')
            * linewidth : float, optional
                Line width (default: 1.0)
            * marker : str, optional
                Marker style (default: 'o')
            * label : str, optional
                Series label for legend
            * note : str or list[str], optional
                Annotations to add

        dxf_path : str, optional
            Path to DXF file for overlay
        size : tuple[float, float], optional
            Plot dimensions in inches (default: (4, 3))
        title_x : str, optional
            X-axis label
        title_y : str, optional
            Y-axis label
        title_chart : str, optional
            Chart title
        show_legend : bool, optional
            Show legend if True (default: False)
        xlim : tuple[float, float], optional
            X-axis limits (min, max)
        ylim : tuple[float, float], optional
            Y-axis limits (min, max)
        invert_y : bool, optional
            Invert Y-axis if True (default: False)

        Examples
        --------
        Simple line plot:
        >>> plotter.plot_series([{
        ...     'x': [0, 1, 2],
        ...     'y': [0, 1, 4],
        ...     'label': 'Quadratic'
        ... }])

        Multiple styled series:
        >>> plotter.plot_series([
        ...     {'x': [0,1], 'y': [0,1], 'color': 'blue'},
        ...     {'x': [0,1], 'y': [1,0], 'color': 'red'}
        ... ], title_chart='Two Lines')

        Notes
        -----
        - Automatically cleans up previous plots
        - DXF overlays are rendered first
        - Series are plotted in order provided
        - Legend only shows if labels exist
        - Thread-safe plot generation
        """
        # Ensure previous figure is closed before creating a new one
        self.close()

        self._initialize_plot(size)
        self.show_legend = show_legend

        if dxf_path:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            for entity in msp.query("LWPOLYLINE"):
                points = entity.get_points()
                x, y = zip(*[(p[0], p[1]) for p in points])
                self.ax1.plot(x, y, **dxf_params)

        for series in data:
            self._plot_single_series(self.ax1, series)

        self._finalize_plot(
            title_x, title_y, title_chart, xlim, ylim, invert_y, format_params
        )

    def add_secondary_y_axis(
        self,
        data: list,
        title_y2: str = "",
        ylim: tuple = None,
        invert_y: bool = False,
        **kwargs,
    ) -> None:
        """
        Add a secondary Y-axis with its own data series.

        Creates a twin axis for plotting additional data series with a different scale.
        Particularly useful for comparing quantities with different units or ranges.

        Parameters
        ----------
        data : list of dict
            List of data series for secondary axis, with same structure as plot_series.
            Additional key:
            - secondary_y : bool, must be True for secondary axis plotting
        title_y2 : str, optional
            Label for secondary Y-axis.
        ylim : tuple, optional
            Y-axis limits for secondary axis (min, max).
        invert_y : bool, optional
            Invert secondary Y-axis direction. Default False.

        Raises
        ------
        RuntimeError
            If called before creating primary plot.

        Notes
        -----
        - Only plots series with secondary_y=True
        - Independent scaling from primary Y-axis
        - Maintains synchronized X-axis with primary plot
        """
        if self.fig is None or self.ax1 is None:
            raise RuntimeError(
                "Primary plot must be created before adding a secondary Y-axis."
            )

        self.ax2 = self.ax1.twinx()

        for series in data:
            if series.get("secondary_y", False):
                self._plot_single_series(self.ax2, series, **kwargs)

        self.ax2.set_ylabel(title_y2)
        # self.ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto'))
        if ylim:
            self.ax2.set_ylim(ylim)
        if invert_y:
            self.ax2.invert_yaxis()

        if any(series.get("label") for series in data) and self.show_legend:
            self.ax2.legend()

    def add_color_bands(
        self, range_band: list, color_band: list, name_band: list, **kwargs
    ) -> None:
        """
        Add color bands to the Y-axis.

        Parameters
        ----------
        range_band : list
            List of ranges for the color bands.
        color_band : list
            List of colors for the bands.
        name_band : list
            List of names for the bands.
        """
        if self.fig is None or self.ax1 is None:
            raise RuntimeError(
                "Primary plot must be created before adding color bands."
            )

        for i in range(len(color_band)):
            self._add_single_color_band(range_band, color_band, name_band, i, **kwargs)

        if self.show_legend:
            self.ax1.legend()

    def add_arrow(
        self,
        data: list,
        position: str = "last",
        angle: float = 0,
        radius: float = 0.1,
        color: str = "red",
    ) -> None:
        """
        Add an arrow indicating a specific angle at the first or last point of the series.

        Parameters
        ----------
        data : list
            List of data series to add arrows to.
        position : str, optional
            Position of the arrow ('first' or 'last'), by default 'last'.
        angle : float, optional
            Angle of the arrow in degrees, by default 0.
        radius : float, optional
            Length of the arrow, by default 0.1.
        color : str, optional
            Color of the arrow, by default 'red'.
        """
        if self.fig is None or self.ax1 is None:
            raise RuntimeError("Primary plot must be created before adding arrows.")

        for series in data:
            self._add_single_arrow(series, position, angle, radius, color)

    def plot_dxf(
        self,
        dxf_path,
        data: list,
        size=(8, 8),
        title_x="",
        title_y="",
        title_chart="",
        show_legend=False,
        xlim=None,
        ylim=None,
        invert_y=False,
        **kwargs,
    ) -> None:
        """
        Plot polylines from a DXF file using the class's figure and axes.

        Parameters
        ----------
        dxf_path : str
            Path to the DXF file.
        size : tuple, optional
            Size of the plot (width, height), by default (8, 8).
        title_x : str, optional
            Title for the X-axis, by default "".
        title_y : str, optional
            Title for the Y-axis, by default "".
        title_chart : str, optional
            Title for the chart, by default "".
        show_legend : bool, optional
            Whether to show the legend, by default False.
        xlim : tuple, optional
            Limits for the X-axis, by default None.
        ylim : tuple, optional
            Limits for the Y-axis, by default None.
        invert_y : bool, optional
            Whether to invert the Y-axis, by default False.
        **kwargs : dict
            Additional style parameters for the lines (color, linestyle, etc.).
        """
        self._initialize_plot(size)
        self.show_legend = show_legend

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        for entity in msp.query("LWPOLYLINE"):
            points = entity.get_points()
            x, y = zip(*[(p[0], p[1]) for p in points])
            self.ax1.plot(x, y, **kwargs)

        for series in data:
            self._plot_single_series(self.ax1, series, **kwargs)

        self._finalize_plot(title_x, title_y, title_chart, xlim, ylim, invert_y)

        if show_legend:
            self.ax1.legend()

    def get_legend(
        self, box_width: int = 4, box_height: int = 2, ncol: int = 1
    ) -> "Drawing":
        """
        Get only the legend as a separate drawing.

        Parameters
        ----------
        box_width : int, optional
            Width of the legend box, by default 4.
        box_height : int, optional
            Height of the legend box, by default 2.
        ncol : int, optional
            Number of columns in the legend, by default 1.

        Returns
        -------
        Drawing
            RLG Drawing object containing the legend.
        """
        if self.fig is None:
            raise RuntimeError(
                "Primary plot must be created before getting the legend."
            )

        return self._create_legend_drawing(box_width, box_height, ncol)

    def get_drawing(self) -> "Drawing":
        """
        Save the plot to a buffer and return an RLG Drawing object.

        Returns
        -------
        Drawing
            RLG Drawing object containing the plot.
        """
        buffer = BytesIO()
        plt.savefig(buffer, format="svg")
        plt.close(self.fig)
        buffer.seek(0)

        return svg2rlg(buffer)

    def get_num_labels(self) -> int:
        """
        Get the number of unique labels in the legend.

        Returns
        -------
        int
            Number of unique legend labels.

        Raises
        ------
        RuntimeError
            If called before creating primary plot.
        """
        if self.fig is None:
            raise RuntimeError(
                "Primary plot must be created before getting number of labels."
            )

        # Get handles and labels from both axes if they exist
        handles, labels = self.ax1.get_legend_handles_labels()
        if self.ax2 is not None:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)

        # Get unique labels using the same logic as _handle_legend
        unique_pairs = self._get_unique_legend_pairs(handles, labels)
        return len(unique_pairs)

    def _initialize_plot(self, size: tuple) -> None:
        """
        Initialize the plot with the given size.

        Parameters
        ----------
        size : tuple
            Size of the plot (width, height).
        """
        self.fig, self.ax1 = plt.subplots(figsize=(size[0], size[1]))

    def _plot_single_series(self, axis: plt.Axes, series: dict, **kwargs) -> None:
        """
        Plot a single data series on the given axis.

        Parameters
        ----------
        axis : plt.Axes
            Axis to plot the series on.
        series : dict
            Data series to plot.
        """
        axis.plot(
            series["x"],
            series["y"],
            color=series.get("color", "blue"),
            linestyle=series.get("linestyle", "-"),
            linewidth=series.get("linewidth", 1),
            marker=series.get("marker", "o"),
            label=series.get("label", None),
            markersize=series.get("markersize", 5),
        )

    def _finalize_plot(
        self,
        title_x: str,
        title_y: str,
        title_chart: str,
        xlim: tuple,
        ylim: tuple,
        invert_y: bool,
        format_params: dict = None,
    ) -> None:
        """
        Finalize the plot by setting titles, limits, and grid.

        Parameters
        ----------
        title_x : str
            Title for the X-axis.
        title_y : str
            Title for the Y-axis.
        title_chart : str
            Title for the chart.
        xlim : tuple
            Limits for the X-axis.
        ylim : tuple
            Limits for the Y-axis.
        invert_y : bool
            Whether to invert the Y-axis.
        format_params : dict, optional
            Dictionary containing formatting parameters.
        """
        self._set_titles(title_x, title_y, title_chart)
        self._apply_format_params(format_params or {})
        self._set_axis_limits(xlim, ylim, invert_y)
        self._handle_legend()

    def _set_titles(self, title_x: str, title_y: str, title_chart: str) -> None:
        """Set all plot titles."""
        self.ax1.set_xlabel(title_x)
        self.ax1.set_ylabel(title_y)
        plt.title(title_chart)

    def _apply_format_params(self, format_params: dict) -> None:
        """Apply formatting parameters to the plot."""
        # Grid visibility
        self.ax1.grid(format_params.get("show_grid", True))

        # Ticks visibility
        if not format_params.get("show_xticks", True):
            self.ax1.set_xticks([])
        if not format_params.get("show_yticks", True):
            self.ax1.set_yticks([])

    def _set_axis_limits(self, xlim: tuple, ylim: tuple, invert_y: bool) -> None:
        """Set axis limits and orientation."""
        if xlim:
            self.ax1.set_xlim(xlim)
        if ylim:
            self.ax1.set_ylim(ylim)
        if invert_y:
            self.ax1.invert_yaxis()

    def _handle_legend(self) -> None:
        """Handle legend visibility based on conditions and remove duplicates."""
        if not self.show_legend:
            return

        # Get handles and labels from both axes if they exist
        handles, labels = self.ax1.get_legend_handles_labels()
        if self.ax2 is not None:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)

        if handles and labels:
            # Remove duplicates while preserving order
            unique_pairs = self._get_unique_legend_pairs(handles, labels)
            if unique_pairs:
                unique_handles, unique_labels = zip(*unique_pairs)
                if self.ax2 is not None:
                    # If we have two axes, place legend between them
                    self.ax1.legend(
                        unique_handles,
                        unique_labels,
                        loc="center right",
                        bbox_to_anchor=(1.15, 0.5),
                    )
                else:
                    self.ax1.legend(unique_handles, unique_labels)

    def _get_unique_legend_pairs(self, handles, labels):
        """Get unique handle-label pairs while preserving order of appearance.

        Parameters
        ----------
        handles : list
            List of legend handles
        labels : list
            List of legend labels

        Returns
        -------
        list
            List of unique (handle, label) pairs
        """
        seen = set()
        unique_pairs = []

        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_pairs.append((handle, label))

        return unique_pairs

    def _add_single_color_band(
        self, range_band: list, color_band: list, name_band: list, index: int, **kwargs
    ) -> None:
        """
        Add a single color band to the Y-axis.

        Parameters
        ----------
        range_band : list
            List of ranges for the color bands.
        color_band : list
            List of colors for the bands.
        name_band : list
            List of names for the bands.
        index : int
            Index of the current band.
        """
        self.ax1.axhspan(
            range_band[index],
            range_band[index + 1],
            facecolor=color_band[index],
            **kwargs,
        )
        self.ax1.axhline(
            y=range_band[index + 1],
            color=color_band[index],
            linestyle="--",
            label=name_band[index],
        )

    def add_notes(
        self, x_point: float, y_point: float, dx: float, dy: float, series: dict
    ) -> None:
        """
        Add text annotations to the plot with automatic positioning.

        Handles single or multiple text annotations with customizable appearance
        and automatic collision avoidance using adjustText.

        Parameters
        ----------
        x_point : float
            X-coordinate of anchor point.
        y_point : float
            Y-coordinate of anchor point.
        dx : float
            X-offset from anchor point.
        dy : float
            Y-offset from anchor point.
        series : dict
            Configuration dictionary containing:
            - note : str or list
                Text to display. Single string or list of strings.
            - note_style : dict, optional
                Style parameters:
                - fontsize : int
                - bbox : dict
                - other matplotlib text properties
            - adjust_text_params : dict, optional
                Parameters for adjustText algorithm

        Notes
        -----
        - Uses path effects for better text visibility
        - Automatically handles multiple notes spacing
        - Falls back to default styles if not specified
        - Single notes skip adjustment for better performance
        """
        notes = series.get("note")
        if not notes:
            return

        # Convert single note to list
        if isinstance(notes, str):
            notes = [notes]

        # Get fontsize from data or use default
        fontsize = series.get(
            "fontsize", self.default_styles.get("note_style", {}).get("fontsize", 10)
        )

        # Get default styles from TOML
        default_note_style = self.default_styles.get("note_style", {})
        default_adjust_params = self.default_styles.get("adjust_text_params", {})

        # Create note style with path effects
        note_style = {
            "fontsize": fontsize,
            "bbox": default_note_style.get("bbox", {}),
            "path_effects": [
                path_effects.withStroke(
                    linewidth=default_note_style.get("linewidth", 0.5),
                    foreground=default_note_style.get("foreground", "white"),
                )
            ],
        }

        # Override with user provided styles
        if "note_style" in series:
            note_style.update(series["note_style"])

        # Get adjust text parameters
        adjust_text_params = default_adjust_params.copy()
        if "adjust_text_params" in series:
            adjust_text_params.update(series["adjust_text_params"])

        texts = []
        base_x = x_point + dx * 1.2
        base_y = y_point + dy * 1.2

        for note in notes:
            text = self.ax1.text(base_x, base_y, note, **note_style)
            texts.append(text)

        # Special handling for single text to avoid adjust_text issues
        if len(texts) == 1:
            return

        # Only use adjust_text for multiple texts
        adjust_text(
            texts,
            x=[base_x] * len(texts),
            y=[base_y] * len(texts),
            **adjust_text_params,
        )

    def _add_single_arrow(
        self,
        series: dict,
        position: str,
        angle: float,
        radius: float,
        color: str,
    ) -> None:
        """
        Add a single arrow to the plot.

        Parameters
        ----------
        series : dict
            Data series to add the arrow to.
        position : str
            Position of the arrow ('first' or 'last').
        angle : float
            Angle of the arrow in degrees.
        radius : float
            Length of the arrow.
        color : str
            Color of the arrow.
        """
        x = series["x"]
        y = series["y"]
        angle = series.get("angle", angle)

        if position == "first":
            x_point, y_point = x[0], y[0]
        elif position == "last":
            x_point, y_point = x[-1], y[-1]
        else:
            raise ValueError("Invalid position value. Use 'first' or 'last'.")

        dx = radius * np.cos(np.radians(angle))
        dy = radius * np.sin(np.radians(angle))
        arrow = patches.FancyArrowPatch(
            (x_point, y_point),
            (x_point + dx, y_point + dy),
            color=color,
            arrowstyle="->",
            mutation_scale=5,
        )
        self.ax1.add_patch(arrow)

    def _create_legend_drawing(
        self, box_width: int, box_height: int, ncol: int = 1
    ) -> "Drawing":
        """Create a drawing of the legend with unique labels."""
        fig_legend = plt.figure(figsize=(box_width, box_height))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis("off")

        # Get handles and labels from both axes if they exist
        handles, labels = self.ax1.get_legend_handles_labels()
        if self.ax2 is not None:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)

        # Get unique labels while preserving order
        unique_pairs = self._get_unique_legend_pairs(handles, labels)
        if unique_pairs:
            handles, labels = zip(*unique_pairs)
            wrapped_labels = [
                self._wrap_label(label, box_width * 5) for label in labels
            ]
            legend = fig_legend.legend(handles, wrapped_labels, loc="center", ncol=ncol)

            fig_legend.canvas.draw()
            bbox = legend.get_window_extent().transformed(
                fig_legend.dpi_scale_trans.inverted()
            )
            fig_legend.set_size_inches(bbox.width, bbox.height)

            buffer = BytesIO()
            fig_legend.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0)
            plt.close(fig_legend)
            buffer.seek(0)
            return svg2rlg(buffer)

        return None

    def _wrap_label(self, text: str, max_width: int) -> str:
        """
        Wrap the label text to fit within the given width.

        Parameters
        ----------
        text : str
            Label text to wrap.
        max_width : int
            Maximum width of the label.

        Returns
        -------
        str
            Wrapped label text.
        """
        import textwrap

        return "\n".join(textwrap.wrap(text, width=max_width))


class PlotMerger:
    """A utility class for combining multiple plots into a grid layout.

    Provides functionality to arrange multiple plot objects in a customizable
    grid layout with automatic scaling and positioning.

    Parameters
    ----------
    fig_size : tuple[int, int], optional
        Maximum figure dimensions (width, height) in inches
        Default is (8, 6)

    Attributes
    ----------
    fig_width : float
        Figure width in points
    fig_height : float
        Figure height in points
    objects : list
        Plot objects to arrange
    positions : list[tuple[int, int]]
        Grid positions for each object
    spans : list[tuple[int, int]]
        Cell spans for each object
    grid_dims : tuple[int, int], optional
        Grid dimensions (rows, cols)
    row_ratios : list[float], optional
        Relative heights of rows
    col_ratios : list[float], optional
        Relative widths of columns

    Examples
    --------
    >>> merger = PlotMerger(fig_size=(10, 8))
    >>> merger.create_grid(2, 2, row_ratios=[0.6, 0.4], col_ratios=[0.5, 0.5])
    >>> merger.add_object(plot1, (0, 0))
    >>> merger.add_object(plot2, (0, 1))
    >>> drawing = merger.build()

    Notes
    -----
    - Automatically handles object scaling
    - Preserves aspect ratios
    - Memory-efficient object handling
    """

    def __init__(self, fig_size: Tuple[int, int] = (8, 6)):
        """Initialize PlotMerger with figure dimensions in points."""
        self._initialize_dimensions(fig_size)
        self.objects: List = []
        self.positions: List[Tuple[int, int]] = []
        self.spans: List[Tuple[int, int]] = []
        self.grid_dims: Tuple[int, int] = None
        self.col_ratios: List[float] = None
        self.row_ratios: List[float] = None

    def _initialize_dimensions(self, fig_size: Tuple[int, int]) -> None:
        """Convert inches to points and store dimensions."""
        POINTS_PER_INCH = 72
        self.fig_width = fig_size[0] * POINTS_PER_INCH
        self.fig_height = fig_size[1] * POINTS_PER_INCH

    def add_object(
        self, obj, position: Tuple[int, int], span: Tuple[int, int] = (1, 1)
    ) -> None:
        """Add an object to the grid with position and span."""
        self.objects.append(obj)
        self.positions.append(position)
        self.spans.append(span)

    def create_grid(
        self,
        rows: int,
        cols: int,
        row_ratios: List[float] = None,
        col_ratios: List[float] = None,
    ) -> "PlotMerger":
        """Set grid dimensions and optional row/column ratios.

        Parameters
        ----------
        rows : int
            Number of rows in the grid
        cols : int
            Number of columns in the grid
        row_ratios : List[float], optional
            Relative heights of each row. Must sum to 1 if provided.
        col_ratios : List[float], optional
            Relative widths of each column. Must sum to 1 if provided.

        Returns
        -------
        PlotMerger
            Self for method chaining
        """
        self.grid_dims = (rows, cols)

        # Validate and set row ratios
        if row_ratios is not None:
            if len(row_ratios) != rows:
                raise ValueError(f"row_ratios must have length {rows}")
            if abs(sum(row_ratios) - 1.0) > 1e-6:
                raise ValueError("row_ratios must sum to 1")
            self.row_ratios = row_ratios
        else:
            self.row_ratios = [1 / rows] * rows

        # Validate and set column ratios
        if col_ratios is not None:
            if len(col_ratios) != cols:
                raise ValueError(f"col_ratios must have length {cols}")
            if abs(sum(col_ratios) - 1.0) > 1e-6:
                raise ValueError("col_ratios must sum to 1")
            self.col_ratios = col_ratios
        else:
            self.col_ratios = [1 / cols] * cols

        return self

    def build(self, color_border: str = "white", cell_spacing: float = 10.0) -> Drawing:
        """Build and return the combined figure."""
        if not self._is_grid_initialized():
            raise ValueError("Must call create_grid before build")

        drawing = Drawing(self.fig_width, self.fig_height)
        cell_dimensions = self._get_cell_dimensions(cell_spacing)

        for obj_data in self._iter_objects():
            self._add_object_to_drawing(
                drawing, obj_data, cell_dimensions, color_border, cell_spacing
            )

        return drawing

    def _is_grid_initialized(self) -> bool:
        """Check if grid dimensions are set."""
        return self.grid_dims is not None

    def _get_cell_dimensions(self, spacing: float) -> Tuple[List[float], List[float]]:
        """Calculate cell width and height with spacing and ratios."""
        rows, cols = self.grid_dims

        # Calculate total available space after spacing
        available_width = self.fig_width - (cols - 1) * spacing
        available_height = self.fig_height - (rows - 1) * spacing

        # Calculate cell dimensions based on ratios
        cell_widths = [ratio * available_width for ratio in self.col_ratios]
        cell_heights = [ratio * available_height for ratio in self.row_ratios]

        return cell_widths, cell_heights

    def _iter_objects(self) -> Tuple:
        """Iterate over object data as tuples."""
        return zip(self.objects, self.positions, self.spans)

    def _add_object_to_drawing(
        self,
        drawing: Drawing,
        obj_data: Tuple,
        cell_dims: Tuple[List[float], List[float]],
        color_border: str,
        cell_spacing: float,
    ) -> None:
        """Add a single object and its border to the drawing."""
        obj, (row, col), (row_span, col_span) = obj_data
        cell_widths, cell_heights = cell_dims

        # Calculate object dimensions and transformations
        span_dims = self._get_span_size(cell_dims, row, col, row_span, col_span)
        scale = self._get_scale_factor(span_dims, (obj.width, obj.height))
        scaled_size = (obj.width * scale, obj.height * scale)
        position = self._get_object_position(
            cell_dims, row, col, row_span, span_dims, scaled_size, cell_spacing
        )

        # Add transformed object
        group = self._create_object_group(obj, scale, position, scaled_size)
        drawing.add(group)

        # Add cell border
        border = self._create_border(
            cell_dims, row, col, row_span, col_span, color_border, cell_spacing
        )
        drawing.add(border)

    def _get_span_size(
        self,
        cell_dims: Tuple[List[float], List[float]],
        row: int,
        col: int,
        row_span: int,
        col_span: int,
    ) -> Tuple[float, float]:
        """Calculate total span width and height."""
        cell_widths, cell_heights = cell_dims

        # Sum the widths and heights of spanned cells
        span_width = sum(cell_widths[col : col + col_span])
        span_height = sum(
            cell_heights[self.grid_dims[0] - row - row_span : self.grid_dims[0] - row]
        )

        return span_width, span_height

    def _get_scale_factor(
        self, container: Tuple[float, float], content: Tuple[float, float]
    ) -> float:
        """Calculate scale factor preserving aspect ratio."""
        return min(container[0] / content[0], container[1] / content[1])

    def _get_object_position(
        self,
        cell_dims: Tuple[List[float], List[float]],
        row: int,
        col: int,
        row_span: int,
        span_dims: Tuple[float, float],
        scaled_size: Tuple[float, float],
        spacing: float,
    ) -> Tuple[float, float]:
        """Calculate centered position for the object."""
        cell_widths, cell_heights = cell_dims
        span_width, span_height = span_dims
        scaled_width, scaled_height = scaled_size

        # Calculate x position based on column ratios
        x = sum(cell_widths[:col]) + col * spacing
        x += (span_width - scaled_width) / 2

        # Calculate y position based on row ratios
        y = (
            sum(cell_heights[: self.grid_dims[0] - row - row_span])
            + (self.grid_dims[0] - row - row_span) * spacing
        )
        y += (span_height - scaled_height) / 2

        return x, y

    def _create_object_group(
        self,
        obj,
        scale: float,
        position: Tuple[float, float],
        scaled_size: Tuple[float, float],
    ) -> Group:
        """Create a transformed group containing the object."""
        group = Group()
        obj.scale(scale, scale)
        obj.translate(-obj.width / 2, -obj.height / 2)
        group.add(obj)
        x, y = position
        scaled_width, scaled_height = scaled_size
        group.translate(x + scaled_width / 2, y + scaled_height / 2)
        return group

    def _create_border(
        self,
        cell_dims: Tuple[List[float], List[float]],
        row: int,
        col: int,
        row_span: int,
        col_span: int,
        color: str,
        spacing: float,
    ) -> Rect:
        """Create a border rectangle for the cell."""
        cell_widths, cell_heights = cell_dims
        x = sum(cell_widths[:col]) + col * spacing
        y = (
            sum(cell_heights[: self.grid_dims[0] - row - row_span])
            + (self.grid_dims[0] - row - row_span) * spacing
        )
        width = sum(cell_widths[col : col + col_span]) + spacing * (col_span - 1)
        height = sum(
            cell_heights[self.grid_dims[0] - row - row_span : self.grid_dims[0] - row]
        ) + spacing * (row_span - 1)
        return Rect(x, y, width, height, strokeColor=color, fillColor=None)

    @staticmethod
    def scale_figure(figure: Drawing, size: Tuple[int, int] = (2, 2)) -> Drawing:
        """Scale a figure to the specified size using PlotMerger.

        A convenience method that creates a single-cell grid and places the figure in it,
        effectively scaling the figure to the desired size.

        Parameters
        ----------
        figure : Drawing
            The figure to scale
        size : tuple[int, int], optional
            Target size in inches (width, height), default (2, 2)

        Returns
        -------
        Drawing
            The scaled figure

        Examples
        --------
        >>> original_fig = some_drawing_object
        >>> scaled_fig = PlotMerger.scale_figure(original_fig, size=(2, 2))
        """
        grid = PlotMerger(fig_size=size)
        grid.create_grid(1, 1)
        grid.add_object(figure, (0, 0))
        return grid.build(color_border="white")
