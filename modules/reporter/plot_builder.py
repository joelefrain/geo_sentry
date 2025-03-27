import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.utils.plot_config import PlotConfig

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import ezdxf
import numpy as np
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics.shapes import Drawing, Group, Rect
from typing import Tuple, List

PlotConfig.setup_matplotlib()


class PlotBuilder:
    def __init__(self):
        """
        Initialize the Plotter class.
        """
        self.fig: plt.Figure = None
        self.ax1: plt.Axes = None
        self.ax2: plt.Axes = None
        self.show_legend: bool = False
        self._is_closed: bool = False

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
        **kwargs
    ) -> None:
        # Ensure previous figure is closed before creating a new one
        self.close()
        """
        Create a plot from multiple data series.

        Parameters
        ----------
        data : list
            List of data series to plot.
        size : tuple, optional
            Size of the plot (width, height), by default (4, 3).
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
        format_params : dict, optional
            Dictionary containing formatting parameters:
            - show_grid (bool): Whether to show grid lines
            - show_xticks (bool): Whether to show x-axis ticks
            - show_yticks (bool): Whether to show y-axis ticks
        """
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
            self._plot_single_series(self.ax1, series, **kwargs)

        self._finalize_plot(
            title_x, title_y, title_chart, xlim, ylim, invert_y, format_params
        )

        if show_legend:
            self.ax1.legend()

    def add_secondary_y_axis(
        self,
        data: list,
        title_y2: str = "",
        ylim: tuple = None,
        invert_y: bool = False,
        **kwargs
    ) -> None:
        """
        Add a secondary Y-axis to the plot.

        Parameters
        ----------
        data : list
            List of data series to plot on the secondary Y-axis.
        title_y2 : str, optional
            Title for the secondary Y-axis, by default "".
        ylim : tuple, optional
            Limits for the secondary Y-axis, by default None.
        invert_y : bool, optional
            Whether to invert the secondary Y-axis, by default False.
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
        **kwargs
    ):
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

    def get_legend(self, box_width: int = 4, box_height: int = 2) -> "Drawing":
        """
        Get only the legend as a separate drawing.

        Parameters
        ----------
        box_width : int, optional
            Width of the legend box, by default 4.
        box_height : int, optional
            Height of the legend box, by default 2.

        Returns
        -------
        Drawing
            RLG Drawing object containing the legend.
        """
        if self.fig is None:
            raise RuntimeError(
                "Primary plot must be created before getting the legend."
            )

        return self._create_legend_drawing(box_width, box_height)

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
            linestyle=series.get("linetype", "-"),
            linewidth=series.get("lineweight", 1),
            marker=series.get("marker", "o"),
            label=series.get("label", None),
            **kwargs,
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
            Dictionary containing formatting parameters:
            - show_grid (bool): Whether to show grid lines
            - show_xticks (bool): Whether to show x-axis ticks
            - show_yticks (bool): Whether to show y-axis ticks
        """
        self.ax1.set_xlabel(title_x)
        self.ax1.set_ylabel(title_y)

        # Apply formatting based on format_params
        if format_params is None:
            format_params = {}

        # Grid visibility
        show_grid = format_params.get("show_grid", True)
        self.ax1.grid(show_grid)

        # Ticks visibility
        if not format_params.get("show_xticks", True):
            self.ax1.set_xticks([])
        if not format_params.get("show_yticks", True):
            self.ax1.set_yticks([])

        if xlim:
            self.ax1.set_xlim(xlim)
        if ylim:
            self.ax1.set_ylim(ylim)
        if invert_y:
            self.ax1.invert_yaxis()
        plt.title(title_chart)

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
            Data series to add the arrow to. Can include:
            - note: str, optional
                Text note to add near the arrow
            - fontsize: int, optional
                Font size for the note text
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
            mutation_scale=10,
        )
        self.ax1.add_patch(arrow)

        # Add note if provided
        note = series.get('note')
        if note:
            fontsize = series.get('fontsize', 8)
            # Calculate optimal note position based on arrow angle
            note_x = x_point + dx * 1.2
            note_y = y_point + dy * 1.2
            
            # Add text with automatic positioning
            self.ax1.annotate(
                note,
                xy=(x_point, y_point),
                xytext=(note_x, note_y),
                fontsize=fontsize,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                ha='center',
                va='center'
            )


    def _create_legend_drawing(self, box_width: int, box_height: int) -> "Drawing":
        """
        Create a drawing of the legend.

        Parameters
        ----------
        box_width : int
            Width of the legend box.
        box_height : int
            Height of the legend box.

        Returns
        -------
        Drawing
            RLG Drawing object containing the legend.
        """
        fig_legend = plt.figure(figsize=(box_width, box_height))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis("off")
        handles, labels = self.ax1.get_legend_handles_labels()

        wrapped_labels = [self._wrap_label(label, box_width * 10) for label in labels]
        legend = fig_legend.legend(handles, wrapped_labels, loc="center")

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
    def __init__(self, fig_size: Tuple[int, int] = (8, 6)):
        """
        Initialize the structure to combine plots.

        Parameters
        ----------
        fig_size : tuple of int, optional
            Maximum figure size in inches (width, height). Default is (8, 6).
        """
        self.fig_width = fig_size[0] * 72  # Convert inches to points (1 in = 72 pt)
        self.fig_height = fig_size[1] * 72
        self.objects: List = []
        self.positions: List[Tuple[int, int]] = []
        self.spans: List[Tuple[int, int]] = []
        self.rows: int = None
        self.cols: int = None

    def add_object(
        self, obj, position: Tuple[int, int], span: Tuple[int, int] = (1, 1)
    ):
        """
        Add an object to a specific position in the grid.

        Parameters
        ----------
        obj : svg2rlg object
            The SVG object to add.
        position : tuple of int
            Tuple (row, col) for the position in the grid.
        span : tuple of int, optional
            Tuple (row_span, col_span) to extend the subplot. Default is (1, 1).
        """
        self.objects.append(obj)
        self.positions.append(position)
        self.spans.append(span)

    def create_grid(self, rows: int, cols: int):
        """
        Create the grid structure.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.

        Returns
        -------
        self : PlotMerger
            The instance itself.
        """
        self.rows = rows
        self.cols = cols
        return self

    def build(self, color_border: str = "white", cell_spacing: float = 10.0) -> Drawing:
        """
        Build the final figure with all subplots.

        Parameters
        ----------
        color_border : str, optional
            Color of the cell borders. Default is 'white'.
        cell_spacing : float, optional
            Spacing between the cells in points. Default is 10.0.

        Returns
        -------
        drawing : Drawing
            A vectorized RLG object with the combined figure.
        """
        if self.rows is None or self.cols is None:
            raise ValueError("Must call create_grid before build")

        cell_width, cell_height = self._calculate_cell_dimensions(cell_spacing)
        drawing = Drawing(self.fig_width, self.fig_height)

        for obj, (row, col), (row_span, col_span) in zip(
            self.objects, self.positions, self.spans
        ):
            obj_width, obj_height = obj.width, obj.height
            span_width, span_height = self._get_span_dimensions(
                cell_width, cell_height, row_span, col_span
            )
            scale_factor = self._calculate_scale_factor(
                span_width, span_height, obj_width, obj_height
            )
            scaled_width, scaled_height = (
                obj_width * scale_factor,
                obj_height * scale_factor,
            )
            x, y = self._calculate_position(
                cell_width,
                cell_height,
                row,
                col,
                row_span,
                span_width,
                span_height,
                scaled_width,
                scaled_height,
                cell_spacing,
            )

            group = self._create_transformed_group(
                obj, scale_factor, x, y, scaled_width, scaled_height
            )
            drawing.add(group)

            cell_border = self._create_cell_border(
                cell_width,
                cell_height,
                row,
                col,
                row_span,
                col_span,
                color_border=color_border,
                cell_spacing=cell_spacing,
            )
            drawing.add(cell_border)

        return drawing

    def _calculate_cell_dimensions(self, cell_spacing: float) -> Tuple[float, float]:
        """
        Calculate the dimensions of each cell in the grid.

        Parameters
        ----------
        cell_spacing : float
            Spacing between the cells in points.

        Returns
        -------
        cell_width : float
            Width of each cell.
        cell_height : float
            Height of each cell.
        """
        cell_width = (self.fig_width - (self.cols - 1) * cell_spacing) / self.cols
        cell_height = (self.fig_height - (self.rows - 1) * cell_spacing) / self.rows
        return cell_width, cell_height

    def _get_span_dimensions(
        self, cell_width: float, cell_height: float, row_span: int, col_span: int
    ) -> Tuple[float, float]:
        """
        Calculate the dimensions of a span.

        Parameters
        ----------
        cell_width : float
            Width of each cell.
        cell_height : float
            Height of each cell.
        row_span : int
            Number of rows to span.
        col_span : int
            Number of columns to span.

        Returns
        -------
        span_width : float
            Width of the span.
        span_height : float
            Height of the span.
        """
        return (cell_width * col_span, cell_height * row_span)

    def _calculate_scale_factor(
        self, span_width: float, span_height: float, obj_width: float, obj_height: float
    ) -> float:
        """
        Calculate the scale factor to maintain aspect ratio.

        Parameters
        ----------
        span_width : float
            Width of the span.
        span_height : float
            Height of the span.
        obj_width : float
            Width of the object.
        obj_height : float
            Height of the object.

        Returns
        -------
        scale_factor : float
            Scale factor to maintain aspect ratio.
        """
        scale_x = span_width / obj_width
        scale_y = span_height / obj_height
        return min(scale_x, scale_y)

    def _calculate_position(
        self,
        cell_width: float,
        cell_height: float,
        row: int,
        col: int,
        row_span: int,
        span_width: float,
        span_height: float,
        scaled_width: float,
        scaled_height: float,
        cell_spacing: float,
    ) -> Tuple[float, float]:
        """
        Calculate the position to center the object in the cell.

        Parameters
        ----------
        cell_width : float
            Width of each cell.
        cell_height : float
            Height of each cell.
        row : int
            Row index.
        col : int
            Column index.
        row_span : int
            Number of rows to span.
        span_width : float
            Width of the span.
        span_height : float
            Height of the span.
        scaled_width : float
            Scaled width of the object.
        scaled_height : float
            Scaled height of the object.
        cell_spacing : float
            Spacing between the cells in points.

        Returns
        -------
        x : float
            X position.
        y : float
            Y position.
        """
        x = col * (cell_width + cell_spacing) + (span_width - scaled_width) / 2
        y = (self.rows - row - row_span) * (cell_height + cell_spacing) + (
            span_height - scaled_height
        ) / 2
        return x, y

    def _create_transformed_group(
        self,
        obj,
        scale_factor: float,
        x: float,
        y: float,
        scaled_width: float,
        scaled_height: float,
    ) -> Group:
        """
        Create a transformed group with the scaled and positioned object.

        Parameters
        ----------
        obj : svg2rlg object
            The SVG object to transform.
        scale_factor : float
            Scale factor to apply.
        x : float
            X position.
        y : float
            Y position.
        scaled_width : float
            Scaled width of the object.
        scaled_height : float
            Scaled height of the object.

        Returns
        -------
        group : Group
            Transformed group with the object.
        """
        group = Group()
        obj.scale(scale_factor, scale_factor)
        obj.translate(-obj.width / 2, -obj.height / 2)
        group.add(obj)
        group.translate(x + scaled_width / 2, y + scaled_height / 2)
        return group

    def _create_cell_border(
        self,
        cell_width: float,
        cell_height: float,
        row: int,
        col: int,
        row_span: int,
        col_span: int,
        color_border: str,
        cell_spacing: float,
    ) -> Rect:
        """
        Create a border for the cell.

        Parameters
        ----------
        cell_width : float
            Width of each cell.
        cell_height : float
            Height of each cell.
        row : int
            Row index.
        col : int
            Column index.
        row_span : int
            Number of rows to span.
        col_span : int
            Number of columns to span.
        color_border : str
            Color of the border.
        cell_spacing : float
            Spacing between the cells in points.

        Returns
        -------
        cell_border : Rect
            Rectangle representing the cell border.
        """
        border_x = col * (cell_width + cell_spacing)
        border_y = (self.rows - row - row_span) * (cell_height + cell_spacing)
        span_width = cell_width * col_span + cell_spacing * (col_span - 1)
        span_height = cell_height * row_span + cell_spacing * (row_span - 1)
        return Rect(
            border_x,
            border_y,
            span_width,
            span_height,
            strokeColor=color_border,
            fillColor=None,
        )
