import rasterio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from io import BytesIO
from reportlab.graphics.shapes import Drawing

from svglib.svglib import svg2rlg
from adjustText import adjust_text
from rasterio.warp import transform_bounds

from libs.utils.config_variables import CHART_CONFIG_DIR

from libs.utils.config_loader import load_toml
from libs.utils.config_plot import PlotConfig
from libs.utils.plot_helpers import dxfParser, parse_path_effects


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
            self.plot_style = load_toml(CHART_CONFIG_DIR, style_file)
        except Exception as e:
            self.plot_style = {}
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
        tif_path: str = None,
        project_epsg: int = None,
        size: tuple = (4, 3),
        title_x: str = "",
        title_y: str = "",
        title_chart: str = "",
        show_legend: bool = False,
        xlim: tuple = None,
        ylim: tuple = None,
        invert_y: bool = False,
        format_params: dict = None,
        dxf_params: dict = None,
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
        format_params : dict, optional
            Formatting parameters for the plot
        dxf_params : dict, optional
            Additional parameters for DXF plotting

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

        # --- Añadir raster GeoTIFF al fondo si se proporciona ---
        if tif_path:
            self._plot_tif_utm(tif_path, project_epsg)

        if dxf_path:
            self._plot_dxf(dxf_path, dxf_params)

        for series in data:
            self._plot_single_series(self.ax1, series)

        self._finalize_plot(
            title_x, title_y, title_chart, xlim, ylim, invert_y, format_params
        )

    def _plot_tif_utm(self, tif_path: str, project_epsg: int = None) -> None:
        """
        Añade un GeoTIFF georreferenciado como fondo, alineado a coordenadas UTM (x, y).
        Si utm_epsg no se especifica, se intenta deducir del raster.
        """
        try:
            with rasterio.open(tif_path) as src:
                img = src.read([1, 2, 3]) if src.count >= 3 else src.read(1)
                bounds = src.bounds
                src_crs = src.crs

                # Determinar EPSG destino (UTM)
                if project_epsg is None:
                    # Si el raster ya está en UTM, usar ese
                    if src_crs and src_crs.is_projected:
                        project_epsg = int(str(src_crs.to_epsg()))
                    else:
                        raise ValueError(
                            "Debe especificar utm_epsg para transformar el raster a UTM."
                        )

                # Transformar bounds a UTM si es necesario
                if src_crs.to_epsg() != project_epsg:
                    utm_bounds = transform_bounds(
                        src_crs,
                        f"EPSG:{project_epsg}",
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )
                else:
                    utm_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

                extent = (utm_bounds[0], utm_bounds[2], utm_bounds[1], utm_bounds[3])
                tif_params = self.plot_style.get("tif_params", {})

                self.ax1.imshow(
                    img.transpose(1, 2, 0) if src.count >= 3 else img,
                    extent=extent,
                    **tif_params,
                )
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el GeoTIFF '{tif_path}': {e}") from e

    def _plot_dxf(self, dxf_path: str, dxf_params: dict = None) -> None:
        """
        Plotea todas las entidades relevantes del DXF usando dxfParser.
        Si dxf_params se proporciona, fuerza los valores de color, linewidth y linestyle.
        Ejemplo de dxf_params: {"color": "#000000", "linewidth": 1, "linestyle": "-"}
        """
        parser = dxfParser(dxf_path)
        entities = parser.parse_entities()
        ax = self.ax1

        for ent in entities:
            color = (
                dxf_params.get("color")
                if dxf_params and "color" in dxf_params
                else ent.get("color", "k")
            )
            linewidth = (
                dxf_params.get("linewidth")
                if dxf_params and "linewidth" in dxf_params
                else ent.get("linewidth", 1)
            )
            linestyle = (
                dxf_params.get("linestyle")
                if dxf_params and "linestyle" in dxf_params
                else ent.get("linestyle", "-")
            )

            if ent["type"] in ("LINE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE"):
                x, y = zip(*ent["points"])
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
                # Cierre de polilínea si corresponde
                if ent.get("closed", False):
                    ax.plot(
                        [x[-1], x[0]],
                        [y[-1], y[0]],
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                    )
            elif ent["type"] == "HATCH":
                x, y = zip(*ent["points"])
                ax.fill(x, y, color=color, alpha=ent.get("alpha", 1.0))
            elif ent["type"] == "HATCH_EDGE":
                x, y = zip(*ent["points"])
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
            elif ent["type"] == "ARC":
                arc = patches.Arc(
                    ent["center"],
                    2 * ent["radius"],
                    2 * ent["radius"],
                    angle=0,
                    theta1=ent["start_angle"],
                    theta2=ent["end_angle"],
                    edgecolor=color,
                    linewidth=linewidth,
                )
                ax.add_patch(arc)
            elif ent["type"] == "CIRCLE":
                circ = patches.Circle(
                    ent["center"],
                    ent["radius"],
                    edgecolor=color,
                    facecolor="none",
                    linewidth=linewidth,
                )
                ax.add_patch(circ)
            elif ent["type"] == "POINT":
                ax.plot(
                    [ent["point"][0]],
                    [ent["point"][1]],
                    marker="o",
                    color=color,
                )
            elif ent["type"] == "TEXT":
                ax.text(
                    ent["position"][0],
                    ent["position"][1],
                    ent["text"],
                    color=color,
                    fontsize=ent.get("height", 8),
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

    def get_colorbar(
        self,
        box_width: int = 4,
        box_height: int = 0.5,
        label: str = "",
        vmin: float = None,
        vmax: float = None,
        colors: list = None,
        thresholds: list = None,
        cmap: str = "cool",
        type_colorbar: str = "continuous",
    ) -> "Drawing":
        """
        Create a horizontal colorbar with continuous or discrete color mapping.

        Parameters
        ----------
        box_width : int
            Width of the colorbar box.
        box_height : int
            Height of the colorbar box.
        label : str
            Label for the colorbar.
        vmin : float, optional
            Minimum value for colorbar scale.
        vmax : float, optional
            Maximum value for colorbar scale.
        colors : list, optional
            List of colors to create custom colormap.
        thresholds : list, optional
            List of thresholds for discrete colorbar.
        cmap : str, optional
            Name of matplotlib colormap to use if colors not provided (default: 'cool').
        type_colorbar : str, optional
            Type of colorbar: "continuous" (default) or "discrete".

        Returns
        -------
        Drawing
            RLG Drawing object containing the colorbar.
        """
        if self.fig is None:
            raise RuntimeError(
                "Primary plot must be created before getting the colorbar."
            )

        # Create a new figure for the colorbar
        fig_cbar = plt.figure(figsize=(box_width, box_height))
        ax_cbar = fig_cbar.add_axes([0.05, 0.5, 0.9, 0.3])

        if type_colorbar == "continuous":
            # Create colormap from colors if provided, otherwise use named cmap
            if colors:
                import matplotlib.colors as mcolors

                cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
            else:
                cmap = plt.get_cmap(cmap)

            # Create a scalar mappable
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # Create continuous colorbar
            plt.colorbar(sm, cax=ax_cbar, orientation="horizontal", label=label)

        elif type_colorbar == "discrete":
            if thresholds is None or colors is None:
                raise ValueError(
                    "For discrete colorbar, both 'thresholds' and 'colors' must be provided."
                )
            if len(colors) != len(thresholds) + 1:
                raise ValueError(
                    "The number of colors must be one more than the number of thresholds."
                )

            # Create a discrete colormap
            import matplotlib.colors as mcolors

            cmap = mcolors.ListedColormap(colors)
            bounds = [vmin] + thresholds + [vmax]
            bounds = sorted(bounds)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Create discrete colorbar
            from matplotlib.colorbar import ColorbarBase

            cb = ColorbarBase(
                ax_cbar,
                cmap=cmap,
                norm=norm,
                boundaries=bounds,
                orientation="horizontal",
                label=label,
            )
            cb.set_ticks(bounds)
            cb.set_ticklabels([f"{b:.2f}" for b in bounds])

        # Save to buffer and convert to Drawing
        buffer = BytesIO()
        fig_cbar.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0)
        plt.close(fig_cbar)
        buffer.seek(0)
        return svg2rlg(buffer)

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

    def add_arrow(
        self,
        data: list,
        position: str = "last",
        radius: float = 0.1,
    ) -> None:
        """
        Add arrows to the plot based on the data series.

        Parameters
        ----------
        data : list
            List of data series to add arrows to.
        position : str, optional
            Position of the arrow ('first' or 'last'), by default 'last'.
        radius : float, optional
            Length of the arrow, by default 0.1.
        """
        if self.fig is None or self.ax1 is None:
            raise RuntimeError("Primary plot must be created before adding arrows.")

        # Load arrow parameters from TOML configuration
        arrow_params = self.plot_style.get("arrow_params", {})

        for series in data:
            x = series["x"]
            y = series["y"]
            angle = series["angle"]

            if position == "first":
                x_point, y_point = x[0], y[0]
            elif position == "last":
                x_point, y_point = x[-1], y[-1]
            else:
                raise ValueError("Invalid position value. Use 'first' or 'last'.")

            dx = radius * np.cos(np.radians(angle))
            dy = radius * np.sin(np.radians(angle))

            # Se da prioridad al color del arrow_params si se especifica
            if "color" in arrow_params.keys():
                color = arrow_params["color"]
            else:
                color = series["color"]

            arrow = patches.FancyArrowPatch(
                (x_point, y_point),
                (x_point + dx, y_point + dy),
                color=color,
                **arrow_params,
            )
            self.ax1.add_patch(arrow)

    def add_triangulation(
        self,
        data: list,
        colorbar: dict,
        alpha: float = 0.8,
    ) -> None:
        """
        Add a triangulated colormap to the plot based on the data series.

        Parameters
        ----------
        data : list
            List of data series to triangulate.
        colorbar : dict
            A dictionary with two keys:
            - "thresholds": A list of thresholds (e.g., [5, 20, 50]).
            - "colors": A list of corresponding colors (e.g., ["green", "yellow", "orange", "red"]).
        alpha : float, optional
            Transparency of the triangulated regions, by default 0.8.
        """
        import matplotlib.tri as tri

        if self.fig is None or self.ax1 is None:
            raise RuntimeError(
                "Primary plot must be created before adding triangulation."
            )

        thresholds = colorbar["thresholds"]
        colors = colorbar["colors"]

        # Ensure the number of colors matches the number of thresholds + 1
        if len(colors) != len(thresholds) + 1:
            raise ValueError(
                "The number of colors must be one more than the number of thresholds."
            )

        # Extract points and values for triangulation
        points = [(series["x"][0], series["y"][0]) for series in data]
        values = [series["value"] for series in data]

        # Perform triangulation
        triang = tri.Triangulation([p[0] for p in points], [p[1] for p in points])

        # Map values to colors
        for triangle in triang.triangles:
            triangle_values = [float(values[triangle[j]]) for j in range(3)]
            avg_value = sum(triangle_values) / 3

            # Determine the color based on thresholds
            color = colors[-1]  # Default to the last color
            for t, c in zip(thresholds, colors):
                if avg_value <= t:
                    color = c
                    break

            # Plot the triangle
            polygon = patches.Polygon(
                [points[triangle[j]] for j in range(3)],
                closed=True,
                color=color,
                alpha=alpha,
            )
            self.ax1.add_patch(polygon)

    def add_notes(self, notes: list) -> None:
        """
        Add text notes to the plot at specified points using annotate and adjustText.

        Parameters
        ----------
        notes : list of dict
            Each dict must have:
                - 'text': str, the note to display
                - 'x': float, x coordinate
                - 'y': float, y coordinate
        """
        if self.fig is None or self.ax1 is None:
            raise RuntimeError("Primary plot must be created before adding notes.")

        note_params = self.plot_style.get("note_params", {})
        adjust_text_params = self.plot_style.get("adjust_text_params", {})

        text_style = note_params.get("text_style", {})
        effects_config = note_params.get("path_effects", [])
        path_effects_list = (
            parse_path_effects(effects_config) if effects_config else None
        )

        text_objects = []

        for note in notes:
            text = note["text"]
            x = note["x"]
            y = note["y"]

            text_obj = self.ax1.text(
                x,
                y,
                text,
                **text_style,
            )

            if path_effects_list:
                text_obj.set_path_effects(path_effects_list)

            text_objects.append(text_obj)

        adjust_text(
            text_objects,
            ax=self.ax1,
            **adjust_text_params,
        )
