from typing import Tuple, List
from reportlab.graphics.shapes import Drawing, Group, Rect


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
