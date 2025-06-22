import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    Select,
    CustomJS,
    HoverTool,
    CheckboxGroup,
    DatetimeTickFormatter,
    Div,
    BoxZoomTool,
    Span,
    Label,
)
from bokeh.layouts import column, row
from bokeh.palettes import Spectral11

from libs.utils.config_loader import load_toml
from libs.utils.config_variables import CALC_CONFIG_DIR

TICK_COUNT_X = 15
TICK_COUNT_Y = 15


class BaseScatterPlot(ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        y_columns: list[str] = None,
        instrument_config_path: Path = None,
        width: int = 1200,
        height: int = 600,
        default_series: dict = None,
        instrument: str = None,
    ):
        self.df = df
        self.width = width
        self.height = height
        self.default_series = default_series or {}
        self.instrument = instrument
        self.config = load_toml(CALC_CONFIG_DIR, instrument)
        self.spanish_names = self.config["names"]["es"]
        self.y_columns = y_columns or self._get_plottable_columns()
        self._validate_columns()

        self.labels_map = self._create_labels_map()
        self.source = ColumnDataSource(data=self.df)

        # Components to be initialized
        self.figure = None
        self.scatter = None
        self.line = None
        self.hover = None

        self._initialize_components()

    def _get_plottable_columns(self) -> list[str]:
        return [
            col for col in self.df.select_dtypes(include=["float64", "int64"]).columns
        ]

    def _validate_columns(self):
        if not self.y_columns:
            raise ValueError("No numeric columns found in DataFrame")
        if "time" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'time' column")

    def _create_labels_map(self) -> dict:
        all_columns = ["time"] + self.y_columns
        return {col: self.spanish_names.get(col, col) for col in all_columns}

    def _initialize_components(self):
        self._create_figure()
        self._create_controls()
        self._setup_callbacks()

    @abstractmethod
    def _create_hover_tool(self, default_y: str) -> HoverTool:
        pass

    @abstractmethod
    def _create_figure_instance(self, default_y: str, hover: HoverTool) -> figure:
        pass

    def _create_figure(self):
        default_y = self._get_default_y_column()
        self.hover = self._create_hover_tool(default_y)
        self.figure = self._create_figure_instance(default_y, self.hover)

        # Configure common axis properties
        self.figure.xaxis[0].ticker.desired_num_ticks = TICK_COUNT_X
        self.figure.yaxis[0].ticker.desired_num_ticks = TICK_COUNT_Y
        self.figure.xaxis.major_label_orientation = 1.5708

        # Create glyphs
        self.scatter = self.figure.scatter(
            "time", default_y, source=self.source, size=10, color="navy", alpha=0.5
        )
        self.line = self.figure.line(
            "time",
            default_y,
            source=self.source,
            color="navy",
            alpha=0.3,
            visible=False,
        )

    def _get_default_y_column(self) -> str:
        return (
            "diff_disp_total_abs"
            if "diff_disp_total_abs" in self.y_columns
            else self.y_columns[0]
        )

    @abstractmethod
    def _create_controls(self):
        """Create the controls specific to each plot type"""
        pass

    @abstractmethod
    def _setup_callbacks(self):
        """Setup callbacks specific to each plot type"""
        pass

    @abstractmethod
    def controls(self):
        """Return the controls layout specific to each plot type"""
        pass

    @property
    def layout(self):
        return column(self.controls, self.figure)


class InteractiveScatterPlot(BaseScatterPlot):
    def __init__(self, df, default_series=None, **kwargs):
        self.x_select = None
        self.y_select = None
        self.show_line = None
        default_series = default_series or {"x": "east", "y": "north"}
        super().__init__(df, default_series=default_series, **kwargs)

    def _create_hover_tool(self, default_y: str) -> HoverTool:
        return HoverTool(
            tooltips=[("X", "@time{0,0.00}"), ("Y", f"@{{{default_y}}}{{0,0.00}}")],
            mode="mouse",
        )

    def _create_figure_instance(self, default_y: str, hover: HoverTool) -> figure:
        # Create box zoom tool that only works on height (Y axis)
        y_zoom = BoxZoomTool(dimensions='height')
        
        fig = figure(
            tools=["pan", "wheel_zoom", hover, y_zoom, "box_zoom", "reset", "save"],
            width=self.width,
            height=self.height,
        )
        fig.xaxis.formatter.use_scientific = False
        fig.yaxis.formatter.use_scientific = False
        return fig

    def _create_controls(self):
        # Create X selector
        x_options = [("time", self.labels_map["time"])]
        x_options += [(col, self.labels_map[col]) for col in self.y_columns]
        self.x_select = Select(
            title="Variable X:",
            value=self.default_series.get("x", "time"),
            options=x_options,
        )

        # Create Y selector
        self.y_select = Select(
            title="Variable Y:",
            value=self.default_series.get("y", self._get_default_y_column()),
            options=[(col, self.labels_map[col]) for col in self.y_columns],
        )

        # Create line toggle
        self.show_line = CheckboxGroup(labels=["Mostrar líneas"], active=[])

    def _setup_callbacks(self):
        js_labels = {k: str(v) for k, v in self.labels_map.items()}

        callback = CustomJS(
            args=dict(
                source=self.source,
                scatter=self.scatter,
                line=self.line,
                x_select=self.x_select,
                y_select=self.y_select,
                show_line=self.show_line,
                x_axis=self.figure.xaxis[0],
                y_axis=self.figure.yaxis[0],
                labels=js_labels,
                hover=self.hover,
                figure=self.figure,
            ),
            code="""
            const x_col = x_select.value;
            const y_col = y_select.value;
            
            // Actualizar glyphs
            scatter.glyph.x = {field: x_col};
            scatter.glyph.y = {field: y_col};
            line.glyph.x = {field: x_col};
            line.glyph.y = {field: y_col};
            
            // Actualizar ejes
            x_axis.axis_label = labels[x_col];
            y_axis.axis_label = labels[y_col];
            
            // Actualizar tooltips
            hover.tooltips = [
                ['X', `@{${x_col}}{0,0.00}`],
                ['Y', `@{${y_col}}{0,0.00}`]
            ];
            
            // Controlar visibilidad de líneas
            line.visible = show_line.active.includes(0);
            
            // Forzar actualización
            source.change.emit();
            figure.change.emit();
            """,
        )

        self.x_select.js_on_change("value", callback)
        self.y_select.js_on_change("value", callback)
        self.show_line.js_on_change("active", callback)

    @property
    def controls(self):
        return row(self.x_select, self.y_select, self.show_line)


class TimeSeriesBasePlot(BaseScatterPlot):
    """Base class for time series plots"""

    def __init__(self, *args, vertical_lines: list[dict] = None, horizontal_lines: list[dict] = None, **kwargs):
        self.x_column = "time"
        self.vertical_lines = vertical_lines or []
        self.horizontal_lines = horizontal_lines or []
        self.vertical_spans = []  # Store vertical spans and labels
        self.horizontal_spans = []  # Store horizontal spans and labels
        self.show_vertical = None  # Checkbox control
        self.show_horizontal = None  # Checkbox control
        super().__init__(*args, **kwargs)

    def _create_figure_instance(self, default_y: str, hover: HoverTool) -> figure:
        # Create box zoom tool that only works on height (Y axis)
        y_zoom = BoxZoomTool(dimensions='height')
        
        fig = figure(
            tools=["pan", "wheel_zoom", hover, y_zoom, "box_zoom", "reset", "save"],
            x_axis_type="datetime",
            width=self.width,
            height=self.height,
        )
        fig.xaxis.formatter = DatetimeTickFormatter(
            milliseconds="%d-%m-%y %H:%M:%S.%3N",
            seconds="%d-%m-%y %H:%M:%S",
            minsec="%d-%m-%y %H:%M:%S",
            minutes="%d-%m-%y %H:%M",
            hourmin="%d-%m-%y %H:%M",
            hours="%d-%m-%y %H:%M",
            days="%d-%m-%y",
            months="%d-%m-%y",
            years="%d-%m-%y",
        )
        fig.yaxis.formatter.use_scientific = False
        return fig

    def _create_hover_tool(self, default_y: str) -> HoverTool:
        return HoverTool(
            tooltips=[
                ("Fecha", "@time{%d-%m-%y %H:%M:%S}"),
                ("Y", f"@{{{default_y}}}{{0,0.00}}"),
            ],
            formatters={"@time": "datetime"},
            mode="mouse",
        )

    def _create_figure(self):
        super()._create_figure()
        self._add_vertical_lines()
        self._add_horizontal_lines()

    def _add_vertical_lines(self):
        self.vertical_spans = []  # Reset spans list
        for line in self.vertical_lines:
            date = pd.to_datetime(line.get("date"))
            text = line.get("text", "")
            color = line.get("color", "black")
            line_style = line.get("style", "dashed")  # Options: solid, dashed, dotted, dotdash, dashdot
            span = Span(
                location=date.timestamp() * 1000,
                dimension='height',
                line_color=color,
                line_dash=line_style,
                line_width=2
            )
            self.figure.add_layout(span)
            self.vertical_spans.append(span)
            if text:
                label = Label(
                    x=date.timestamp() * 1000,
                    y=0,
                    text=text,
                    text_color=color,
                    text_font_size='10pt',
                    text_align='center',
                    angle=1.5708
                )
                self.figure.add_layout(label)
                self.vertical_spans.append(label)

    def _add_horizontal_lines(self):
        self.horizontal_spans = []  # Reset spans list
        for line in self.horizontal_lines:
            value = line.get("value")
            text = line.get("text", "")
            color = line.get("color", "black")
            line_style = line.get("style", "dashed")  # Options: solid, dashed, dotted, dotdash, dashdot
            span = Span(
                location=value,
                dimension='width',
                line_color=color,
                line_dash=line_style,
                line_width=2
            )
            self.figure.add_layout(span)
            self.horizontal_spans.append(span)
            if text:
                label = Label(
                    x=self.figure.x_range.start,
                    y=value,
                    text=text,
                    text_color=color,
                    text_font_size='10pt',
                    text_align='left'
                )
                self.figure.add_layout(label)
                self.horizontal_spans.append(label)

    def _create_controls(self):
        # Add visibility controls for lines if they exist
        if self.vertical_lines:
            self.show_vertical = CheckboxGroup(
                labels=["Mostrar líneas verticales"],
                active=[0],  # Visible by default
            )
            self.show_vertical.js_on_change('active', CustomJS(
                args=dict(spans=self.vertical_spans),
                code="""
                const visible = cb_obj.active.includes(0);
                spans.forEach(span => span.visible = visible);
                """
            ))

        if self.horizontal_lines:
            self.show_horizontal = CheckboxGroup(
                labels=["Mostrar líneas horizontales"],
                active=[0],  # Visible by default
            )
            self.show_horizontal.js_on_change('active', CustomJS(
                args=dict(spans=self.horizontal_spans),
                code="""
                const visible = cb_obj.active.includes(0);
                spans.forEach(span => span.visible = visible);
                """
            ))

    @property
    def line_controls(self):
        controls = []
        if self.show_vertical:
            controls.append(self.show_vertical)
        if self.show_horizontal:
            controls.append(self.show_horizontal)
        return row(*controls) if controls else None


class TimeSeriesScatterPlot(TimeSeriesBasePlot):
    def __init__(self, df, default_series=None, **kwargs):
        self.y_select = None
        self.show_line = None
        default_series = default_series or {"y": "diff_disp_total_abs"}
        super().__init__(df, default_series=default_series, **kwargs)

    def _create_controls(self):
        # Call parent's _create_controls to setup line visibility controls
        super()._create_controls()
        
        # Create Y selector only
        self.y_select = Select(
            title="Variable Y:",
            value=self.default_series.get("y", self._get_default_y_column()),
            options=[(col, self.labels_map[col]) for col in self.y_columns],
        )
        self.show_line = CheckboxGroup(labels=["Mostrar líneas"], active=[])

    def _setup_callbacks(self):
        js_labels = {k: str(v) for k, v in self.labels_map.items()}

        callback = CustomJS(
            args=dict(
                source=self.source,
                scatter=self.scatter,
                line=self.line,
                y_select=self.y_select,
                show_line=self.show_line,
                y_axis=self.figure.yaxis[0],
                labels=js_labels,
                hover=self.hover,
            ),
            code="""
            const y_col = y_select.value;
            
            scatter.glyph.y = {field: y_col};
            line.glyph.y = {field: y_col};
            y_axis.axis_label = labels[y_col];
            
            hover.tooltips = [
                ['Fecha', '@time{%F %H:%M:%S}'],
                ['Y', `@{${y_col}}{0,0.00}`]
            ];
            
            hover.formatters = {
                '@time': 'datetime',
                [`@${y_col}`]: 'printf'
            };
            
            line.visible = show_line.active.includes(0);
            source.change.emit();
            """,
        )

        self.y_select.js_on_change("value", callback)
        self.show_line.js_on_change("active", callback)

    @property
    def controls(self):
        main_controls = row(self.y_select, self.show_line)
        if self.line_controls:
            return column(main_controls, self.line_controls)
        return main_controls


class MultiTimeSeriesScatterPlot(TimeSeriesBasePlot):
    def __init__(self, df, default_series=None, **kwargs):
        self.y_select = None
        self.show_line = None
        self.glyphs = {}
        self.selected_div = None
        self.initial_series = default_series or ["east", "north", "elevation"]
        super().__init__(df, **kwargs)

    def _create_hover_tool(self, default_y: str) -> HoverTool:
        return HoverTool(
            tooltips=[
                ("Fecha", "@time{%d-%m-%y %H:%M:%S}"),
                ("Serie", "$name"),
                ("Valor", "$y{0.00}"),
            ],
            formatters={"@time": "datetime"},
            mode="mouse",
        )

    def _create_figure_instance(self, default_y: str, hover: HoverTool) -> figure:
        fig = figure(
            tools=["pan", "wheel_zoom", "box_zoom", "reset", "save", hover],
            x_axis_type="datetime",
            width=self.width,
            height=self.height,
        )
        fig.xaxis.formatter = DatetimeTickFormatter(
            milliseconds="%d-%m-%y %H:%M:%S.%3N",
            seconds="%d-%m-%y %H:%M:%S",
            minsec="%d-%m-%y %H:%M:%S",
            minutes="%d-%m-%y %H:%M",
            hourmin="%d-%m-%y %H:%M",
            hours="%d-%m-%y %H:%M",
            days="%d-%m-%y",
            months="%d-%m-%y",
            years="%d-%m-%y",
        )

        # Disable scientific notation for Y axis
        fig.yaxis.formatter.use_scientific = False

        return fig

    def _create_figure(self):
        default_y = self._get_default_y_column()
        self.hover = self._create_hover_tool(default_y)
        self.figure = self._create_figure_instance(default_y, self.hover)

        # Configure axes
        self.figure.xaxis[0].ticker.desired_num_ticks = TICK_COUNT_X
        self.figure.yaxis[0].ticker.desired_num_ticks = TICK_COUNT_Y
        self.figure.xaxis.major_label_orientation = 1.5708

        # Create glyphs for each Y column
        colors = Spectral11 * (len(self.y_columns) // len(Spectral11) + 1)
        for i, y_col in enumerate(self.y_columns):
            color = colors[i % len(Spectral11)]
            label = self.labels_map[y_col]

            scatter = self.figure.scatter(
                x="time",
                y=y_col,
                source=self.source,
                color=color,
                size=8,
                alpha=0.6,
                visible=y_col in self.initial_series,
                name=label,
            )

            line = self.figure.line(
                x="time",
                y=y_col,
                source=self.source,
                color=color,
                alpha=0.4,
                visible=False,
                name=label,
            )

            self.glyphs[y_col] = (scatter, line)

        self._add_vertical_lines()
        self._add_horizontal_lines()

    def _create_controls(self):
        # Call parent's _create_controls to setup line visibility controls
        super()._create_controls()
        
        # Simple dropdown for adding series with descriptive names
        self.y_select = Select(
            title="Agregar serie:",
            value="",
            options=[("", "-- Seleccionar serie --")]
            + [(col, self.labels_map[col]) for col in self.y_columns],
            width=300,
        )

        # Create line toggle
        self.show_line = CheckboxGroup(labels=["Mostrar líneas"], active=[])

        # Div for selected series with remove buttons
        self.selected_div = Div(
            text='<div class="selected-series"></div>',
            styles={"margin-top": "10px", "min-height": "40px"},
        )

    def _setup_callbacks(self):
        callback = CustomJS(
            args=dict(
                glyphs=self.glyphs,
                y_select=self.y_select,
                show_line=self.show_line,
                selected_div=self.selected_div,
                colors=Spectral11,
                y_columns=self.y_columns,
            ),
            code="""
            const selected_y = y_select.value;
            if (!selected_y) return;
            
            // Get current selected series from div
            const container = document.createElement('div');
            container.innerHTML = selected_div.text;
            const currentSeries = Array.from(container.querySelectorAll('.series-tag'))
                .map(tag => tag.dataset.series);
            
            // Add new series if not already selected
            if (!currentSeries.includes(selected_y)) {
                const colorIndex = y_columns.indexOf(selected_y) % colors.length;
                const color = colors[colorIndex];
                
                const selectedOption = y_select.options.find(opt => opt[0] === selected_y);
                const label = selectedOption ? selectedOption[1] : selected_y;
                
                // Show series in plot
                glyphs[selected_y][0].visible = true;
                glyphs[selected_y][1].visible = show_line.active.includes(0);
                
                // Add series tag with color picker and remove button
                const seriesHtml = `
                    <div class="series-tag" data-series="${selected_y}" style="
                        display: inline-flex;
                        align-items: center;
                        background: #f5f5f5;
                        padding: 4px 8px;
                        margin: 2px;
                        border-radius: 3px;
                        border: 1px solid ${color};
                        font-size: 12px;
                    ">
                        <input type="color" value="${color}" 
                            style="width: 20px; height: 20px; padding: 0; margin-right: 5px; cursor: pointer;"
                            onchange="changeSeriesColor('${selected_y}', this.value)"
                        />
                        ${label}
                        <button onclick="removeSeries('${selected_y}')" style="
                            border: none;
                            background: none;
                            color: #999;
                            margin-left: 5px;
                            cursor: pointer;
                            padding: 0 3px;
                        ">×</button>
                    </div>
                `;
                container.querySelector('.selected-series').innerHTML += seriesHtml;
                selected_div.text = container.innerHTML;
            }
            
            // Reset select
            y_select.value = "";
            
            // Add helper functions if not exists
            if (!window.removeSeries) {
                window.removeSeries = function(series) {
                    glyphs[series][0].visible = false;
                    glyphs[series][1].visible = false;
                    
                    const container = document.createElement('div');
                    container.innerHTML = selected_div.text;
                    const tag = container.querySelector(`[data-series="${series}"]`);
                    if (tag) tag.remove();
                    selected_div.text = container.innerHTML;
                }
            }
            
            if (!window.changeSeriesColor) {
                window.changeSeriesColor = function(series, color) {
                    if (glyphs[series]) {
                        glyphs[series][0].glyph.line_color = color;
                        glyphs[series][0].glyph.fill_color = color;
                        glyphs[series][1].glyph.line_color = color;
                        
                        const tag = document.querySelector(`[data-series="${series}"]`);
                        if (tag) tag.style.borderColor = color;
                    }
                }
            }
            """,
        )

        line_callback = CustomJS(
            args=dict(glyphs=self.glyphs, show_line=self.show_line),
            code="""
            const showLines = show_line.active.includes(0);
            Object.values(glyphs).forEach(([scatter, line]) => {
                if (scatter.visible) {
                    line.visible = showLines;
                }
            });
            """,
        )

        self.y_select.js_on_change("value", callback)
        self.show_line.js_on_change("active", line_callback)

    @property
    def controls(self):
        controls = [
            self.y_select,
            self.show_line,
            self.selected_div,
        ]
        if self.line_controls:
            controls.append(self.line_controls)
        return column(*controls, sizing_mode="stretch_width")


class MultiDataFrameTimeSeriesPlot(TimeSeriesBasePlot):
    def __init__(self, dfs: dict[str, pd.DataFrame], series: str, default_series: list[str] = None, **kwargs):
        self.df_dict = dfs
        self.series = series
        self.sources = {}
        self.initial_series = default_series or []
        self.add_df_select = None
        # Initialize selected_div before _create_figure
        self.selected_div = Div(
            text='<div class="selected-series"></div>',
            styles={"margin-top": "10px", "min-height": "40px"},
        )
        self.y_select = None
        first_df = next(iter(dfs.values()))
        kwargs['y_columns'] = kwargs.get('y_columns', [col for col in first_df.columns if col != 'time'])
        super().__init__(first_df, **kwargs)
        
    def _create_figure(self):
        self.hover = self._create_hover_tool(self.series)
        self.figure = self._create_figure_instance(self.series, self.hover)

        # Configure axes
        self.figure.xaxis[0].ticker.desired_num_ticks = TICK_COUNT_X
        self.figure.yaxis[0].ticker.desired_num_ticks = TICK_COUNT_Y
        self.figure.xaxis.major_label_orientation = 1.5708

        # Create glyphs for each DataFrame
        colors = Spectral11 * (len(self.df_dict) // len(Spectral11) + 1)
        for i, (name, df) in enumerate(self.df_dict.items()):
            color = colors[i % len(Spectral11)]
            source = ColumnDataSource(data=df)
            self.sources[name] = {
                'source': source,
                'color': color
            }

            scatter = self.figure.scatter(
                x="time",
                y=self.series,
                source=source,
                color=color,
                size=8,
                alpha=0.6,
                name=name,
                visible=name in self.initial_series  # Show if in default series
            )

            line = self.figure.line(
                x="time",
                y=self.series,
                source=source,
                color=color,
                alpha=0.4,
                visible=False,
                name=name
            )

            self.sources[name]['renderers'] = (scatter, line)
            
            # Add initial series to div if visible
            if name in self.initial_series:
                self._add_series_to_div(name, color)

        self._add_vertical_lines()
        self._add_horizontal_lines()

    def _add_series_to_div(self, series_name: str, color: str):
        """Helper method to add a series tag to the div"""
        html = self.selected_div.text.replace('</div>', '')  # Remove closing div
        series_html = f"""
            <div class="series-tag" data-series="{series_name}" style="
                display: inline-flex;
                align-items: center;
                background: #f5f5f5;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                border: 1px solid {color};
                font-size: 12px;
            ">
                <input type="color" value="{color}" 
                    style="width: 20px; height: 20px; padding: 0; margin-right: 5px; cursor: pointer;"
                    onchange="changeSeriesColor('{series_name}', this.value)"
                />
                {series_name}
                <button onclick="removeSeries('{series_name}')" style="
                    border: none;
                    background: none;
                    color: #999;
                    margin-left: 5px;
                    cursor: pointer;
                    padding: 0 3px;
                ">×</button>
            </div>
        """
        self.selected_div.text = html + series_html + '</div>'

    def _create_controls(self):
        # Call parent's _create_controls to setup line visibility controls
        super()._create_controls()
        
        # Y series selector - filter only columns that have a name in TOML
        named_columns = [(col, name) for col, name in self.labels_map.items() 
                        if col != "time" and col in next(iter(self.df_dict.values())).columns and name != col]
        
        self.y_select = Select(
            title="Variable Y:",
            value=self.series,
            options=sorted(named_columns, key=lambda x: x[1]),
        )
        
        # DataFrame selector - use dictionary keys as names
        self.add_df_select = Select(
            title="Agregar serie temporal:",
            value="",
            options=[("", "-- Seleccionar serie --")] + 
                    [(name, name) for name in self.df_dict.keys()],
            width=300,
        )

        self.show_line = CheckboxGroup(labels=["Mostrar líneas"], active=[])

    def _setup_callbacks(self):
        df_callback = CustomJS(
            args=dict(
                sources=self.sources,
                add_df_select=self.add_df_select,
                show_line=self.show_line,
                selected_div=self.selected_div,
                colors=Spectral11,
            ),
            code="""
            const selected_df = add_df_select.value;
            if (!selected_df) return;
            
            // Get current selected series from div
            const container = document.createElement('div');
            container.innerHTML = selected_div.text;
            const currentSeries = Array.from(container.querySelectorAll('.series-tag'))
                .map(tag => tag.dataset.series);
            
            // Add new series if not already selected
            if (!currentSeries.includes(selected_df)) {
                const source_info = sources[selected_df];
                const color = source_info.color;
                
                // Show renderers
                source_info.renderers[0].visible = true;  // scatter
                source_info.renderers[1].visible = show_line.active.includes(0);  // line
                
                // Add series tag with color picker and remove button
                const seriesHtml = `
                    <div class="series-tag" data-series="${selected_df}" style="
                        display: inline-flex;
                        align-items: center;
                        background: #f5f5f5;
                        padding: 4px 8px;
                        margin: 2px;
                        border-radius: 3px;
                        border: 1px solid ${color};
                        font-size: 12px;
                    ">
                        <input type="color" value="${color}" 
                            style="width: 20px; height: 20px; padding: 0; margin-right: 5px; cursor: pointer;"
                            onchange="changeSeriesColor('${selected_df}', this.value)"
                        />
                        ${selected_df}
                        <button onclick="removeSeries('${selected_df}')" style="
                            border: none;
                            background: none;
                            color: #999;
                            margin-left: 5px;
                            cursor: pointer;
                            padding: 0 3px;
                        ">×</button>
                    </div>
                `;
                container.querySelector('.selected-series').innerHTML += seriesHtml;
                selected_div.text = container.innerHTML;
            }
            
            // Reset select
            add_df_select.value = "";
            
            // Add helper functions if not exists
            if (!window.removeSeries) {
                window.removeSeries = function(series_name) {
                    const source_info = sources[series_name];
                    source_info.renderers[0].visible = false;
                    source_info.renderers[1].visible = false;
                    
                    const container = document.createElement('div');
                    container.innerHTML = selected_div.text;
                    const tag = container.querySelector(`[data-series="${series_name}"]`);
                    if (tag) tag.remove();
                    selected_div.text = container.innerHTML;
                }
            }
            
            if (!window.changeSeriesColor) {
                window.changeSeriesColor = function(series_name, color) {
                    const source_info = sources[series_name];
                    if (source_info) {
                        source_info.renderers[0].glyph.line_color = color;
                        source_info.renderers[0].glyph.fill_color = color;
                        source_info.renderers[1].glyph.line_color = color;
                        
                        const tag = document.querySelector(`[data-series="${series_name}"]`);
                        if (tag) tag.style.borderColor = color;
                    }
                }
            }
            """,
        )

        y_select_callback = CustomJS(
            args=dict(
                sources=self.sources,
                y_select=self.y_select,
                y_axis=self.figure.yaxis[0],
                labels=self.labels_map,
                hover=self.hover,
            ),
            code="""
            const y_col = y_select.value;
            
            // Update y field for all renderers
            Object.values(sources).forEach(source_info => {
                source_info.renderers[0].glyph.y = {field: y_col};  // scatter
                source_info.renderers[1].glyph.y = {field: y_col};  // line
            });

            // Update y axis label
            y_axis.axis_label = labels[y_col] || y_col;
            
            // Update hover tooltips
            hover.tooltips = [
                ['Fecha', '@time{%d-%m-%y %H:%M:%S}'],
                ['Serie', '$name'],
                ['Valor', `@{${y_col}}{0,0.00}`]
            ];
            """,
        )

        line_callback = CustomJS(
            args=dict(sources=self.sources, show_line=self.show_line),
            code="""
            const showLines = show_line.active.includes(0);
            Object.values(sources).forEach(source_info => {
                if (source_info.renderers[0].visible) {  // if scatter is visible
                    source_info.renderers[1].visible = showLines;  // update line visibility
                }
            });
            """
        )

        self.add_df_select.js_on_change("value", df_callback)
        self.y_select.js_on_change("value", y_select_callback)
        self.show_line.js_on_change("active", line_callback)

    @property
    def controls(self):
        controls = [
            row(self.y_select, self.add_df_select),
            self.show_line,
            self.selected_div,
        ]
        if self.line_controls:
            controls.append(self.line_controls)
        return column(*controls, sizing_mode="stretch_width")


def save_chart(layout: column, output_path: str) -> None:
    # Get the filename without extension as the title
    title = Path(output_path).stem
    output_file(output_path, title=title)
    save(layout)


if __name__ == "__main__":
    # Ejemplo de uso
    csv_path = (
        Path(__file__).parent.parent.parent
        / "var\Shahuindo_SAC\Shahuindo\processed_data\PCT\PAD_2B_2C.2B-3.csv"
    )

    csv_path_2 = (
        Path(__file__).parent.parent.parent
        / "var\Shahuindo_SAC\Shahuindo\processed_data\PCT\PAD_2B_2C.OVER-6.csv"
    )

    # Leer datos con soporte para milisegundos usando el método recomendado
    def read_dataframe(path: Path) -> pd.DataFrame:
        # Leer CSV con la columna time como objeto
        df = pd.read_csv(path)
        # Convertir la columna time a datetime preservando milisegundos
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        return df

    df = read_dataframe(csv_path)
    df_2 = read_dataframe(csv_path_2)

    vertical_lines = [
        {"date": "2024-01-14 12:00:00", "text": "   Evento A", "color": "red", "style": "dashed"},
        {"date": "2024-09-14 12:00:00", "text": "   Evento B", "color": "blue", "style": "dotted"},
        {"date": "2024-11-14 12:00:00", "text": "   ", "color": "green", "style": "solid"},
    ]

    horizontal_lines = [
        {"value": 50, "text": "Límite superior", "color": "red", "style": "dashed"},
        {"value": -50, "text": "Límite inferior", "color": "red", "style": "dashed"},
        {"value": 0, "text": "Base", "color": "black", "style": "dotted"},
    ]

    # Interactive plot con series por defecto
    plot = InteractiveScatterPlot(
        df,
        default_series={"x": "east", "y": "north"},
        instrument="prism",
    )
    save_chart(plot.layout, "output.html")

    # Time series plot con serie por defecto
    plot = TimeSeriesScatterPlot(
        df,
        default_series={"y": "diff_disp_total_abs"},
        instrument="prism",
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
    )
    save_chart(plot.layout, "timeseries.html")

    # Multi series plot con series por defecto
    plot = MultiTimeSeriesScatterPlot(
        df,
        default_series=["east", "north", "elevation"],
        instrument="prism",
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
    )
    save_chart(plot.layout, "multi_timeseries.html")

    # Multi DataFrame Time Series plot - using dictionary input
    plot = MultiDataFrameTimeSeriesPlot(
        dfs={
            "2B-3": df,
            "OVER-6": df_2
        },
        series="diff_disp_total_abs",
        default_series=["P2B-3"],  # Show first series by default
        instrument="prism",
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
    )
    save_chart(plot.layout, "multi_dataframe_timeseries.html")
