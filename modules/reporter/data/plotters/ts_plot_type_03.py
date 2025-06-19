import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd

from modules.reporter.plot_builder import PlotMerger, PlotBuilder
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.plot_helpers import get_unique_marker_convo
from libs.utils.calc_helpers import get_typical_range
from libs.utils.config_loader import load_toml
from libs.utils.calc_helpers import round_decimal, format_date_long, format_date_short
from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
)

COLOR_PALETTE = "cool"


def calculate_note_variables(dfs, sensor_names, serie_x, target_column, mask=None):
    """Calculate variables for notes based on multiple dataframes."""
    all_vars = []
    for df, name in zip(dfs, sensor_names):
        # Apply mask if provided
        if mask is not None:
            df = df[mask(df)]

        first_date = pd.to_datetime(df[serie_x].iloc[0])
        last_date = pd.to_datetime(df[serie_x].iloc[-1])
        idx_max = df[target_column].idxmax()
        max_value = df.loc[idx_max, target_column]
        max_date = pd.to_datetime(df.loc[idx_max, serie_x])

        total_records = len(df)
        mean_freq = (
            (last_date - first_date).days / total_records if total_records > 0 else 0
        )

        all_vars.append(
            {
                "sensor_name": name,
                "total_records": total_records,
                "max_value": max_value,
                "max_date": max_date,
                "last_value": df[target_column].iloc[-1],
                "first_date": first_date,
                "last_date": last_date,
                "mean_freq": round_decimal(mean_freq, 2),
            }
        )
    return all_vars


def get_note_content(
    group_args,
    data_sensors,
    target_column,
    unit_target,
    series_names,
    serie_x,
    sensor_aka,
    mask=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler()

    target_column_name = series_names[target_column].lower().split("(")[0]

    # Calculate variables for all sensors
    calc_vars = calculate_note_variables(
        data_sensors["df"], data_sensors["names"], serie_x, target_column, mask
    )

    # Define sections with narrative style for each sensor
    sections = [
        {
            "title": "Ubicación:",
            "content": [group_args["location"]],
            "format_type": "paragraph",
        },
        # {
        #     "title": "Material:",
        #     "content": [group_args["material"]],
        #     "format_type": "paragraph",
        # },
        {
            "title": "Notas:",
            "content": [
                f"{sensor_aka} {v['sensor_name']}, en el periodo entre {format_date_long(v['first_date'])} y "
                f"{format_date_long(v['last_date'])}, presenta {v['total_records']} registros. "
                f"El valor máximo de {target_column_name} corresponde a {round_decimal(v['max_value'], 2)} {unit_target} registrado el día {format_date_short(v['max_date'])}. "
                f"Su último registro corresponde a {round_decimal(v['last_value'], 2)} {unit_target} en el día {format_date_short(v['last_date'])}. "
                f"La frecuencia de monitoreo promedio es de un registro cada {v['mean_freq']} días."
                for v in calc_vars
            ],
            "format_type": "numbered",
        },
    ]

    return note_handler.create_notes(sections)


def create_map(dxf_path, data_sensors):
    plotter = PlotBuilder(ts_serie=True, ymargin=0)
    map_args = {
        "dxf_path": dxf_path,
        "size": [2.0, 1.5],
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": True,
        "dxf_params": {"color": "black", "linestyle": "-", "linewidth": 0.02},
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    series_data = []

    # Generate unique color combinations for each sensor
    for i, name in enumerate(data_sensors["names"]):
        color, _ = get_unique_marker_convo(
            i, len(data_sensors["names"]), color_palette=COLOR_PALETTE
        )
        series_data.append(
            {
                "x": data_sensors["east"][i],
                "y": data_sensors["north"][i],
                "color": color,
                "linetype": "",
                "lineweight": 0,
                "marker": "o",
                "markersize": 10,
                "label": "",
                "note": name,
                "fontsize": 6,
            }
        )

    plotter.plot_series(
        data=series_data,
        dxf_path=map_args["dxf_path"],
        size=map_args["size"],
        title_x=map_args["title_x"],
        title_y=map_args["title_y"],
        title_chart=map_args["title_chart"],
        show_legend=map_args["show_legend"],
        dxf_params=map_args["dxf_params"],
        format_params=map_args["format_params"],
    )

    return plotter.get_drawing()


def create_cell_1(
    data_sensors,
    series_names,
    primary_column,
    top_reference_column,
    bottom_reference_column,
    serie_x,
    primary_title_y,
    sensor_type_name,
):
    plotter = PlotBuilder()

    # Series style definitions
    series_styles = {
        top_reference_column: {
            "color": "peru",
            "linetype": "-",
            "lineweight": 1,
            "marker": "s",
            "markersize": 2,
            "label_prefix": series_names[top_reference_column],
        },
        bottom_reference_column: {
            "color": "dimgrey",
            "linetype": "-",
            "lineweight": 1,
            "marker": "s",
            "markersize": 2,
            "label_prefix": series_names[bottom_reference_column],
        },
    }

    # Add primary_column style if provided
    if primary_column:
        series_styles[primary_column] = {
            "color": "blue",
            "linetype": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[primary_column],
        }

    unique_serie = primary_column

    # Plot formatting
    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": primary_title_y,
        "title_chart": f"Registro histórico de {sensor_type_name}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    total_dfs = len(data_sensors["df"])
    for i, (df, name) in enumerate(zip(data_sensors["df"], data_sensors["names"])):
        for column, style in series_styles.items():
            if column in df.columns:
                # Special handling for unique_serie
                if primary_column and column == unique_serie:
                    color, marker = get_unique_marker_convo(
                        i, total_dfs, color_palette=COLOR_PALETTE
                    )
                else:
                    color = style["color"]
                    marker = style["marker"]

                label = (
                    name
                    if primary_column and column == unique_serie
                    else style["label_prefix"]
                )

                series.append(
                    {
                        "x": df[serie_x].tolist(),
                        "y": df[column].tolist(),
                        "label": label,
                        "color": color,
                        "linetype": style["linetype"],
                        "lineweight": style["lineweight"],
                        "marker": marker,
                        "markersize": style["markersize"],
                    }
                )

    plotter.plot_series(
        data=series,
        size=plot_format["size"],
        title_x=plot_format["title_x"],
        title_y=plot_format["title_y"],
        title_chart=plot_format["title_chart"],
        show_legend=plot_format["show_legend"],
    )
    return plotter.get_drawing(), plotter.get_legend(
        box_width=7.5,
        box_height=0.5,
        ncol=plotter.get_num_labels(),
    )


def create_cell_2(
    data_sensors, start_query, end_query, series_names, secondary_column, serie_x
):
    plotter = PlotBuilder()

    # Series style definitions
    series_styles = {
        secondary_column: {
            "color": "blue",
            "linetype": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[secondary_column],
        },
    }

    unique_serie = secondary_column

    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": series_names[secondary_column],
        "title_chart": "",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    all_secondary_values = []
    total_dfs = len(data_sensors["df"])

    for i, (df, name) in enumerate(zip(data_sensors["df"], data_sensors["names"])):
        for column, style in series_styles.items():
            if column in df.columns:
                if column == unique_serie:
                    color, marker = get_unique_marker_convo(
                        i, total_dfs, color_palette=COLOR_PALETTE
                    )
                else:
                    color = style["color"]
                    marker = style["marker"]

                label = name if column == unique_serie else style["label_prefix"]

                x_vals = df[serie_x].tolist()
                y_vals = df[column].tolist()

                series.append(
                    {
                        "x": x_vals,
                        "y": y_vals,
                        "label": label,
                        "color": color,
                        "linetype": style["linetype"],
                        "lineweight": style["lineweight"],
                        "marker": marker,
                        "markersize": style["markersize"],
                    }
                )

                all_secondary_values.extend(y_vals)

    # Calcular límites típicos solo si hay datos
    if all_secondary_values:
        lower_limit, upper_limit = get_typical_range(all_secondary_values, percentile=90, scale=1.5)
        ylim = (lower_limit, upper_limit)
    else:
        ylim = None  # Dejar que el gráfico elija automáticamente

    plotter.plot_series(
        data=series,
        size=plot_format["size"],
        title_x=plot_format["title_x"],
        title_y=plot_format["title_y"],
        title_chart=plot_format["title_chart"],
        show_legend=plot_format["show_legend"],
        invert_y=True,
        ylim=ylim,  # Usamos ylim ya que aplica al eje Y
    )

    return plotter.get_drawing(), plotter.get_legend(
        box_width=7.5,
        box_height=0.5,
        ncol=plotter.get_num_labels(),
    )


def generate_report(
    data_sensors,
    group_args,
    dxf_path,
    start_query,
    end_query,
    appendix,
    start_item,
    structure_code,
    structure_name,
    sensor_type,
    output_dir,
    static_report_params,
    column_config,
):
    """Generate chart report with notes and merged plots.

    Returns:
        str: Path to the generated PDF file (e.g. './outputs/A_200.pdf')
    """
    # Extract column configuration
    target_column = column_config["target_column"]
    unit_target = column_config["unit_target"]
    primary_column = column_config["primary_column"]
    primary_title_y = column_config["primary_title_y"]
    secondary_column = column_config["secondary_column"]
    top_reference_column = column_config["top_reference_column"]
    bottom_reference_column = column_config["bottom_reference_column"]
    serie_x = column_config["serie_x"]
    sensor_type_name = column_config["sensor_type_name"]
    sensor_aka = column_config["sensor_aka"]

    # Load configuration
    calc_config = load_toml(CALC_CONFIG_DIR, sensor_type)
    series_names = calc_config["names"]["es"]

    # Generate chart components
    chart_cell1, legend1 = create_cell_1(
        data_sensors,
        series_names,
        primary_column,
        top_reference_column,
        bottom_reference_column,
        serie_x,
        primary_title_y,
        sensor_type_name,
    )
    chart_cell2, legend2 = create_cell_2(
        data_sensors, start_query, end_query, series_names, secondary_column, serie_x
    )

    # Define mask for filtering data if start_query and end_query are provided
    mask = (
        (lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query))
        if start_query and end_query
        else None
    )

    # Create and configure plot grid
    plot_grid = PlotMerger(fig_size=(7.5, 5.5))
    plot_grid.create_grid(4, 1, row_ratios=[0.03, 0.47, 0.03, 0.47])

    # Add components to grid
    plot_grid.add_object(chart_cell1, (0, 0))
    plot_grid.add_object(legend1, (1, 0))
    plot_grid.add_object(chart_cell2, (2, 0))
    plot_grid.add_object(legend2, (3, 0))
    chart_cell = plot_grid.build(color_border="white", cell_spacing=0)

    # Create report components
    upper_cell = get_note_content(
        group_args,
        data_sensors,
        target_column,
        unit_target,
        series_names,
        serie_x,
        sensor_aka,
        mask,
    )
    lower_cell = create_map(dxf_path, data_sensors)
    logo_cell = load_svg(LOGO_SVG, 0.95)
    chart_title_elements = [
        f"Registro histórico de {sensor_type_name}",
        " - ".join(data_sensors["names"]),
        structure_name,
    ]
    chart_title = " / ".join(filter(None, chart_title_elements))

    # Set up PDF generator
    pdf_generator = ReportBuilder(
        sample="chart_landscape_a4_type_01",
        logo_cell=logo_cell,
        upper_cell=upper_cell,
        lower_cell=lower_cell,
        chart_cell=chart_cell,
        chart_title=chart_title,
        num_item=f"{appendix}.{start_item}",
        **static_report_params,
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Format the PDF filename
    structure_formatted = structure_code.replace(" ", "_")
    sensor_type_formatted = sensor_type.replace(" ", "_").upper()
    pdf_filenames = []

    # Generate base filename
    base_filename = f"{output_dir}/{appendix}_{start_item:03}_{structure_formatted}_{sensor_type_formatted}_{group_args['name']}.pdf"
    pdf_filenames.append(base_filename)

    # Generate PDF
    pdf_generator.generate_pdf(pdf_path=base_filename)

    return pdf_filenames
