import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from matplotlib import colormaps
from matplotlib.colors import rgb2hex
import pandas as pd
import numpy as np

from modules.reporter.plot_builder import PlotMerger, PlotBuilder
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.config_loader import load_toml
from libs.utils.calculations import round_decimal, format_date_long, format_date_short
from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
)

COLOR_PALETTE = "cool"


def get_unique_combination(df_index, used_combinations, total_dfs):
    """
    Generate a unique combination of color and marker for a given dataframe index.
    Ensures consistency across series.
    """
    from itertools import cycle

    # Define unique styles for markers
    unique_styles = {"markers": ["o", "s", "D", "v", "^", "<", ">", "p", "h"]}

    # Generate random colors from the colormap
    colormap = colormaps[COLOR_PALETTE]
    if total_dfs == 1:
        color = rgb2hex(
            colormap(0.5)
        )  # Use a fixed value if there's only one dataframe
    else:
        # Calculate equidistant position based on df_index
        pos = df_index / (total_dfs - 1)
        color = rgb2hex(colormap(pos))

    # Cycle through markers to ensure consistency
    marker_cycle = cycle(unique_styles["markers"])
    for _ in range(df_index + 1):
        marker = next(marker_cycle)

    combination = (color, marker)
    # while combination in used_combinations:
    #     random_pos = np.random.random()
    #     color = rgb2hex(colormap(random_pos))
    #     marker = next(marker_cycle)
    #     combination = (color, marker)

    # used_combinations.add(combination)
    return combination


def calculate_note_variables(dfs, sensor_names, serie_x, target_column, mask=None):
    """Calculate variables for notes based on multiple dataframes."""
    all_vars = []
    for df, name in zip(dfs, sensor_names):
        # Apply mask if provided
        if mask is not None:
            df = df[mask(df)]

        first_date = pd.to_datetime(df[serie_x].iloc[0])
        last_date = pd.to_datetime(df[serie_x].iloc[-1])
        idx_max_abs = df[target_column].abs().idxmax()
        max_value = df.loc[idx_max_abs, target_column]
        max_date = pd.to_datetime(df.loc[idx_max_abs, serie_x])

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


def create_note(
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
                f"La frecuencia de monitoreo promedio es de {v['mean_freq']} días."
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

    # Keep track of used combinations for consistent colors
    used_combinations = set()
    series_data = []

    # Generate unique color combinations for each sensor
    for i, name in enumerate(data_sensors["names"]):
        color, marker = get_unique_combination(
            i, used_combinations, len(data_sensors["names"])
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
    target_column,
    serie_x,
    sensor_type_name,
):
    plotter = PlotBuilder()

    # Keep track of used combinations
    used_combinations = set()

    # Series style definitions
    series_styles = {
        target_column: {
            "color": "blue",
            "linetype": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[target_column],
        },
    }

    # Plot formatting
    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": series_names[target_column],
        "title_chart": f"Registro histórico de {sensor_type_name}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    total_dfs = len(data_sensors["df"])
    for i, (df, name) in enumerate(zip(data_sensors["df"], data_sensors["names"])):
        if target_column in df.columns:
            color, marker = get_unique_combination(i, used_combinations, total_dfs)

            series.append(
                {
                    "x": df[serie_x].tolist(),
                    "y": df[target_column].tolist(),
                    "label": name,
                    "color": color,
                    "linetype": series_styles[target_column]["linetype"],
                    "lineweight": series_styles[target_column]["lineweight"],
                    "marker": marker,
                    "markersize": series_styles[target_column]["markersize"],
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
    data_sensors, start_query, end_query, series_names, target_column, serie_x
):
    plotter = PlotBuilder()

    # Keep track of used combinations
    used_combinations = set()

    # Series style definitions
    series_styles = {
        target_column: {
            "color": "blue",
            "linetype": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[target_column],
        },
    }

    # Format start and end dates
    start_date_formatted = format_date_long(pd.to_datetime(start_query))
    end_date_formatted = format_date_long(pd.to_datetime(end_query))

    # Plot formatting
    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": series_names[target_column],
        "title_chart": f"Registro de {series_names[target_column].lower().split('(')[0]} entre {start_date_formatted} y {end_date_formatted}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    total_dfs = len(data_sensors["df"])
    for i, (df, name) in enumerate(zip(data_sensors["df"], data_sensors["names"])):
        # Aplicar máscara para filtrar datos según la consulta
        mask = (df[serie_x] >= start_query) & (df[serie_x] <= end_query)
        filtered_df = df[mask]

        if target_column in filtered_df.columns:
            color, marker = get_unique_combination(i, used_combinations, total_dfs)

            series.append(
                {
                    "x": filtered_df[serie_x].tolist(),
                    "y": filtered_df[target_column].tolist(),
                    "label": name,
                    "color": color,
                    "linetype": series_styles[target_column]["linetype"],
                    "lineweight": series_styles[target_column]["lineweight"],
                    "marker": marker,
                    "markersize": series_styles[target_column]["markersize"],
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
        target_column,
        serie_x,
        sensor_type_name,
    )
    chart_cell2, legend2 = create_cell_2(
        data_sensors, start_query, end_query, series_names, target_column, serie_x
    )

    # Define mask for filtering data if start_query and end_query are provided
    mask = None
    if start_query and end_query:
        mask = lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query)

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
    upper_cell = create_note(
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
    base_filename = f"{output_dir}/{appendix}_{start_item}_{structure_formatted}_{sensor_type_formatted}_{group_args['name']}.pdf"
    pdf_filenames.append(base_filename)

    # Generate PDF
    pdf_generator.generate_pdf(pdf_path=base_filename)

    return pdf_filenames
