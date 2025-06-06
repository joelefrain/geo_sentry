import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


import pandas as pd

# Add 'libs' path to sys.path

from modules.reporter.plot_builder import PlotMerger, PlotBuilder
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.calculations import get_percentile_value, round_upper
from libs.utils.plot_helpers import get_unique_marker_convo
from libs.utils.text_helpers import to_sentence_format
from libs.utils.config_loader import load_toml
from libs.utils.calculations import round_decimal, format_date_long, format_date_short
from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
)

COLOR_PALETTE_1 = "Spectral"
COLOR_PALETTE_2 = "PiYG"


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
    serie_y,
    mask=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler()
    target_column_name = to_sentence_format(
        series_names[target_column], mode="decapitalize"
    ).split("(")[0]
    serie_y_name = series_names[serie_y]
    serie_y_name_decap = to_sentence_format(serie_y_name, mode="decapitalize").split(
        "("
    )[0]
    y_unit = (
        serie_y_name[serie_y_name.find("(") + 1 : serie_y_name.find(")")].strip()
        if "(" in serie_y_name and ")" in serie_y_name
        else None
    )

    # Apply mask and calculate combined statistics
    dfs = [df[mask(df)] if mask else df for df in data_sensors["df"]]
    combined_df = pd.concat(dfs, ignore_index=True)

    first_date = pd.to_datetime(combined_df[serie_x].iloc[0])
    last_date = pd.to_datetime(combined_df[serie_x].iloc[-1])

    # Get max value and corresponding serie_y
    idx_max = combined_df[target_column].abs().idxmax()
    max_value = combined_df.loc[idx_max, target_column]
    max_date = pd.to_datetime(combined_df.loc[idx_max, serie_x])
    max_serie_y = combined_df.loc[idx_max, serie_y]

    # Get last date's maximum value
    last_date = pd.to_datetime(combined_df[serie_x]).max()
    last_date_df = combined_df[pd.to_datetime(combined_df[serie_x]) == last_date]

    if not last_date_df.empty:
        last_idx = last_date_df[target_column].abs().idxmax()
        last_value = last_date_df.loc[last_idx, target_column]
        last_serie_y = last_date_df.loc[last_idx, serie_y]
    else:
        last_value = None
        last_date = None
        last_serie_y = None

    # Define sections with separated notes
    sections = [
        {
            "title": "Ubicación:",
            "content": [f"{group_args['name']} - {group_args['location']}"],
            "format_type": "paragraph",
        },
        {
            "title": "Notas:",
            "content": [
                f"En el periodo entre {format_date_long(first_date)} y {format_date_long(last_date)}, "
                f"se registró un valor máximo de {target_column_name} de {round_decimal(max_value, 2)} {unit_target} "
                f"a {round_decimal(max_serie_y, 2)} {y_unit} de {serie_y_name_decap} "
                f"el día {format_date_short(max_date)}.",
                f"El último valor registrado de {target_column_name} fue de {round_decimal(last_value, 2)} {unit_target} "
                f"a {round_decimal(last_serie_y, 2)} {y_unit} de {serie_y_name_decap} "
                f"el día {format_date_short(last_date)}.",
            ],
            "format_type": "numbered",
        },
    ]

    return note_handler.create_notes(sections)


def create_map(dxf_path, data_sensors):
    plotter = PlotBuilder(ts_serie=True, ymargin=0)
    map_args = {
        "dxf_path": dxf_path,
        "size": [1.3, 0.975],
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
            i, len(data_sensors["names"]), color_palette=COLOR_PALETTE_1
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


def create_non_ts_cell_1(
    data_sensors,
    series_names,
    target_column,
    serie_x,
    serie_y,
):
    plotter = PlotBuilder(ts_serie=False, ymargin=0.0)

    # Plot formatting
    plot_format = {
        "size": (8.5, 10),
        "title_x": series_names[target_column],
        "title_y": series_names[serie_y],
        "title_chart": f"{series_names[target_column]}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    max_abs_values = []  # Lista para almacenar valores máximos absolutos

    for df, name in zip(data_sensors["df"], data_sensors["names"]):
        if target_column in df.columns and serie_y in df.columns:
            # Get unique dates
            unique_dates = sorted(df[serie_x].unique())

            for i, date in enumerate(unique_dates):
                date_df = df[df[serie_x] == date]
                color, marker = get_unique_marker_convo(
                    i, len(unique_dates), color_palette=COLOR_PALETTE_1
                )

                # Get absolute maximum value for this series
                abs_values = date_df[target_column].abs()
                if not abs_values.empty:
                    max_abs_values.append(abs_values.max())

                series.append(
                    {
                        "x": date_df[target_column].tolist(),
                        "y": date_df[serie_y].tolist(),
                        "label": format_date_short(date),
                        "color": color,
                        "linestyle": "-",
                        "lineweight": 1,
                        "marker": marker,
                        "markersize": 4,
                    }
                )

    # Calculate x-axis limits based on 90th percentile of absolute maximum values
    percentile = get_percentile_value(max_abs_values, 90.0)
    limit = round_upper(abs(percentile * 1.5))

    plotter.plot_series(
        data=series,
        size=plot_format["size"],
        title_x=plot_format["title_x"],
        title_y=plot_format["title_y"],
        title_chart=plot_format["title_chart"],
        show_legend=plot_format["show_legend"],
        invert_y=True,
        xlim=(-limit, limit),  # Symmetric limits based on percentile
    )

    return limit, plotter.get_drawing(), plotter.get_legend(
        box_width=7.5,
        box_height=1.0,
        ncol=2,
    )


def create_ts_cell_2(
    data_sensors,
    series_names,
    target_column,
    serie_x,
    serie_y,
    limit,
):
    plotter = PlotBuilder(ts_serie=True)
    target_column_name = to_sentence_format(
        series_names[target_column], mode="decapitalize"
    )
    serie_y_name = to_sentence_format(series_names[serie_y], mode="decapitalize")

    # Plot formatting
    plot_format = {
        "size": (8, 4),
        "title_x": series_names[serie_x],
        "title_y": series_names[target_column],
        "title_chart": f"Registro histórico de {target_column_name.split('(')[0]} por {serie_y_name.split('(')[0]}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    all_y_value = []
    all_values = []

    for df, name in zip(data_sensors["df"], data_sensors["names"]):
        if target_column in df.columns and serie_y in df.columns:
            unique_group = sorted(df[serie_y].unique())
            all_y_value.extend(unique_group)
            all_values.extend(df[target_column].dropna().tolist())

            for i, y_value in enumerate(unique_group):
                y_value_df = df[df[serie_y] == y_value]
                color, marker = get_unique_marker_convo(
                    i, len(unique_group), color_palette=COLOR_PALETTE_2
                )
                series.append(
                    {
                        "x": y_value_df[serie_x].tolist(),
                        "y": y_value_df[target_column].tolist(),
                        "label": y_value,
                        "color": color,
                        "linestyle": "-",
                        "lineweight": 1,
                        "marker": marker,
                        "markersize": 4,
                    }
                )

    plotter.plot_series(
        data=series,
        size=plot_format["size"],
        title_x=plot_format["title_x"],
        title_y=plot_format["title_y"],
        title_chart=plot_format["title_chart"],
        show_legend=plot_format["show_legend"],
        ylim=(-limit, limit),
    )

    # Get colors from series for colorbar
    colors = [series["color"] for series in series]

    return plotter.get_drawing(), plotter.get_colorbar(
        box_width=7.5,
        box_height=0.5,
        label=series_names[serie_y],
        vmin=min(all_y_value),
        vmax=max(all_y_value),
        colors=colors,
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
    """Generate chart reports for time series plots defined in column_config.

    Returns:
        list: Paths to the generated PDF files.
    """
    plots = column_config["plots"]
    sensor_type_name = column_config["sensor_type_name"]

    # Load configuration
    calc_config = load_toml(CALC_CONFIG_DIR, sensor_type)
    series_names = calc_config["names"]["es"]

    os.makedirs(output_dir, exist_ok=True)
    pdf_filenames = []
    current_item = start_item

    for plot in plots:
        target_column = plot["target_column"]
        unit_target = plot.get("unit_target", None)
        serie_x = plot["series_x"]
        serie_y = plot["series_y"]

        # Generate chart components
        limit, chart_cell1, legend1 = create_non_ts_cell_1(
            data_sensors,
            series_names,
            target_column,
            serie_x,
            serie_y,
        )

        chart_cell2, legend2 = create_ts_cell_2(
            data_sensors,
            series_names,
            target_column,
            serie_x,
            serie_y,
            limit,
        )

        # Define mask for filtering data
        mask = None
        if start_query and end_query:
            mask = lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query)

        # Create and configure plot grid
        plot_grid = PlotMerger(fig_size=(5, 8))
        plot_grid.create_grid(3, 1, row_ratios=[0.03, 0.29, 0.68])
        plot_grid.add_object(chart_cell1, (0, 0))
        plot_grid.add_object(chart_cell2, (1, 0))
        plot_grid.add_object(legend2, (2, 0))

        chart_cell = plot_grid.build(color_border="white", cell_spacing=0)

        plot_grid = PlotMerger(fig_size=(1.5, 5.5))
        plot_grid.create_grid(1, 1, row_ratios=[1])
        plot_grid.add_object(legend1, (0, 0))

        upper_cell = plot_grid.build(color_border="white", cell_spacing=0)

        target_column_name = to_sentence_format(
            series_names[target_column], mode="decapitalize"
        )

        # Create report components
        middle_cell = get_note_content(
            group_args,
            data_sensors,
            target_column,
            unit_target,
            series_names,
            serie_x,
            serie_y,
            mask,
        )
        lower_cell = create_map(dxf_path, data_sensors)
        logo_cell = load_svg(LOGO_SVG, 0.75)
        chart_title = " / ".join(
            filter(
                None,
                [
                    f"Registro histórico de {target_column_name}",
                    group_args["name"],
                    structure_name,
                ],
            )
        )

        # Generate PDF
        pdf_generator = ReportBuilder(
            sample="chart_portrait_a4_type_03",
            logo_cell=logo_cell,
            upper_cell=upper_cell,
            lower_cell=lower_cell,
            middle_cell=middle_cell,
            chart_cell=chart_cell,
            chart_title=chart_title,
            num_item=f"{appendix}.{current_item}",
            **static_report_params,
        )

        structure_formatted = structure_code.replace(" ", "_")
        sensor_type_formatted = sensor_type.replace(" ", "_").upper()
        pdf_filename = f"{output_dir}/{appendix}_{current_item:03}_{structure_formatted}_{sensor_type_formatted}.pdf"

        pdf_generator.generate_pdf(pdf_path=pdf_filename)
        pdf_filenames.append(pdf_filename)
        current_item += 1

    return pdf_filenames
