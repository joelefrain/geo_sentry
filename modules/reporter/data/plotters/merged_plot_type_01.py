from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
)
from libs.utils.calc_helpers import round_decimal, format_date_long, format_date_short
from libs.utils.config_loader import load_toml
from libs.utils.plot_helpers import get_unique_marker_convo
from modules.reporter.note_handler import NotesHandler
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.plot_builder import PlotMerger, PlotBuilder
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")))


# Add 'libs' path to sys.path


COLOR_PALETTE = "Spectral"


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
            (last_date - first_date).days /
            total_records if total_records > 0 else 0
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
    mask=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler()

    target_column_name = series_names[target_column].lower().split("(")[0]

    # Apply mask and calculate combined statistics
    dfs = [df[mask(df)] if mask else df for df in data_sensors["df"]]
    combined_df = pd.concat(dfs, ignore_index=True)

    first_date = pd.to_datetime(combined_df[serie_x].iloc[0])
    last_date = pd.to_datetime(combined_df[serie_x].iloc[-1])
    max_value = combined_df[target_column].max()
    max_date = pd.to_datetime(
        combined_df.loc[combined_df[target_column].idxmax(), serie_x]
    )

    valid_dfs = [
        df
        for df in dfs
        if not df.empty
        and target_column in df.columns
        and len(df[target_column].dropna()) > 0
    ]
    if valid_dfs:
        last_value = max(df[target_column].iloc[-1] for df in valid_dfs)
    else:
        last_value = None

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
                f"el día {format_date_short(max_date)}.",
                f"El último valor registrado de {target_column_name} fue de {round_decimal(last_value, 2)} {unit_target} "
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
            i, len(data_sensors["names"]), color_palette=COLOR_PALETTE)
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


def create_ts_cell_1(
    data_sensors,
    series_names,
    target_column,
    serie_x,
    primary_title_y,
    sensor_type_name,
):
    plotter = PlotBuilder(ts_serie=True)

    # Series style definitions
    series_styles = {
        target_column: {
            "color": "blue",
            "linestyle": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[target_column],
        }
    }

    # Plot formatting
    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": primary_title_y,
        "title_chart": f"Registro histórico de {series_names[target_column].lower().split('(')[0]}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    total_dfs = len(data_sensors["df"])
    for i, (df, name) in enumerate(zip(data_sensors["df"], data_sensors["names"])):
        if target_column in df.columns:
            color, marker = get_unique_marker_convo(
                i, total_dfs, color_palette=COLOR_PALETTE
            )
            label = name

            series.append(
                {
                    "x": df[serie_x].tolist(),
                    "y": df[target_column].tolist(),
                    "label": label,
                    "color": color,
                    "linestyle": series_styles[target_column]["linestyle"],
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

    len_series = len(series)
    if len_series >= 6:
        ncol = 6
    else:
        ncol = max(1, len_series / 2)

    return plotter.get_drawing(), plotter.get_legend(
        box_width=7.5,
        box_height=1.0,
        ncol=ncol,
    )


def create_ts_cell_2(
    data_sensors, start_query, end_query, series_names, target_column, serie_x
):
    plotter = PlotBuilder(ts_serie=True)

    # Series style definitions
    series_styles = {
        target_column: {
            "color": "blue",
            "linestyle": "-",
            "lineweight": 1,
            "marker": "o",
            "markersize": 4,
            "label_prefix": series_names[target_column],
        },
    }

    unique_serie = target_column

    # Format start and end dates
    start_date_formatted = format_date_long(pd.to_datetime(start_query))
    end_date_formatted = format_date_long(pd.to_datetime(end_query))

    # Plot formatting
    plot_format = {
        "size": (8, 3),
        "title_x": series_names[serie_x],
        "title_y": series_names[target_column],
        "title_chart": f"Registro de {series_names[target_column].lower().split('(')[0]}entre {start_date_formatted} y {end_date_formatted}",
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

        for column, style in series_styles.items():
            if column in filtered_df.columns:
                if column == unique_serie:
                    color, marker = get_unique_marker_convo(
                        i, total_dfs, color_palette=COLOR_PALETTE
                    )
                else:
                    color = style["color"]
                    marker = style["marker"]

                label = name if column == unique_serie else style["label_prefix"]

                series.append(
                    {
                        "x": filtered_df[serie_x].tolist(),
                        "y": filtered_df[column].tolist(),
                        "label": label,
                        "color": color,
                        "linestyle": style["linestyle"],
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

    len_series = len(series)
    if len_series >= 6:
        ncol = 6
    else:
        ncol = max(1, len_series / 2)

    return plotter.get_drawing(), plotter.get_legend(
        box_width=7.5,
        box_height=1.0,
        ncol=ncol,
    )


def create_non_ts_cell_1(
    data_sensors,
    series_names,
    target_column,
    serie_x,
    primary_title_y,
    serie_aka,
    sensor_type_name,
    name,
):
    plotter = PlotBuilder(ts_serie=False)

    # Series style definitions
    series_styles = {
        target_column: {
            "color": "green",
            "linestyle": "-",
            "lineweight": 1,
            "marker": "s",
            "markersize": 4,
            "label_prefix": series_names[target_column],
        },
        "initial_position": {
            "color": "orange",
            "linestyle": "",
            "lineweight": 0,
            "marker": "^",
            "markersize": 6,
            "label": "Posición inicial",
        },
        "final_position": {
            "color": "red",
            "linestyle": "",
            "lineweight": 0,
            "marker": "s",
            "markersize": 6,
            "label": "Posición final",
        },
    }

    # Plot formatting
    plot_format = {
        "size": (8.5, 5.5),
        "title_x": series_names[serie_x],
        "title_y": primary_title_y,
        "title_chart": f"{serie_aka} de {sensor_type_name} {name}",
        "show_legend": True,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    total_dfs = len(data_sensors["df"])
    for i, (df, series_name) in enumerate(
        zip(data_sensors["df"], data_sensors["names"])
    ):
        if target_column in df.columns:
            color, marker = get_unique_marker_convo(
                i, total_dfs, color_palette=COLOR_PALETTE
            )
            label = series_name

            # Add the main series
            series.append(
                {
                    "x": df[serie_x].tolist(),
                    "y": df[target_column].tolist(),
                    "label": label,
                    "color": color,
                    "linestyle": series_styles[target_column]["linestyle"],
                    "lineweight": series_styles[target_column]["lineweight"],
                    "marker": marker,
                    "markersize": series_styles[target_column]["markersize"],
                }
            )

            # Calculate and add the initial position
            initial_x = df[serie_x].iloc[0]
            initial_y = df[target_column].iloc[0]
            series.append(
                {
                    "x": [initial_x],
                    "y": [initial_y],
                    "label": series_styles["initial_position"]["label"],
                    "color": series_styles["initial_position"]["color"],
                    "linestyle": series_styles["initial_position"]["linestyle"],
                    "lineweight": series_styles["initial_position"]["lineweight"],
                    "marker": series_styles["initial_position"]["marker"],
                    "markersize": series_styles["initial_position"]["markersize"],
                }
            )

            # Calculate and add the final position
            final_x = df[serie_x].iloc[-1]
            final_y = df[target_column].iloc[-1]
            series.append(
                {
                    "x": [final_x],
                    "y": [final_y],
                    "label": series_styles["final_position"]["label"],
                    "color": series_styles["final_position"]["color"],
                    "linestyle": series_styles["final_position"]["linestyle"],
                    "lineweight": series_styles["final_position"]["lineweight"],
                    "marker": series_styles["final_position"]["marker"],
                    "markersize": series_styles["final_position"]["markersize"],
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
    return plotter.get_drawing()


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
    """Generate chart reports for a list of plots defined in column_config.

    Returns:
        list: Paths to the generated PDF files.
    """
    # Extract plot configurations
    plots = column_config["plots"]  # List of plot configurations
    sensor_type_name = column_config["sensor_type_name"]

    # Load configuration
    calc_config = load_toml(CALC_CONFIG_DIR, sensor_type)
    series_names = calc_config["names"]["es"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    pdf_filenames = []
    current_item = start_item

    for plot in plots:
        target_column = plot["target_column"]
        # Optional for ts_serie=False
        unit_target = plot.get("unit_target", None)
        ts_serie = plot["ts_serie_flag"]
        serie_x = plot["series_x"]
        serie_aka = plot.get(
            "serie_aka", None
        )  # Default to series name if not provided

        if ts_serie:
            # Generate chart components for ts_serie=True
            chart_cell1, legend1 = create_ts_cell_1(
                data_sensors,
                series_names,
                target_column,
                serie_x,
                series_names[target_column],  # Use the name of target_column
                sensor_type_name,
            )
            chart_cell2, legend2 = create_ts_cell_2(
                data_sensors,
                start_query,
                end_query,
                series_names,
                target_column,
                serie_x,
            )

            # Define mask for filtering data if start_query and end_query are provided
            mask = (lambda df: (df[serie_x] >= start_query) & (
                df[serie_x] <= end_query)) if start_query and end_query else None

            # Create and configure plot grid
            plot_grid = PlotMerger(fig_size=(7.5, 5.5))
            plot_grid.create_grid(4, 1, row_ratios=[0.06, 0.44, 0.06, 0.44])
            plot_grid.add_object(chart_cell1, (0, 0))
            plot_grid.add_object(legend1, (1, 0))
            plot_grid.add_object(chart_cell2, (2, 0))
            plot_grid.add_object(legend2, (3, 0))

            chart_cell = plot_grid.build(color_border="white", cell_spacing=0)

            target_column_name = series_names[target_column].lower().split("(")[
                0]

            # Create report components
            upper_cell = get_note_content(
                group_args,
                data_sensors,
                target_column,
                unit_target,
                series_names,
                serie_x,
                mask,
            )
            lower_cell = create_map(dxf_path, data_sensors)
            logo_cell = load_svg(LOGO_SVG, 0.95)
            chart_title_elements = [
                f"Registro histórico de {target_column_name}",
                group_args["name"],
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
                num_item=f"{appendix}.{current_item}",
                **static_report_params,
            )

            # Format the PDF filename
            structure_formatted = structure_code.replace(" ", "_")
            sensor_type_formatted = sensor_type.replace(" ", "_").upper()
            pdf_filename = f"{output_dir}/{appendix}_{current_item:03}_{structure_formatted}_{sensor_type_formatted}.pdf"

            # Generate PDF
            pdf_generator.generate_pdf(pdf_path=pdf_filename)
            pdf_filenames.append(pdf_filename)

            # Increment the item number for the next report
            current_item += 1

        else:
            # Generate chart components for ts_serie=False
            for i, (df, name) in enumerate(
                zip(data_sensors["df"], data_sensors["names"])
            ):
                chart_cell1 = create_non_ts_cell_1(
                    # Pass only the current series
                    {"df": [df], "names": [name]},
                    series_names,
                    target_column,
                    serie_x,
                    series_names[target_column],
                    serie_aka,
                    sensor_type_name,
                    name,  # Pass name to the function
                )

                # Create and configure plot grid
                plot_grid = PlotMerger(fig_size=(8.5, 5.5))
                plot_grid.create_grid(1, 1, row_ratios=[1])
                plot_grid.add_object(chart_cell1, (0, 0))

                chart_cell = plot_grid.build(
                    color_border="white", cell_spacing=0)

                chart_title_elements = [
                    f"{serie_aka} de {sensor_type_name}",
                    name,
                    group_args["name"],
                    structure_name,
                ]
                chart_title = " / ".join(filter(None, chart_title_elements))

                # Create report components
                logo_cell = load_svg(LOGO_SVG, 0.95)

                # Set up PDF generator
                pdf_generator = ReportBuilder(
                    sample="chart_landscape_a4_type_02",
                    logo_cell=logo_cell,
                    chart_cell=chart_cell,
                    chart_title=chart_title,
                    num_item=f"{appendix}.{current_item}",
                    **static_report_params,
                )

                # Format the PDF filename
                structure_formatted = structure_code.replace(" ", "_")
                sensor_type_formatted = sensor_type.replace(" ", "_").upper()
                pdf_filename = f"{output_dir}/{appendix}_{current_item:03}_{structure_formatted}_{sensor_type_formatted}_{name.replace(' ', '_')}.pdf"

                # Generate PDF
                pdf_generator.generate_pdf(pdf_path=pdf_filename)
                pdf_filenames.append(pdf_filename)

                # Increment the item number for the next report
                current_item += 1

    return pdf_filenames
