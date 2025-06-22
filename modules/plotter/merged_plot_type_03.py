import os

import pandas as pd

from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
    SENSOR_VISUAL_CONFIG,
)

from libs.utils.config_loader import load_toml
from libs.utils.text_helpers import to_sentence_format
from libs.utils.plot_helpers import get_unique_marker_convo

from libs.utils.calc_helpers import get_symetric_range
from libs.utils.calc_helpers import round_decimal, format_date_long, format_date_short

from modules.reporter.plot_merger import PlotMerger
from modules.reporter.plot_builder import PlotBuilder
from modules.reporter.note_handler import NotesHandler
from modules.reporter.report_builder import ReportBuilder, load_svg


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
    limit,
    mask=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler(style_name="small")

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

    # Get historical maximum (without mask but with limit)
    historical_combined_df = pd.concat(
        [df for df in data_sensors["df"]], ignore_index=True
    )
    if limit:
        # Group by date and check if all values are within limits
        date_groups = historical_combined_df.groupby(serie_x)
        valid_dates = [
            date
            for date, group in date_groups
            if all(
                group[target_column].isna()
                | group[target_column].between(limit[0], limit[1])
            )
        ]
        historical_combined_df = historical_combined_df[
            historical_combined_df[serie_x].isin(valid_dates)
        ]

    # Apply mask and calculate combined statistics
    dfs = [df[mask(df)] if mask else df for df in data_sensors["df"]]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter by limit before any analysis
    if limit:
        # Group by date and check if all values are within limits
        date_groups = combined_df.groupby(serie_x)
        valid_dates = [
            date
            for date, group in date_groups
            if all(
                group[target_column].isna()
                | group[target_column].between(limit[0], limit[1])
            )
        ]
        combined_df = combined_df[combined_df[serie_x].isin(valid_dates)]

    # Get statistics from filtered data
    first_date = pd.to_datetime(combined_df[serie_x].iloc[0])
    last_date = pd.to_datetime(combined_df[serie_x].iloc[-1])

    # Get max value from filtered data
    idx_max = combined_df[target_column].abs().idxmax()
    max_value = combined_df.loc[idx_max, target_column]
    max_date = pd.to_datetime(combined_df.loc[idx_max, serie_x])
    max_serie_y = combined_df.loc[idx_max, serie_y]

    # Get historical maximum from filtered data
    hist_idx_max = historical_combined_df[target_column].abs().idxmax()
    hist_max_value = historical_combined_df.loc[hist_idx_max, target_column]
    hist_max_date = pd.to_datetime(historical_combined_df.loc[hist_idx_max, serie_x])
    hist_max_serie_y = historical_combined_df.loc[hist_idx_max, serie_y]

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

    # Filter values by limit if provided
    if limit:
        combined_df = combined_df[combined_df[target_column].abs() <= limit[1]]

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
                f"Históricamente, el valor máximo registrado de {target_column_name} fue de {round_decimal(hist_max_value, 2)} {unit_target} "
                f"a {round_decimal(hist_max_serie_y, 2)} {y_unit} de {serie_y_name_decap} "
                f"el día {format_date_short(hist_max_date)}.",
                f"En el periodo entre {format_date_long(first_date)} y {format_date_long(last_date)}, "
                f"se registró un valor máximo de {target_column_name} de {round_decimal(max_value, 2)} {unit_target} "
                f"a {round_decimal(max_serie_y, 2)} {y_unit} de {serie_y_name_decap} "
                f"el día {format_date_short(max_date)}.",
                f"El último valor registrado de {target_column_name} fue de {round_decimal(last_value, 2)} {unit_target} "
                f"a {round_decimal(last_serie_y, 2)} {y_unit} de {serie_y_name_decap} "
                f"el día {format_date_short(last_date)}.",
                f"La frecuencia de monitoreo promedio es de un registro cada {round_decimal(combined_df[serie_x].drop_duplicates().sort_values().diff().dt.days.mean(), 2)} días."
                if len(combined_df[serie_x].drop_duplicates()) > 1
                else "No es posible calcular la frecuencia de monitoreo promedio por falta de datos suficientes.",
            ],
            "format_type": "numbered",
        },
    ]

    return note_handler.create_notes(sections), first_date


def create_map(data_sensors, dxf_path, tif_path, project_epsg, sensor_visual_config):
    plotter = PlotBuilder(style_file="default", ts_serie=True, ymargin=0)
    map_args = {
        "dxf_path": dxf_path,
        "tif_path": tif_path,
        "project_epsg": project_epsg,
        "size": [1.3, 0.975],
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": True,
        "dxf_params": {"linestyle": "-", "linewidth": 0.01, "color": "whitesmoke"},
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    # Set marker style from sensor visual config or default to 'o'
    marker = sensor_visual_config.get("mpl_marker", "o")

    series_data = []
    notes = []

    # Generate unique color combinations for each sensor
    for i, name in enumerate(data_sensors["code"]):
        color, _ = get_unique_marker_convo(
            i, len(data_sensors["code"]), color_palette=COLOR_PALETTE_1
        )
        series_data.append(
            {
                "x": data_sensors["east"][i],
                "y": data_sensors["north"][i],
                "color": color,
                "linestyle": "",
                "linewidth": 0,
                "marker": marker,
                "markersize": 10,
                "label": "",
            }
        )
        note = {
            "text": name,
            "x": data_sensors["east"][i],
            "y": data_sensors["north"][i],
        }
        notes.append(note)

    plotter.plot_series(
        data=series_data,
        **map_args,
    )

    plotter.add_notes(notes)
    
    return plotter.get_drawing()


def create_non_ts_cell(
    data_sensors,
    series_names,
    target_column,
    serie_x,
    serie_y,
):
    plotter = PlotBuilder(ts_serie=False, ymargin=0.0)
    # Plot formatting
    plot_format = {
        "size": (8, 14),
        "title_x": series_names[target_column],
        "title_y": series_names[serie_y],
        "title_chart": f"{series_names[target_column]}",
        "show_legend": False,
        "legend_location": "upper right",
        "grid": True,
    }

    series = []
    max_abs_values = []  # Lista para almacenar valores máximos absolutos

    for df, name in zip(data_sensors["df"], data_sensors["code"]):
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
                        "linewidth": 1,
                        "marker": marker,
                        "markersize": 4,
                    }
                )

    # Calculate limits based on percentile of absolute maximum values
    limit = get_symetric_range(max_abs_values, percentile=95, scale=1.5)

    plotter.plot_series(
        data=series,
        size=plot_format["size"],
        title_x=plot_format["title_x"],
        title_y=plot_format["title_y"],
        title_chart=plot_format["title_chart"],
        show_legend=plot_format["show_legend"],
        invert_y=True,
        xlim=limit,  # Symmetric limits based on percentile
    )

    return (
        limit,
        plotter.get_drawing(),
        plotter.get_legend(
            box_width=2.0,
            box_height=1.0,
            ncol=2,
        ),
    )


def generate_report(
    data_sensors,
    group_args,
    appendix,
    start_item,
    structure_code,
    structure_name,
    sensor_type,
    output_dir,
    static_report_params,
    column_config,
    **plot_params,
):
    """Generate chart reports for time series plots defined in column_config.

    Returns:
        list: Paths to the generated PDF files.
    """
    plots = column_config["plots"]
    sensor_type_name = column_config["sensor_type_name"]

    dxf_path = plot_params.get("dxf_path", None)
    if not dxf_path:
        raise ValueError("DXF path must be provided in plot_params.")

    tif_path = plot_params.get("tif_path", None)
    if not tif_path:
        raise ValueError("TIF path must be provided in plot_params.")

    project_epsg = plot_params.get("project_epsg", None)
    if not project_epsg:
        raise ValueError("Project EPSG must be provided in plot_params.")

    start_query = plot_params.get("start_query", None)
    end_query = plot_params.get("end_query", None)

    # Load configuration
    calc_config = load_toml(CALC_CONFIG_DIR, sensor_type)
    series_names = calc_config["names"]["es"]

    # Load visual config for sensor type
    sensor_visual_config = SENSOR_VISUAL_CONFIG.get(sensor_type, {})
    if not sensor_visual_config:
        raise ValueError(
            f"No visual configuration found for sensor type: {sensor_type}"
        )

    os.makedirs(output_dir, exist_ok=True)
    pdf_filenames = []
    chart_titles = []
    current_item = start_item

    # Ordenar las series según el nombre del sensor antes de procesar
    sorted_indices = sorted(
        range(len(data_sensors["code"])), key=lambda k: data_sensors["code"][k]
    )
    data_sensors = {
        "df": [data_sensors["df"][i] for i in sorted_indices],
        "code": [data_sensors["code"][i] for i in sorted_indices],
        "east": [data_sensors["east"][i] for i in sorted_indices],
        "north": [data_sensors["north"][i] for i in sorted_indices],
    }

    for plot in plots:
        target_column = plot["target_column"]
        unit_target = plot.get("unit_target", None)
        serie_x = plot["series_x"]
        serie_y = plot["series_y"]

        # Generate chart components
        limit, chart_cell, legend = create_non_ts_cell(
            data_sensors,
            series_names,
            target_column,
            serie_x,
            serie_y,
        )

        # Define mask for filtering data
        mask = (
            (lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query))
            if start_query and end_query
            else None
        )

        # Create and configure plot grid
        plot_grid = PlotMerger(fig_size=(5, 8))
        plot_grid.create_grid(1, 1, row_ratios=[1.0])
        plot_grid.add_object(chart_cell, (0, 0))

        chart_cell = plot_grid.build(color_border="white", cell_spacing=0)

        plot_grid = PlotMerger(fig_size=(1.5, 5.5))
        plot_grid.create_grid(1, 1, row_ratios=[1])
        plot_grid.add_object(legend, (0, 0))

        upper_cell = plot_grid.build(color_border="white", cell_spacing=0)

        target_column_name = to_sentence_format(
            series_names[target_column], mode="decapitalize"
        )

        # Create report components
        middle_cell, first_date = get_note_content(
            group_args,
            data_sensors,
            target_column,
            unit_target,
            series_names,
            serie_x,
            serie_y,
            limit,
            mask,
        )
        lower_cell = create_map(
            data_sensors, dxf_path, tif_path, project_epsg, sensor_visual_config
        )
        logo_cell = load_svg(LOGO_SVG, 0.75)
        chart_title = " / ".join(
            filter(
                None,
                [
                    f"Registro histórico de {target_column_name.split('(')[0]}",
                    group_args["name"],
                    f"Medida base en {format_date_short(first_date)}",
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
        chart_titles.append(chart_title)
        current_item += 1

    return pdf_filenames, chart_titles
