from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
)
from libs.utils.calc_helpers import (
    round_decimal,
    format_date_long,
    format_date_short,
    get_typical_range,
)
from libs.utils.config_loader import load_toml
from libs.utils.plot_helpers import get_unique_marker_convo
from modules.reporter.note_handler import NotesHandler
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.plot_builder import PlotMerger, PlotBuilder
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


COLOR_PALETTE = "cool"


def calculate_note_variables(dfs, sensor_names, serie_x, target_column, mask=None):
    """Calculate variables for notes based on multiple dataframes."""
    all_vars = []
    for df, name in zip(dfs, sensor_names):
        # Apply mask if provided
        if mask is not None:
            df = df[mask(df)]

        # Skip empty DataFrames
        if df.empty:
            continue

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
    limit,
    mask=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler()

    target_column_name = series_names[target_column].lower().split("(")[0]

    # Calculate variables for all data
    calc_vars = calculate_note_variables(
        data_sensors["df"], data_sensors["names"], serie_x, target_column, mask
    )

    # Find the maximum value across all series
    max_var = max(calc_vars, key=lambda v: v["max_value"])
    max_note = (
        f"El valor máximo de {target_column_name} es {round_decimal(max_var['max_value'], 2)} {unit_target}, "
        f"registrado en el sensor {max_var['sensor_name']} el día {format_date_short(max_var['max_date'])}."
    )

    azimuth_note = (
        "Se muestra el vector azimut absoluto sobre cada prisma de control topográfico."
    )

    # Define sections with the maximum value note
    sections = [
        {
            "title": "Notas:",
            "content": [max_note, azimuth_note],
            "format_type": "numbered",
        },
    ]

    return note_handler.create_notes(sections)


def create_map(
    dxf_path,
    data_sensors,
    target_column,
    arrow_column,
    unit_target,
    series_names,
    colorbar,
):
    target_column_name = series_names[target_column]

    plotter = PlotBuilder(ts_serie=True, ymargin=0)
    map_args = {
        "dxf_path": dxf_path,
        "size": [6.0, 4.5],  # Enlarged 3 times from original [2.0, 1.5]
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": False,
        "dxf_params": {"color": "black", "linestyle": "-", "linewidth": 0.02},
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    # Sort sensor data by maximum value of the target column in descending order
    sorted_sensors = sorted(
        zip(
            data_sensors["names"],
            data_sensors["df"],
            data_sensors["east"],
            data_sensors["north"],
        ),
        key=lambda x: x[1][target_column].max() if not x[1].empty else float("-inf"),
        reverse=True,
    )
    (
        data_sensors["names"],
        data_sensors["df"],
        data_sensors["east"],
        data_sensors["north"],
    ) = zip(*sorted_sensors)

    series_data = []

    # Generate unique color combinations for each sensor
    for i, name in enumerate(data_sensors["names"]):
        # Validate that all required data exists for this sensor
        if (
            i >= len(data_sensors["df"])
            or i >= len(data_sensors["east"])
            or i >= len(data_sensors["north"])
        ):
            continue

        # Check if dataframe is empty or missing required columns
        df = data_sensors["df"][i]
        if (
            df.empty
            or target_column not in df.columns
            or arrow_column not in df.columns
        ):
            continue

        color, marker = get_unique_marker_convo(
            i, len(data_sensors["names"]), color_palette=COLOR_PALETTE
        )

        try:
            # Skip if DataFrame is empty
            if df.empty:
                continue

            last_value = df[target_column].iloc[-1]
            last_value_formatted = round_decimal(last_value, 2)
            angle = df[arrow_column].iloc[-1]

            series_data.append(
                {
                    "x": [data_sensors["east"][i]],
                    "y": [data_sensors["north"][i]],
                    "color": color,
                    "linetype": "",
                    "lineweight": 0,
                    "marker": marker,
                    "markersize": 3.0,
                    "label": f"{name} - {last_value_formatted} {unit_target}",
                    "value": last_value,
                    "angle": angle,
                }
            )
        except Exception:
            continue

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

    # Add triangulation
    plotter.add_triangulation(
        data=series_data,
        colorbar=colorbar,
    )

    # Add arrows to the plot
    plotter.add_arrow(
        data=series_data,
        position="last",
        radius=50.0,
    )

    # Generate discrete colorbar
    return (
        plotter.get_drawing(),
        plotter.get_legend(
            box_width=6.0, box_height=4.5, ncol=1 if len(series_data) <= 25 else 2
        ),
        plotter.get_colorbar(
            box_width=2.0,
            box_height=1.0,
            label=target_column_name,
            vmin=min([s["value"] for s in series_data]),
            vmax=max([s["value"] for s in series_data]),
            colors=colorbar["colors"],
            thresholds=colorbar["thresholds"],
            type_colorbar="discrete",
        ),
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
    arrow_column = column_config["arrow_column"]
    unit_target = column_config["unit_target"]
    serie_x = column_config["serie_x"]
    sensor_type_name = column_config["sensor_type_name"]
    sensor_aka = column_config["sensor_aka"]
    colorbar_config = column_config["colorbar"]

    # Load configuration
    calc_config = load_toml(CALC_CONFIG_DIR, sensor_type)
    series_names = calc_config["names"]["es"]

    # Define mask for filtering data if start_query and end_query are provided
    mask = (
        (lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query))
        if start_query and end_query
        else None
    )

    # Create map and get its legend
    chart_cell, legend, colorbar = create_map(
        dxf_path,
        data_sensors,
        target_column,
        arrow_column,
        unit_target,
        series_names,
        colorbar_config,
    )

    plot_grid = PlotMerger(fig_size=(2, 4.5))
    plot_grid.create_grid(2, 1, row_ratios=[0.20, 0.80])
    plot_grid.add_object(legend, (0, 0))
    plot_grid.add_object(colorbar, (1, 0))

    map_legend = plot_grid.build(color_border="white", cell_spacing=0)

    # Create report components
    upper_cell = map_legend  # Map legend goes to upper_cell

    lower_cell = get_note_content(
        group_args,
        data_sensors,
        target_column,
        unit_target,
        series_names,
        serie_x,
        sensor_aka,
        None,  # No limit needed for notes-only
        mask,
    )

    logo_cell = load_svg(LOGO_SVG, 0.95)
    target_column_name = series_names[target_column].lower().split("(")[0]
    chart_title = f"Mapa de distribución de {target_column_name} / {structure_name}"

    # Set up PDF generator
    pdf_generator = ReportBuilder(
        sample="chart_landscape_a4_type_04",
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
    # Generate PDF
    pdf_generator.generate_pdf(pdf_path=base_filename)

    return pdf_filenames
