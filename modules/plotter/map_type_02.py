import os

import pandas as pd

from libs.utils.config_variables import (
    LOGO_SVG,
    CALC_CONFIG_DIR,
    SENSOR_VISUAL_CONFIG,
)

from libs.utils.calc_helpers import (
    round_decimal,
    format_date_short,
)

from libs.utils.config_loader import load_toml

from modules.reporter.plot_merger import PlotMerger
from modules.reporter.plot_builder import PlotBuilder
from modules.reporter.note_handler import NotesHandler
from modules.reporter.report_builder import ReportBuilder, load_svg


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
    static_notes=None,
):
    # Initialize NotesHandler
    note_handler = NotesHandler()

    target_column_name = series_names[target_column].lower().split("(")[0]

    # Calculate variables for all data
    calc_vars = calculate_note_variables(
        data_sensors["df"], data_sensors["names"], serie_x, target_column, mask
    )

    # Find the maximum value among the last values of each sensor
    if calc_vars:
        max_var = max(calc_vars, key=lambda v: v["last_value"])
        max_note = (
            f"El valor máximo de {target_column_name} es {round_decimal(max_var['last_value'], 2)} {unit_target}, "
            f"registrado en el sensor {max_var['sensor_name']} el día {format_date_short(max_var['last_date'])}."
        )
    else:
        max_note = "No hay registros válidos para mostrar el valor máximo de los últimos registros."

    notes = [max_note]
    if static_notes:
        notes.extend(static_notes)

    sections = [
        {
            "title": "Notas:",
            "content": notes,
            "format_type": "numbered",
        },
    ]

    return note_handler.create_notes(sections)


def create_map(
    sensor_type,
    dxf_path,
    data_sensors,
    target_column,
    arrow_column,
    unit_target,
    series_names,
    colorbar,
    tif_path,
    project_epsg,
):
    marker = SENSOR_VISUAL_CONFIG[sensor_type]["mpl_marker"]
    color = SENSOR_VISUAL_CONFIG[sensor_type]["color"]

    # Crear un diccionario para almacenar la última información de cada ubicación
    location_dict = {}

    # Crear lista de notas
    notes = []

    # Recopilar la última información de cada sensor por ubicación
    for name, df, east, north in zip(
        data_sensors["names"],
        data_sensors["df"],
        data_sensors["east"],
        data_sensors["north"],
    ):
        # Omitir si el DataFrame está vacío
        if df.empty:
            continue

        note = {
            "text": name,
            "x": east,
            "y": north,
        }
        notes.append(note)

        # Obtener el último valor y ángulo
        last_value = df[target_column].iloc[-1]
        angle = df[arrow_column].iloc[-1]

        # Usar como clave una tupla de la ubicación (este, norte)
        key = (east, north)

        # Actualizar el diccionario solo si la ubicación no está ya en el diccionario
        if key not in location_dict:
            location_dict[key] = {
                "name": name,
                "value": last_value,
                "angle": angle,
            }

    # Generar series_data usando solo el máximo valor por ubicación
    series_data = []
    for idx, ((east, north), info) in enumerate(location_dict.items()):
        last_value = info["value"]
        angle = info["angle"]
        last_value_formatted = round_decimal(last_value, 2)
        series_data.append(
            {
                "x": [east],
                "y": [north],
                "color": color,
                "linestyle": "",
                "linewidth": 0,
                "marker": marker,
                "markersize": 3.0,
                "label": f"{info['name']} ({last_value_formatted} {unit_target})",
                "value": last_value,
                "angle": angle,
            }
        )

    plotter = PlotBuilder(style_file="high_quality", ts_serie=True, ymargin=0)
    map_args = {
        "dxf_path": dxf_path,
        "tif_path": tif_path,
        "project_epsg": project_epsg,
        "size": [6.0, 4.5],
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": False,
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    plotter.plot_series(
        data=series_data,
        **map_args,
    )

    plotter.add_triangulation(
        data=series_data,
        colorbar=colorbar,
    )

    # plotter.add_arrow(
    #     data=series_data,
    #     position="last",
    #     radius=50.0,
    # )

    plotter.add_notes(notes)

    # Generate discrete colorbar
    return (
        plotter.get_drawing(),
        plotter.get_legend(
            box_width=6.0, box_height=4.5, ncol=1 if len(series_data) <= 25 else 2
        ),
        plotter.get_colorbar(
            box_width=2.0,
            box_height=1.0,
            label=series_names[target_column],
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
    static_notes = column_config.get("static_notes", None)

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

    # Define mask for filtering data if start_query and end_query are provided
    mask = (
        (lambda df: (df[serie_x] >= start_query) & (df[serie_x] <= end_query))
        if start_query and end_query
        else None
    )

    # Create map and get its legend
    chart_cell, legend, colorbar = create_map(
        sensor_type,
        dxf_path,
        data_sensors,
        target_column,
        arrow_column,
        unit_target,
        series_names,
        colorbar_config,
        tif_path=tif_path,
        project_epsg=project_epsg,
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
        static_notes,
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
    chart_titles = []

    # Generate base filename
    base_filename = f"{output_dir}/{appendix}_{start_item:03}_{structure_formatted}_{sensor_type_formatted}.pdf"
    pdf_filenames.append(base_filename)
    chart_titles.append(chart_title)

    # Generate PDF
    pdf_generator.generate_pdf(pdf_path=base_filename)

    return pdf_filenames, chart_titles
