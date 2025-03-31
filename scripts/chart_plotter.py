import os
import sys
import locale
import random
import numpy as np
from matplotlib import pyplot as plt

# Add 'libs' path to sys.path
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_PATH)

import pandas as pd
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet

from libs.utils.config_loader import load_toml
from libs.utils.logger_config import get_logger, log_execution_time
from modules.reporter.plot_builder import PlotBuilder, PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from modules.reporter.note_handler import NotesHandler
from libs.utils.calculations import round_decimal, get_iqr_limits, round_lower, round_upper
from libs.utils.df_helpers import read_df_on_time_from_csv

logger = get_logger("scripts.chart_plotter")
locale.setlocale(locale.LC_TIME, "es_ES.utf8")

def generate_random_color():
    """Generate a random hex color."""
    return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

def get_palette_colors(n_colors, palette_name='viridis'):
    """Generate n colors from a matplotlib colormap."""
    cmap = plt.colormaps[palette_name]
    colors = [f'#{int(x[0]*255):02x}{int(x[1]*255):02x}{int(x[2]*255):02x}' 
              for x in cmap(np.linspace(0, 1, n_colors))]
    return colors

# Define rounding functions

def create_plot_data(df, df_name, cell_config, plot_series, plot_colors, plot_linestyles, 
                    plot_linewidths, plot_markers, plot_markersizes, to_combine, config, df_colors):
    """Create plot data dictionary for a single dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    df_name : str
        Name of the dataframe
    cell_config : dict
        Configuration for the cell
    plot_series : list
        List of series to plot
    plot_colors : list
        List of colors for each series
    plot_linestyles : list
        List of line styles for each series
    plot_linewidths : list
        List of line widths for each series
    plot_markers : list
        List of markers for each series
    plot_markersizes : list
        List of marker sizes for each series
    to_combine : bool
        Whether plots are being combined
    config : dict
        Main configuration dictionary
    df_colors : dict
        Dictionary mapping df_names to colors
        
    Returns
    -------
    list
        List of dictionaries containing plot data
    """
    return [
        {
            "x": df[cell_config["title_x"]].tolist(),
            "y": df[s].tolist(),
            "color": df_colors[df_name] if (to_combine and s == cell_config.get("common_title")) else c,
            "linestyle": lt,
            "linewidth": lw,
            "marker": m,
            "markersize": ms,
            "secondary_y": False,
            "label": df_name if (to_combine and s == cell_config.get("common_title")) 
                    else config["names"]["es"][s],
        }
        for s, c, lt, lw, m, ms in zip(
            plot_series, plot_colors, plot_linestyles, plot_linewidths,
            plot_markers, plot_markersizes
        )
    ]

def process_dataframes(dfs, df_names, config, start_query, end_query):
    """Process dataframes and generate charts and legends.
    
    Parameters
    ----------
    dfs : list
        List of dataframes to process
    df_names : list
        List of dataframe names
    config : dict
        Configuration dictionary from TOML file
    start_query : str
        Start date for filtering
    end_query : str
        End date for filtering
        
    Returns
    -------
    dict
        Dictionary containing 'charts' and 'legends' lists
    """
    # Generate colors from palette for each DataFrame
    palette_colors = get_palette_colors(len(df_names), 'viridis')
    df_colors = dict(zip(df_names, palette_colors))
    
    results = {
        'charts': [],
        'legends': []
    }

    # Filter dataframes based on plot query type
    for plot_key, plot_config in config["plots"].items():
        if plot_config["to_combine"]:
            cells_draw = {}
            for cell_key, cell_config in plot_config["cells"].items():
                plot_data = []
                for df_idx, (df_data, df_name) in enumerate(zip(dfs, df_names)):
                    # Apply query filter per cell
                    if cell_config.get("query_type") == "all":
                        df_filtered = df_data.copy()
                    else:  # query_type == "query"
                        mask = (df_data["time"] >= pd.to_datetime(start_query)) & (
                            df_data["time"] <= pd.to_datetime(end_query)
                        )
                        df_filtered = df_data[mask]
                    
                    plot_data.extend(create_plot_data(
                        df_filtered, df_name, cell_config,
                        cell_config["serie"], cell_config["color"],
                        cell_config["linestyle"], cell_config["linewidth"],
                        cell_config["marker"], cell_config["markersize"],
                        plot_config["to_combine"], config, df_colors
                    ))

                plotter = PlotBuilder()
                plotter_args = {
                    "data": plot_data,
                    "size": tuple(cell_config["size"]),
                    "title_x": config["names"]["es"][cell_config["title_x"]],
                    "title_y": config["names"]["es"][cell_config["title_y"]],
                    "title_chart": plot_config["title_chart"],
                    "show_legend": cell_config["show_legend"],
                }
                plotter.plot_series(**plotter_args)
                cells_draw[tuple(cell_config["position"])] = plotter.get_drawing()
                
                if cell_config["show_legend"]:
                    results['legends'].append(plotter.get_legend(4, 2))
                    
                plotter.close()

            grid = PlotMerger(fig_size=tuple(plot_config["fig_size"]))
            grid.create_grid(*plot_config["grid_size"])

            for position, draw in cells_draw.items():
                grid.add_object(draw, position)

            results['charts'].append({
                'key': plot_key,
                'draw': grid.build(color_border="white"),
                'title': plot_config["title_chart"],
                'combined': True
            })

        else:
            for df_idx, (df_data, df_name) in enumerate(zip(dfs, df_names)):
                cells_draw = {}
                for cell_key, cell_config in plot_config["cells"].items():
                    # Apply query filter per cell
                    if cell_config.get("query_type") == "all":
                        df_filtered = df_data.copy()
                    else:  # query_type == "query"
                        mask = (df_data["time"] >= pd.to_datetime(start_query)) & (
                            df_data["time"] <= pd.to_datetime(end_query)
                        )
                        df_filtered = df_data[mask]
                    
                    plot_data = create_plot_data(
                        df_filtered, df_name, cell_config,
                        cell_config["serie"], cell_config["color"],
                        cell_config["linestyle"], cell_config["linewidth"],
                        cell_config["marker"], cell_config["markersize"],
                        plot_config["to_combine"], config, df_colors
                    )
                    
                    plotter = PlotBuilder()
                    plotter_args = {
                        "data": plot_data,
                        "size": tuple(cell_config["size"]),
                        "title_x": config["names"]["es"][cell_config["title_x"]],
                        "title_y": config["names"]["es"][cell_config["title_y"]],
                        "title_chart": f"{plot_config['title_chart']} - {df_name}",
                        "show_legend": cell_config["show_legend"],
                    }
                    plotter.plot_series(**plotter_args)
                    cells_draw[tuple(cell_config["position"])] = plotter.get_drawing()
                    
                    if cell_config["show_legend"]:
                        results['legends'].append(plotter.get_legend(4, 2))
                        
                    plotter.close()

                if cells_draw:
                    grid = PlotMerger(fig_size=tuple(plot_config["fig_size"]))
                    grid.create_grid(*plot_config["grid_size"])

                    for position, draw in cells_draw.items():
                        grid.add_object(draw, position)

                    results['charts'].append({
                        'key': plot_key,
                        'draw': grid.build(color_border="white"),
                        'title': f"{plot_config['title_chart']} - {df_name}",
                        'combined': False,
                        'df_name': df_name
                    })

    return results

def generate_pdfs(charts_and_legends, report_params):
    """Generate PDFs for each chart with its corresponding legend.
    
    Parameters
    ----------
    charts_and_legends : dict
        Dictionary containing 'charts' and 'legends' lists
    report_params : dict
        Dictionary containing parameters for the report
    """
    item = 200
    
    for chart in charts_and_legends['charts']:
        appendix_item = f"{report_params['appendix_num']}.{item}"
        
        pdf_generator = ReportBuilder(
            sample=report_params['sample'],
            theme_color=report_params['theme_color'],
            theme_color_font=report_params['theme_color_font'],
            logo_cell=report_params['logo_cell'],
            upper_cell=report_params['note_paragraph'],
            lower_cell=report_params['map_draw'],
            chart_cell=chart['draw'],
            chart_title=chart['title'],
            num_item=appendix_item,
            project_code=report_params['engineer_code'],
            company_name=report_params['company_name'],
            project_name=report_params['engineer_name'],
            date=report_params['date_chart'],
            revision=report_params['revision'],
            elaborated_by=report_params['elaborated_by'],
            approved_by=report_params['approved_by'],
            doc_title=report_params['doc_title'],
        )
        
        filename = (f"{report_params['appendix_num']}.{item}_{chart['key']}"
                   f"{'_' + chart['df_name'] if not chart['combined'] else ''}.pdf")
        pdf_generator.generate_pdf(pdf_path=filename)
        item += 1

if __name__ == "__main__":
    client_keys = {
        "names": ["Shahuindo SAC", "Shahuindo"],
        "codes": ["Shahuindo_SAC", "Shahuindo"],
    }

    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }

    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }

    company_code = client_keys["codes"][0]
    project_code = client_keys["codes"][1]
    company_name = client_keys["names"][0]
    project_name = client_keys["names"][1]

    engineer_code = "1408.10.0050-0000"
    engineer_name = "Ingeniería de registro, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque"
    date_chart = "23-03-25"
    revision = "B"
    appendix_num = "4"

    elaborated_by = "J.A."
    approved_by = "R.L."

    doc_title = "@joelefrain"
    theme_color = "#006D77"
    theme_color_font = "white"
    logo_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\logo\logo_main.svg"

    sensor_type_name = sensor_names["PTA"]
    sensor_code = "PTA-SH23-101"
    
    config = load_toml(
        data_dir=r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\config\Shahuindo_SAC\Shahuindo\charts",
        toml_name="pta",
    )
    start_query = "2024-01-01 00:00:00"
    end_query = "2025-03-23 00:00:00"

    east = 808779.55
    north = 9157518.99

    dxf_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf"

    start_formatted = datetime.strptime(start_query, "%Y-%m-%d %H:%M:%S")
    start_formatted = start_formatted.strftime("%B %Y")
    end_formatted = datetime.strptime(end_query, "%Y-%m-%d %H:%M:%S")
    end_formatted = end_formatted.strftime("%B %Y")

    target = config["target"]["column"]
    unit = config["target"]["unit"]
    target_phrase = config["inline"]["es"]["target_phrase"]

    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\Shahuindo_SAC\Shahuindo\processed_data\PTA\DME_CHO.PH-SH23-103A.csv"
    )
    df_name = "PH-SH23-103A"
    df_2 = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\Shahuindo_SAC\Shahuindo\processed_data\PTA\DME_CHO.PH-SH23-103B.csv"
    )
    df2_name = "PH-SH23-103B"
    
    # Filtrar datos según el rango de tiempo
    mask = (df["time"] >= pd.to_datetime(start_query)) & (
        df["time"] <= pd.to_datetime(end_query)
    )
    df_filtered = df[mask]

    mask_2 = (df_2["time"] >= pd.to_datetime(start_query)) & (
        df_2["time"] <= pd.to_datetime(end_query)
    )
    df2_filtered = df_2[mask_2]

    total_records = len(df_filtered)
    avg_diff_time_rel = df_filtered["diff_time_rel"].mean()
    avg_diff_time_rel = round_decimal(avg_diff_time_rel, 2)
    last_value, last_time = df_filtered.iloc[-1][[target, "time"]]
    max_value, max_time = df_filtered.loc[df_filtered[target].idxmax(), [target, "time"]]

    note_number_records = f"Se registraron {total_records} lecturas durante el periodo entre {start_formatted} y {end_formatted}."
    note_freq_monit = f"La frecuencia de registro promedio es de {avg_diff_time_rel} días durante el periodo entre {start_formatted} y {end_formatted}."
    note_max_record = f"El valor máximo registrado de {target_phrase} fue {round_decimal(max_value, 2)} {unit} el {max_time.strftime('%d-%m-%Y')}."
    note_last_record = f"El último valor registrado de {target_phrase} fue {round_decimal(last_value, 2)} {unit} el {last_time.strftime('%d-%m-%Y')}."
    
    notes = [note_number_records, note_freq_monit, note_max_record, note_last_record]
    sections = [
        {'title': 'Ubicación:', 'content': ['Talud izquierdo'], 'format_type': 'paragraph'},
        {'title': 'Material:', 'content': ['Desmonte de mina'], 'format_type': 'paragraph'},
        {'title': 'Notas:', 'content': notes, 'format_type': 'numbered'},
        
    ]

    # Crear instancia del manejador de notas con estilo de bullets
    notes_handler = NotesHandler("default")  # Usar el nuevo estilo

    # Donde antes se usaba create_notes directamente, ahora usar:
    note_paragraph = notes_handler.create_notes(sections)

    map_plotter_args = {
        "data": [
            {
                "x": [east],
                "y": [north],
                "color": "red",
                "linestyle": "",
                "linewidth": 0,
                "marker": "o",
                "secondary_y": False,
                "label": "",
                "note": [sensor_code],
                "fontsize": 15,
                "markersize": 25,
            },
        ],
        "dxf_path": dxf_path,
        "size": (5, 5),
        "title_x": "",
        "title_y": "",
        "title_chart": "",
        "show_legend": True,
        "dxf_params": {
            "color": "black",
            "linestyle": "-",
            "linewidth": 0.02,
        },
        "format_params": {
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    }

    map_plotter = PlotBuilder(style_file="default")
    map_plotter.plot_series(**map_plotter_args)
    map_plotter.add_arrow(
        data=map_plotter_args["data"],
        position="first",
        angle=0,
        radius=0,
        color="red",
    )
    map_draw = map_plotter.get_drawing()
    map_plotter.close()
    
    map_draw = PlotMerger.scale_figure(map_draw, size=(1.5, 1.5))

    sample = "chart_landscape_a4_type_01"
    logo_cell = load_svg(logo_path, 0.08)

    # Create report parameters dictionary
    report_params = {
        'sample': sample,
        'theme_color': theme_color,
        'theme_color_font': theme_color_font,
        'logo_cell': logo_cell,
        'note_paragraph': note_paragraph,
        'map_draw': map_draw,
        'engineer_code': engineer_code,
        'company_name': company_name,
        'engineer_name': engineer_name,
        'date_chart': date_chart,
        'revision': revision,
        'elaborated_by': elaborated_by,
        'approved_by': approved_by,
        'doc_title': doc_title,
        'appendix_num': appendix_num
    }
    
    # Process dataframes and generate charts and legends
    charts_and_legends = process_dataframes(
        dfs=[df, df_2],
        df_names=[df_name, df2_name],
        config=config,
        start_query=start_query,
        end_query=end_query
    )
    
    # Generate PDFs with charts and legends
    generate_pdfs(charts_and_legends, report_params)
