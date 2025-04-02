import os
import sys
from glob import glob
from pathlib import Path

# Add 'libs' path to sys.path
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_PATH)

from modules.reporter.plot_builder import PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from svglib.svglib import svg2rlg
from io import BytesIO

def get_base_filename(filepath):
    return Path(filepath).stem.split('_', 1)[1]

def get_structure_from_filename(filename):
    # Extract structure code from filename (e.g., PCT-01_GC_HOR -> gc)
    parts = Path(filename).stem.split('_')
    return parts[1].lower() if len(parts) > 1 else None

def get_chart_title(base_name, structure_name):
    # Extract the code (e.g., PCT-01)
    code = base_name.split('_')[0]
    
    if base_name.endswith('_TOT'):
        return f"Análisis estadístico de desplazamiento total absoluto - {code} / {structure_name}"
    elif base_name.endswith('_HOR'):
        return f"Análisis estadístico de desplazamiento horizontal absoluto - {code} / {structure_name}"
    else:
        return f"Análisis de la serie de tiempo de {base_name} / {structure_name}"

def process_charts(base_paths, output_dir, structure_name, start_item=1):
    for idx, paths in enumerate(base_paths, start=start_item):
        decomp_path, forecast_path, report_path = paths
        
        # Load SVGs
        decomposition_draw = load_svg(decomp_path, 1)
        forecast_draw = load_svg(forecast_path, 1)
        report_draw = load_svg(report_path, 1)

        # Create grid
        grid = PlotMerger(fig_size=(6, 8))
        grid.create_grid(2, 2, row_ratios=[0.55, 0.45], col_ratios=[0.7, 0.3])

        # Add objects
        grid.add_object(decomposition_draw, (0, 0))
        grid.add_object(report_draw, (0, 1))
        grid.add_object(forecast_draw, (1, 0), span=(1, 2))

        # Build final svg2rlg object
        chart_svg = grid.build(color_border="white", cell_spacing=5)

        # Get base filename and create appropriate title
        base_name = get_base_filename(decomp_path)
        chart_title = get_chart_title(base_name, structure_name)
        
        # PDF generation parameters
        params = {
            "sample": "chart_portrait_a4_type_02",
            "project_code": "1421.10.0083-0000",
            "company_name": "Compañía Minera Raura SA",
            "project_name": "Ingeniería de Registro para componentes geotécnicos 2025",
            "date": "04-04-25",
            "revision": "B",
            "num_item": f"4.{idx}",
            "elaborated_by": "J.A.",
            "approved_by": "D.B.",
            "doc_title": "SIG-AND",
            "chart_title": chart_title,
            "theme_color": "#0069AA",
            "theme_color_font": "white",
        }

        # Load logo
        logo_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\logo\logo_main_anddes.svg"
        logo_cell = load_svg(logo_path, 0.65)

        # Generate PDF
        pdf_generator = ReportBuilder(
            logo_cell=logo_cell,
            chart_cell=chart_svg,
            **params
        )
        
        output_filename = f"4.{idx:03d}_{base_name}.pdf"
        pdf_generator.generate_pdf(pdf_path=os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    base_dir = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var"
    
    # Define structures in order
    structures = ["dd_abra", "dd_hidro", "dd_brunilda", "dd_gayco_630", "dd_gayco_580", "dd_gerencia"]
    names = ["Depósito de desmonte Abra", "Depósito de desmonte Hidro", "Depósito de desmonte Brunilda", 
             "Depósito de desmonte Gayco 630", "Depósito de desmonte Gayco 580", "Depósito de desmonte Gerencia"]
    structure_dict = dict(zip(structures, names))
    
    current_item = 1  # Initialize counter for consecutive numbering
    
    # Process structures in order
    for structure_code in structures:
        structure_name = structure_dict[structure_code]
        
        # Define structure-specific paths
        plot_dir = os.path.join(base_dir, structure_code, "plots")
        report_dir = os.path.join(base_dir, structure_code, "reports")
        output_dir = os.path.join(base_dir, structure_code, "pdf_reports")
        
        os.makedirs(output_dir, exist_ok=True)

        # Get and sort decomposition files for consistent ordering
        decomp_files = sorted(glob(os.path.join(plot_dir, "decomposition_*.svg")))
        base_paths = []
        
        for decomp_file in decomp_files:
            base_name = get_base_filename(decomp_file)
            forecast_file = os.path.join(plot_dir, f"forecast_{base_name}.svg")
            report_file = os.path.join(report_dir, f"analysis_{base_name}.svg")
            
            if os.path.exists(forecast_file) and os.path.exists(report_file):
                base_paths.append((decomp_file, forecast_file, report_file))

        if base_paths:
            print(f"Processing {structure_name} (items {current_item}-{current_item + len(base_paths) - 1})...")
            process_charts(base_paths, output_dir, structure_name, start_item=current_item)
            current_item += len(base_paths)  # Update counter for next structure
