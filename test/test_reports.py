from modules.reporter.plot_builder import PlotBuilder
from modules.reporter.plot_merger import PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from svglib.svglib import svg2rlg
from io import BytesIO


def test_plot_dxf(size=(6, 4)):
    plotter = PlotBuilder()
    plotter.plot_series(
        data=[
            {
                "x": [810743.01, 809012.08],
                "y": [9156589.35, 9158189.5],
                "size": 50,
                "color": "red",
                "linetype": "",
                "lineweight": 0.02,
                "marker": "x",
                "label": "Series 1",
            },
        ],
        dxf_path=r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\test.dxf",
        size=size,
        title_x="",
        title_y="",
        title_chart="",
        show_legend=False,
        format_params={
            "show_grid": False,
            "show_xticks": False,
            "show_yticks": False,
        },
    )

    drawing = plotter.get_drawing()
    legend = plotter.get_legend(box_width=1.5, box_height=5)

    return drawing, legend


def test_plot_series(size=(6, 4)):
    data_1 = [
        {
            "x": [1, 2, 3.5],
            "y": [4, 5, 6],
            "angle": 45,
            "color": "red",
            "linetype": "--",
            "lineweight": 2,
            "marker": "x",
            "label": "Series 1",
        },
        {
            "x": [2, 2.5],
            "y": [5, 6.5],
            "angle": 58,
            "color": "blue",
            "linetype": "-",
            "lineweight": 1.5,
            "marker": "h",
            "secondary_y": False,
            "label": "serie otra",
        },
    ]

    data_2 = [
        {
            "x": [1, 2, 3],
            "y": [450, 451, 452],
            "color": "green",
            "linetype": "-",
            "lineweight": 1.5,
            "marker": "s",
            "secondary_y": True,
            "label": "",
        },
    ]

    data_3 = [
        {
            "x": [2, 2.5],
            "y": [5, 6.5],
            "color": "blue",
            "linetype": "-",
            "lineweight": 0,
            "marker": "*",
            "secondary_y": True,
            "label": "",
        },
    ]
    plotter = PlotBuilder()
    plotter.plot_series(
        data_1,
        size=size,
        title_x="X Axis",
        title_y="Y Axis",
        title_chart="Sample Chart",
        show_legend=True,
        xlim=(0, 5),
        ylim=(0, 10),
        invert_y=True,
    )
    plotter.add_secondary_y_axis(data_2, title_y2="Secondary Y Axis")
    plotter.add_color_bands(
        [4.5, 5, 5.5, 6], ["lightgreen", "khaki", "salmon"], ["Low", "Medium", "High"]
    )
    plotter.add_arrow(data=data_3, position="last", angle=45, radius=0.25, color="red")

    drawing = plotter.get_drawing()

    # Get the legend as a separate drawing
    legend = plotter.get_legend(box_width=1.5, box_height=5)

    return drawing, legend


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def create_chart_cell(chart_title, max_width, max_height):
        fig, ax = plt.subplots(figsize=(max_width / 100, max_height / 100))
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title(chart_title)
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf")
        buffer.seek(0)
        return svg2rlg(buffer)

    def create_note(notes):
        styles = getSampleStyleSheet()
        text_style = styles["Normal"]
        text_style.fontName = "Helvetica"
        return Paragraph(f"<i><b>Notas:</b></i><br/>{notes}", text_style)

    # Create multiple identical Plotter objects
    # drawing, legend = test_plot_series(size=(6, 4))
    draw1, legend = test_plot_series(size=(3.8, 2.5))
    draw2, legend = test_plot_series(size=(2, 2.5))
    draw3, legend = test_plot_dxf(size=(3.8, 2.5))

    grid = PlotMerger(fig_size=(7.5, 5.5))
    grid.create_grid(2, 2)

    # Añadir objetos con sus posiciones
    grid.add_object(draw1, (0, 0))
    grid.add_object(draw2, (0, 1))
    grid.add_object(draw3, (1, 0), span=(1, 2))

    # Construir y obtener el objeto svg2rlg final
    combined_svg = grid.build(color_border="white")

    sample = "chart_landscape_a4_type_01.toml"  # Ensure the correct TOML file is used
    project_code = "1408.10.0050-0000"
    company_name = "Shahindo SAC"
    project_name = "Ingeniería de registro, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque"
    date = "03-06-19"
    revision = "B"
    num_item = "4.100"

    elaborated_by = "J.A."
    approved_by = "R.L."
    doc_title = "SIG-AND"
    chart_title = "Ingeniería de plotting, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque"
    theme_color = "#0069AA"
    theme_color_font = "white"

    # Create the plot drawing
    chart_cell, upper_cell = combined_svg, legend
    # upper_cell = create_note("1. El gato<br/>2. El oso<br/>3. El perro Ingeniería de plotting, monitoreo y análisis del pad 1, pad 2A, pad 2B-2C, DME Sur y DME Choloque")

    # upper_cell = create_chart_cell("chart_title", 50, 200)
    lower_cell = create_chart_cell("lower_cell", 50, 80)

    # Load the logo
    logo_path = "data/logo/logo_main.svg"
    logo_cell = load_svg(logo_path, 0.95)  # Example max_width and max_height

    pdf_generator = ReportBuilder(
        sample=sample,
        theme_color=theme_color,
        theme_color_font=theme_color_font,
        logo_cell=logo_cell,
        upper_cell=upper_cell,
        lower_cell=lower_cell,
        chart_cell=chart_cell,
        chart_title=chart_title,
        num_item=num_item,
        project_code=project_code,
        company_name=company_name,
        project_name=project_name,
        date=date,
        revision=revision,
        elaborated_by=elaborated_by,
        approved_by=approved_by,
        doc_title=doc_title,
    )
    pdf_generator.generate_pdf(pdf_path="chart_landscape_a4_type_01.pdf")
