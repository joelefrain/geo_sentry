if __name__ == "__main__":
    import pdfkit
    
    # Configuración de wkhtmltopdf
    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
    
    html_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\analysis\reports\analysis_PCT-01_GC_HOR.html"
    
    # Opciones para mejorar la calidad del PDF
    options = {
        'page-width': '65mm',
        'page-height': '78mm',
        'margin-top': '0mm',
        'margin-right': '0mm',
        'margin-bottom': '0mm',
        'margin-left': '0mm',
        'encoding': 'UTF-8'
    }
    
    pdfkit.from_file(html_path, 'out2.pdf', configuration=config, options=options)