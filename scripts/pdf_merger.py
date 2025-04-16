import os
import argparse
from pathlib import Path
from PyPDF2 import PdfMerger

def find_pdf_files(input_dir: str) -> list:
    """Busca recursivamente archivos PDF en el directorio de entrada.
    
    Args:
        input_dir (str): Ruta del directorio de entrada
        
    Returns:
        list: Lista de rutas de archivos PDF encontrados
    """
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files, key=lambda x: os.path.basename(x))

def merge_pdfs(pdf_files: list, output_path: str) -> None:
    """Fusiona los archivos PDF en un único archivo.
    
    Args:
        pdf_files (list): Lista de rutas de archivos PDF a fusionar
        output_path (str): Ruta del archivo PDF de salida
    """
    merger = PdfMerger()
    
    for pdf_file in pdf_files:
        merger.append(pdf_file)
    
    # Asegurarse que el directorio de salida existe
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Guardar el PDF fusionado
    merger.write(output_path)
    merger.close()

def main():

    # Convertir rutas a absolutas
    input_dir = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\outputs"
    output_file = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\outputs\Anexos de procesamiento.pdf"
    
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio de entrada '{input_dir}' no existe")
        return
    
    # Encontrar todos los archivos PDF
    pdf_files = find_pdf_files(input_dir)
    
    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{input_dir}'")
        return
    
    print(f"Se encontraron {len(pdf_files)} archivos PDF")
    
    try:
        merge_pdfs(pdf_files, output_file)
        print(f"Los PDFs se han fusionado exitosamente en '{output_file}'")
    except Exception as e:
        print(f"Error al fusionar los PDFs: {str(e)}")

if __name__ == '__main__':
    main()