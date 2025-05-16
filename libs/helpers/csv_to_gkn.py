import os
import pandas as pd

def parse_metadata(lines):
    meta = {}
    for line in lines:
        if "Project Name" in line:
            meta["PROJECT"] = line.split(":")[1].strip()
        elif "Hole Name" in line:
            meta["HOLE NO."] = line.split(":")[1].strip()
        elif "Reading Date" in line:
            meta["DATE"] = line.split(":")[1].strip()
        elif "Reading Time" in line:
            meta["TIME"] = line.split(":")[1].strip()
        elif "Probe Name" in line:
            meta["PROBE NO."] = line.split(":")[1].strip()
        elif "File Name" in line:
            meta["FILE NAME"] = line.split(":")[1].strip()

    return meta

def read_csv_with_metadata(path):
    print(f"Leyendo archivo: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer metadatos de las líneas 2 a 11
    meta = parse_metadata(lines[2:11])
    
    # Determinar el separador analizando la línea 16 (índice 15)
    sample_line = lines[15]
    delimiter = ',' if ',' in sample_line else ';'
    print(f"Separador detectado: '{delimiter}'")  # Depuración

    # Determinar encabezados según el tipo de archivo
    if "Profile/A" in path:
        headers = ['A+', 'A-', 'Sum', 'Diff', 'Diff/2', 'Defl', 'Level']
    elif "Profile/B" in path:
        headers = ['B+', 'B-', 'Sum', 'Diff', 'Diff/2', 'Defl', 'Level']
    else:
        raise ValueError(f"No se puede determinar el tipo de archivo para: {path}")
    
    # Leer el DataFrame desde la línea 16 (índice 15) y asignar encabezados manualmente
    df = pd.read_csv(path, skiprows=15, header=None, names=headers, sep=delimiter)
    df.columns = df.columns.str.strip()  # Limpiar nombres de columnas

    # Eliminar filas no válidas (que contengan valores no numéricos o NaN en columnas relevantes)
    numeric_columns = ['Level'] + [col for col in headers if col not in ['Level']]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Level'])  # Asegurarse de que 'Level' no tenga valores NaN

    print(f"Columnas detectadas en {path}: {df.columns.tolist()}")  # Depuración
    print(f"Leído: {len(df)} filas")
    print(f"Meta: {meta}")
    return meta, df

def process_files(a_folder, b_folder, output_folder):
    for file in os.listdir(a_folder):
        if not file.endswith(".csv"):
            continue

        a_path = os.path.join(a_folder, file)
        # Reemplazar la letra 'A' por 'B' en el nombre del archivo para buscar en la carpeta B
        b_file = file.replace("A.csv", "B.csv")
        b_path = os.path.join(b_folder, b_file)

        # Leer archivos A y B
        a_meta, a_df = read_csv_with_metadata(a_path)
        b_meta, b_df = (None, None)
        if os.path.exists(b_path):
            print(f"Archivo B encontrado: {b_path}")
            b_meta, b_df = read_csv_with_metadata(b_path)
        else:
            print(f"Archivo B no encontrado: {b_path}, se usará un DataFrame vacío.")

        # Filtrar columnas relevantes
        print(f"Columnas en archivo A ({file}): {a_df.columns.tolist()}")  # Depuración
        if 'Level' not in a_df.columns or 'A+' not in a_df.columns or 'A-' not in a_df.columns:
            print(f"Saltando archivo A: {file}, columnas requeridas no encontradas.")
            continue
        if b_df is not None:
            print(f"Columnas en archivo B ({file}): {b_df.columns.tolist()}")  # Depuración
            if 'Level' not in b_df.columns or 'B+' not in b_df.columns or 'B-' not in b_df.columns:
                print(f"Saltando archivo B: {file}, columnas requeridas no encontradas.")
                continue

        # Verificar que los valores en 'Level' sean números y eliminar registros no válidos
        a_df['Level'] = pd.to_numeric(a_df['Level'], errors='coerce')
        a_df = a_df.dropna(subset=['Level'])

        if b_df is not None:
            b_df['Level'] = pd.to_numeric(b_df['Level'], errors='coerce')
            b_df = b_df.dropna(subset=['Level'])

        # Crear DataFrame combinado
        df = pd.DataFrame({
            'FLEVEL': a_df['Level'],
            'A+': a_df['A+'],
            'A-': a_df['A-']
        })

        if b_df is not None:
            df = df.merge(b_df[['Level', 'B+', 'B-']], left_on='FLEVEL', right_on='Level', how='left')
            df.drop(columns='Level', inplace=True)
        else:
            df['B+'] = ''
            df['B-'] = ''

        # Formatear la fecha para eliminar símbolos ',', ';' y '"'
        formatted_date = a_meta.get('DATE', '').replace(',', '').replace(';', '').replace('"', '')

        # Nombre de salida
        gkn_name = file.replace(".csv", ".gkn")
        output_path = os.path.join(output_folder, gkn_name)

        # Escribir archivo GKN
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write("***\n")
            out.write("GK 604M(v1.0,02/25);2.0;FORMAT II\n")
            out.write(f"PROJECT  :{a_meta.get('PROJECT', '')}\n")
            out.write(f"HOLE NO. :{a_meta.get('HOLE NO.', '')}\n")
            out.write(f"DATE     :{formatted_date}\n")  # Usar la fecha formateada
            out.write(f"TIME     :{a_meta.get('TIME', '')}\n")
            out.write(f"PROBE NO.:{a_meta.get('PROBE NO.', '')}\n")
            out.write(f"FILE NAME:{a_meta.get('FILE NAME', '')}\n")
            out.write(f"#READINGS:{len(df)}\n")
            out.write("FLEVEL,    A+,    A-,    B+,    B-\n")
            for _, row in df.iterrows():
                out.write(f"{row['FLEVEL']:6}, {row['A+']:6}, {row['A-']:6}, {row['B+']:6}, {row['B-']:6}\n")

        print(f"[✓] Escrito: {output_path}")

if __name__ == "__main__":
    # Definir rutas de entrada y salida

    structure_name = "DME Sur"
    structure_code = "DME_SUR"
    sensor = "DMS-100"
    cut_off = "250430_Abril"
    a_folder = f"seed/sample_client/sample_project/{cut_off}/{structure_name}/INCLINOMETROS/{sensor}/Profile/A"
    b_folder = f"seed/sample_client/sample_project/{cut_off}/{structure_name}/INCLINOMETROS/{sensor}/Profile/B"
    output_folder = f"var/sample_client/sample_project/processed_data/INC/{structure_code}/{sensor}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Carpeta A: {a_folder}")
    print(f"Carpeta B: {b_folder}")
    print(f"Carpeta de salida: {output_folder}")

    # Ejecutar el proceso
    process_files(a_folder, b_folder, output_folder)
