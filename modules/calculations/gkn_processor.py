import os
import pandas as pd
from datetime import datetime

def parse_gkn_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # Extraer fecha y hora
        date_line = lines[4].strip()   # línea 5 (índice 4)
        time_line = lines[5].strip()   # línea 6 (índice 5)
        
        date_str = date_line.split(':')[-1].strip()
        time_str = time_line.split(':')[-1].strip()

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H")
        except ValueError:
            dt = pd.NaT  # En caso de error
        
        # Extraer datos numéricos desde línea 10
        data_lines = lines[9:]  # línea 10 en adelante
        data = [line.strip().split(',') for line in data_lines if line.strip()]
        df = pd.DataFrame(data, columns=["FLEVEL", "A+", "A-", "B+", "B-"])
        
        # Convertir a tipo numérico
        df = df.apply(lambda x: pd.to_numeric(x.str.strip(), errors='coerce'))
        
        # Añadir las columnas "time" y "base_line"
        df.insert(0, "time", dt)

        df["base_line"] = False
        df["azimuth"] = 0.0
        
        return df

def process_all_gkn_files(folder_path):
    all_dfs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".gkn"):
            filepath = os.path.join(folder_path, filename)
            df = parse_gkn_file(filepath)
            all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# USO
folder = "/home/joelefrain/Downloads/geo_sentry/geo_sentry/var/sample_client/sample_project/processed_data/INC/DME_CHO/INC-101/"
output_file = "var/sample_client/sample_project/250430_Abril/preprocess/INC/DME_CHO.INC-101.csv"
df_final = process_all_gkn_files(folder)
df_final.to_csv(output_file, index=False, sep=";")
