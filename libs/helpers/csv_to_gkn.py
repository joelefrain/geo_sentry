import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd

from libs.utils.config_logger import get_logger

logger = get_logger("libs.helpers.csv_to_gkn")


def parse_metadata(lines):
    keys = {
        "Project Name": "PROJECT",
        "Hole Name": "HOLE NO.",
        "Reading Date": "DATE",
        "Reading Time": "TIME",
        "Probe Name": "PROBE NO.",
        "File Name": "FILE NAME",
    }

    meta = {}
    for line in lines:
        for k, v in keys.items():
            if k in line:
                meta[v] = line.split(":", 1)[1].strip()
                break
    return meta


def read_csv_with_metadata(path):
    try:
        logger.info(f"Leyendo archivo: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        meta = parse_metadata(lines[2:11])
        sample_line = lines[15]
        delimiter = "," if "," in sample_line else ";"
        logger.info(f"Separador detectado: '{delimiter}'")

        profile_map = {
            "Profile/A": ["A+", "A-", "Sum", "Diff", "Diff/2", "Defl", "Level"],
            "Profile/B": ["B+", "B-", "Sum", "Diff", "Diff/2", "Defl", "Level"],
        }

        headers = next((h for key, h in profile_map.items() if key in path), None)
        if not headers:
            raise ValueError(f"No se puede determinar el tipo de archivo para: {path}")

        df = pd.read_csv(path, skiprows=15, header=None, names=headers, sep=delimiter)
        df.columns = df.columns.str.strip()

        for col in headers:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Level"])

        logger.info(f"Columnas detectadas en {path}: {df.columns.tolist()}")
        logger.info(f"Leído: {len(df)} filas")
        logger.info(f"Meta: {meta}")

        return meta, df

    except Exception as e:
        logger.error(f"Error leyendo {path}: {e}")
        return None, pd.DataFrame()


def read_and_validate(path, required_columns):
    meta, df = read_csv_with_metadata(path)
    if df.empty or not all(col in df.columns for col in required_columns):
        logger.warning(f"Archivo inválido o columnas faltantes: {path}")
        return meta, None
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df = df.dropna(subset=["Level"])
    return meta, df


def build_combined_dataframe(a_df, b_df):
    df = pd.DataFrame({"FLEVEL": a_df["Level"], "A+": a_df["A+"], "A-": a_df["A-"]})
    if b_df is not None:
        df = df.merge(
            b_df[["Level", "B+", "B-"]], left_on="FLEVEL", right_on="Level", how="left"
        ).drop(columns="Level")
    else:
        df["B+"] = ""
        df["B-"] = ""
    return df


def format_metadata_field(meta, key):
    return meta.get(key, "") if meta else ""


def clean_date_field(date_str):
    return (
        date_str.replace(",", "").replace(";", "").replace('"', "") if date_str else ""
    )


def write_gkn_file(path, meta, df):
    try:
        with open(path, "w", encoding="utf-8") as out:
            out.write("***\n")
            out.write("GK 604M(v1.0,02/25);2.0;FORMAT II\n")
            out.write(f"PROJECT  :{format_metadata_field(meta, 'PROJECT')}\n")
            out.write(f"HOLE NO. :{format_metadata_field(meta, 'HOLE NO.')}\n")
            out.write(f"DATE     :{clean_date_field(meta.get('DATE', ''))}\n")
            out.write(f"TIME     :{format_metadata_field(meta, 'TIME')}\n")
            out.write(f"PROBE NO.:{format_metadata_field(meta, 'PROBE NO.')}\n")
            out.write(f"FILE NAME:{format_metadata_field(meta, 'FILE NAME')}\n")
            out.write(f"#READINGS:{len(df)}\n")
            out.write("FLEVEL,    A+,    A-,    B+,    B-\n")
            for _, row in df.iterrows():
                out.write(
                    f"{row['FLEVEL']:6}, {row['A+']:6}, {row['A-']:6}, {row['B+']:6}, {row['B-']:6}\n"
                )
        logger.info(f"Escrito: {path}")
    except Exception as e:
        logger.exception(f"Error escribiendo {path}: {e}")


def process_files(a_folder, b_folder, output_folder):
    for file in os.listdir(a_folder):
        if not file.endswith(".csv"):
            continue

        try:
            a_path = os.path.join(a_folder, file)
            b_file = file.replace("A.csv", "B.csv")
            b_path = os.path.join(b_folder, b_file)

            logger.info(f"Procesando archivo: {file}")

            a_meta, a_df = read_and_validate(a_path, ["Level", "A+", "A-"])
            if a_df is None:
                continue

            b_meta, b_df = (None, None)
            if os.path.exists(b_path):
                logger.info(f"Archivo B encontrado: {b_path}")
                b_meta, b_df = read_and_validate(b_path, ["Level", "B+", "B-"])
                if b_df is None:
                    continue
            else:
                logger.info(
                    f"Archivo B no encontrado: {b_path}, se usará DataFrame vacío."
                )

            combined_df = build_combined_dataframe(a_df, b_df)
            output_path = os.path.join(output_folder, file.replace(".csv", ".gkn"))
            write_gkn_file(output_path, a_meta, combined_df)

        except Exception as e:
            logger.exception(f"Error procesando archivo {file}: {e}")

# --- USO ---
if __name__ == "__main__":
    structure_name = "DME Sur"
    structure_code = "DME_SUR"
    sensor = "DMS-100"
    cut_off = "250430_Abril"
    a_folder = f"seed/sample_client/sample_project/{cut_off}/{structure_name}/INCLINOMETROS/{sensor}/Profile/A"
    b_folder = f"seed/sample_client/sample_project/{cut_off}/{structure_name}/INCLINOMETROS/{sensor}/Profile/B"
    output_folder = (
        f"var/sample_client/sample_project/processed_data/INC/{structure_code}/{sensor}"
    )
    os.makedirs(output_folder, exist_ok=True)

    print(f"Carpeta A: {a_folder}")
    print(f"Carpeta B: {b_folder}")
    print(f"Carpeta de salida: {output_folder}")

    # Ejecutar el proceso
    process_files(a_folder, b_folder, output_folder)
