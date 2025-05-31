import os
import pandas as pd
from datetime import datetime

DEFAULT_INC_PARAMS = {
    "date_format": "%m/%d/%y %H",
    "date_lines": (4, 5),
    "azimuth_rad": 0.0,
    "enbankment_slope_rad": 0.0,
    "a_axis_scale": 0.005,
    "b_axis_scale": 0.005,
}


def parse_gkn_file(filepath, **kwargs):
    params = {**DEFAULT_INC_PARAMS, **kwargs}

    with open(filepath, "r") as file:
        lines = file.readlines()

        date_line = lines[params["date_lines"][0]].strip()
        time_line = lines[params["date_lines"][1]].strip()

        date_str = date_line.split(":")[-1].strip()
        time_str = time_line.split(":")[-1].strip()

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", params["date_format"])
        except ValueError:
            dt = pd.NaT

        data_lines = lines[9:]
        data = [line.strip().split(",") for line in data_lines if line.strip()]
        df = pd.DataFrame(data, columns=["FLEVEL", "A+", "A-", "B+", "B-"])
        df = df.apply(lambda x: pd.to_numeric(x.str.strip(), errors="coerce"))

        df.insert(0, "time", dt)
        df["base_line"] = False

        # Asignar parámetros usando dict comprehension
        for key in [
            "azimuth_rad",
            "enbankment_slope_rad",
            "a_axis_scale",
            "b_axis_scale",
        ]:
            df[key] = params[key]

        return df


def process_all_gkn_files(folder, **kwargs):
    all_dfs = [
        parse_gkn_file(os.path.join(folder, f), **kwargs)
        for f in os.listdir(folder)
        if f.lower().endswith(".gkn")
    ]
    return pd.concat(all_dfs, ignore_index=True)


# USO
folder = "/home/joelefrain/Downloads/geo_sentry/geo_sentry/var/sample_client/sample_project/processed_data/INC/DME_CHO/INC-101/"
output_file = (
    "var/sample_client/sample_project/250430_Abril/preprocess/INC/DME_CHO.INC-101.csv"
)

inc_params = {
    "date_format": "%m/%d/%y %H",
    "date_lines": (4, 5),
    "azimuth_rad": 0.0,
    "enbankment_slope_rad": 0.0,
    "a_axis_scale": 0.005,
    "b_axis_scale": 0.005,
}

df_final = process_all_gkn_files(
    folder,
    **inc_params,
)
df_final.to_csv(output_file, index=False, sep=";")
