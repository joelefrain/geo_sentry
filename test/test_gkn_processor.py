if __name__ == "__main__":
    # Example usage
    folder = "seed/sample_client/sample_project/250228_Febrero/DME Choloque/INCLINOMETRO/INC-101"
    output_file = (
        "var/sample_client/sample_project/250430_Abril/preprocess/INC/DME_CHO.INC-101.csv"
    )
    match_columns = ["time", "FLEVEL"]
    inc_params = {
        "date_format": "%m/%d/%y %H",
        "date_lines": (4, 5),
        "data_lines_start": 9,
        "azimuth_rad": 0.0,
        "enbankment_slope_rad": 0.0,
        "a_axis_scale": 0.005,
        "b_axis_scale": 0.005,
    }

    df_final = process_all_gkn_files(
        folder,
        match_columns,
        **inc_params,
    )
    df_final.to_csv(output_file, index=False, sep=";")