import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from modules.calculations.excel_processor import ExcelProcessor
from modules.calculations.data_analysis import JumpDetector, AnomalyDetector
from modules.calculations.data_processor import DataProcessor
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_loader import load_toml

from pathlib import Path

base_path = Path(__file__).parent.parent


class TestExcelProcessor:
    def __init__(self):
        self.input_folder = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\seed\250231_Febrero\Pad 2A\CELDAS DE PRESIÓN"
        self.output_folder_base = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\var\process\ShahuinoSAC\Shahuindo"
        self.toml_path = r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\data\config\Shahuindo_SAC\Shahuindo\excel_format\cpcv.toml"

    def test_process_directory(self):
        processor = ExcelProcessor(self.toml_path)
        processor.preprocess_excel_directory(
            input_folder=self.input_folder,
            output_folder_base=self.output_folder_base,
            sensor_type="CPCV",
            code="PAD2A",
            exclude_sheets=[],
            data_config=processor.config,
            custom_functions={},
            selected_attr=processor.config["process"].get("selected_attr"),
        )


class TestJumpDetectorPosition:
    def test_pre_post_selection(self):
        csv_path = base_path / "data/test/serie_03.csv"

        df = read_df_on_time_from_csv(csv_path)
        print(df.head())

        # Probando selección 'pre'
        detector_pre = JumpDetector(position="pre")
        pre_jumps = detector_pre.detect(df, "temperature")
        print("Saltos detectados --->")
        print(pre_jumps)

        # Probando selección 'post'
        detector_post = JumpDetector(position="post")
        post_jumps = detector_post.detect(df, "temperature")
        print("Saltos detectados --->")
        print(post_jumps)


class TestAnomalyDetector:
    def test_detect_anomalies(self):
        # Crear dataset sintético
        csv_path = base_path / "data/test/serie_03.csv"

        df = read_df_on_time_from_csv(csv_path)
        print(df.head())

        # Configurar detector
        detector = AnomalyDetector(window_size=5, min_threshold=98, sensitivity=95)
        anomalies = detector.detect(df, "temperature")

        print("Anomalías detectadas --->")
        print(anomalies)

class TestMatchProcessor:
    def test_match_processor(self):

        toml_path = "modules/calculations/data"
        input_csv = "var/sample_client/sample_project/250430_Abril/preprocess/INC/DME_CHO.INC-101.csv"
        output_csv = "var/sample_client/sample_project/processed_data/INC/DME_CHO.INC-101.processed.csv"

        config_toml = load_toml(data_dir=toml_path, toml_name="inc")
        match_columns = config_toml["process_config"]["match_columns"]
        print(f"Match columns: {match_columns}")

        # Initialize processor and process data
        processor = DataProcessor(config_name="inc")
        df = processor.load_data(input_csv)
        print(df)
        df = processor.clean_data(df, match_columns=match_columns)
        print(df)
        df = processor.process_absolute_values_by_match(df)

        # Save results
        df.to_csv(output_csv, index=False, sep=';')
        print(f"Processed data saved to: {output_csv}")



if __name__ == "__main__":
    # tester = TestExcelProcessor()
    # tester.test_process_directory()
    # TestJumpDetectorPosition().test_pre_post_selection()
    # TestAnomalyDetector().test_detect_anomalies()
    TestMatchProcessor().test_match_processor()

