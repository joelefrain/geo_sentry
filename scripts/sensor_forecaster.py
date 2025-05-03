<<<<<<< HEAD
import os
import sys
import locale
import glob

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from darts import TimeSeries
from modules.reporter.plot_builder import PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_plot import PlotConfig

PlotConfig.setup_matplotlib()

# Configuración global de formato numérico
locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")


@dataclass
class Config:
    """Configuration settings"""
    FORECAST_HORIZON: int = 3
    FORMAT_TYPE: str = "svg"


class TimeSeriesData:
    """Data handling class for a single time series"""

    def __init__(self, file_path: str, date_col: str = "time", target_col: str = None):
        """Initialize TimeSeriesData with a single target column.

        Args:
            file_path (str): Path to the CSV file
            date_col (str): Name of the date column
            target_col (str): Target column to analyze
            separator (str): CSV separator character
        """
        self.df = read_df_on_time_from_csv(file_path, set_index=False)
        self._prepare_data(date_col, target_col)

    def to_timeseries(self) -> TimeSeries:
        """Convert the prepared data to Darts TimeSeries format"""
        return TimeSeries.from_dataframe(
            self.series_df, time_col=self.date_col, value_cols=self.target_column
        )

    def _prepare_data(self, date_col: str, target_col: str = None) -> None:
        """Prepare time series data for analysis.

        Args:
            date_col (str): Name of the date column
            target_col (str): Name of the target column
        """
        if target_col is None:
            raise ValueError("Target column must be specified")

        if target_col not in self.df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in columns: {list(self.df.columns)}"
            )

        self.target_column = target_col
        self.date_col = date_col
        self.series_df = self.df[[date_col, target_col]].dropna()

        # Filtro de anomalías
        window_size = 7  # Ventana rolling de 7 días
        rolling_mean = (
            self.series_df[self.target_column]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        rolling_std = (
            self.series_df[self.target_column]
            .rolling(window=window_size, min_periods=1)
            .std()
        )

        # Umbrales dinámicos ±2 sigma
        upper_bound = rolling_mean + 2 * rolling_std
        lower_bound = rolling_mean - 2 * rolling_std

        # Identificar y reemplazar anomalías
        anomalies_mask = (self.series_df[self.target_column] > upper_bound) | (
            self.series_df[self.target_column] < lower_bound
        )
        self.series_df.loc[anomalies_mask, self.target_column] = np.nan

        # Interpolación lineal para valores faltantes
        self.series_df[self.target_column] = self.series_df[
            self.target_column
        ].interpolate(method="linear")


class PlotSaver:
    """Handles plot saving operations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, name: str, suffix: str = "") -> None:
        filename = (
            f"{name}_{suffix}.{Config.FORMAT_TYPE}"
            if suffix
            else f"{name}.{Config.FORMAT_TYPE}"
        )
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()


class ResultSaver:
    """Handles saving statistical test results and interpretations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_result(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)


class SVGReportGenerator:
    """Generates SVG reports with proper styling"""

    @classmethod
    def _create_text_element(
        cls,
        x: float,
        y: float,
        text: str,
        font_size: int = 10,
        font_weight: str = "normal",
        fill: str = "black",
    ) -> str:
        return (
            f'<text x="{x}" y="{y}" font-family="Arial" font-size="{font_size}" '
            f'font-weight="{font_weight}" fill="{fill}">{text}</text>'
        )

    @classmethod
    def _create_table_row(
        cls, x: float, y: float, cells: list, header: bool = False
    ) -> tuple:
        # Ajustar anchos de columnas: primera columna más ancha para métricas
        first_col_width = 150  # Más espacio para métricas
        second_col_width = 60  # Más compacto para valores
        total_width = first_col_width + second_col_width
        cell_height = 20
        row_elements = []

        # Background for header
        if header:
            row_elements.append(
                f'<rect x="{x}" y="{y-15}" width="{total_width}" '
                f'height="{cell_height}" fill="#0069AA"/>'
            )
        else:
            row_elements.append(
                f'<line x1="{x}" y1="{y+5}" x2="{x+total_width}" '
                f'y2="{y+5}" stroke="#ddd" stroke-width="1"/>'
            )

        # Add text for each cell with adjusted positions
        widths = [first_col_width, second_col_width]
        x_positions = [x, x + first_col_width]

        for i, cell in enumerate(cells):
            text_color = "white" if header else "black"
            # Ajustar alineación: izquierda para métricas, derecha para valores
            text_x = x_positions[i] + (5 if i == 0 else widths[i] - 5)
            text_anchor = "start" if i == 0 else "end"

            row_elements.append(
                f'<text x="{text_x}" y="{y}" font-family="Arial" font-size="10" font-weight="normal" fill="{text_color}" text-anchor="{text_anchor}">{cell}</text>'
            )

        return "\n".join(row_elements), y + cell_height

    @classmethod
    def _series_to_table_rows(
        cls, series: pd.Series, start_x: float, start_y: float
    ) -> tuple:
        translations = {
            "Test Statistic": "Estadístico de prueba",
            "p-value": "p-value",
            "No. of Lags used": "N° de lags usados",
            "Number of observations used": "Observaciones totales",
            "Critical Value (1%)": "Valor crítico (1%)",
            "Critical Value (5%)": "Valor crítico (5%)",
            "Critical Value (10%)": "Valor crítico (10%)",
        }

        elements = []
        current_y = start_y

        # Header
        header_row, current_y = cls._create_table_row(
            start_x, current_y, ["Métrica", "Valor"], True
        )
        elements.append(header_row)

        # Data rows
        for k, v in series.items():
            k = translations.get(k, k)
            if k in ["N° de lags usados", "Observaciones totales"]:
                formatted_value = str(int(v))
            else:
                formatted_value = (
                    locale.format_string("%.2f", v) if isinstance(v, float) else str(v)
                )

            row, current_y = cls._create_table_row(
                start_x, current_y, [k, formatted_value]
            )
            elements.append(row)

        return "\n".join(elements), current_y

    @classmethod
    def generate_combined_report(
        cls,
        column: str,
        adf_output: pd.Series,
        is_stationary: bool,
        max_value: float,
        model_type: str,
    ) -> str:
        description, _ = TimeSeriesAnalyzer.get_column_description(column)
        formatted_max = locale.format_string("%.2f", max_value)

        # SVG dimensions (4x6 inches converted to pixels at 96 DPI)
        width = 250  # 4 inches * 96 DPI
        height = 500  # 6 inches * 96 DPI

        elements = []
        current_y = 30

        # Stationarity analysis title
        elements.append(
            cls._create_text_element(
                20, current_y, "Análisis de estacionariedad", font_weight="bold"
            )
        )
        current_y += 20

        # Stationarity table
        table_elements, current_y = cls._series_to_table_rows(adf_output, 20, current_y)
        elements.append(table_elements)
        current_y += 20

        # Stationarity result
        elements.append(
            cls._create_text_element(
                20,
                current_y,
                f"La serie temporal es {'estacionaria' if is_stationary else 'no estacionaria'}.",
                fill="black",
            )
        )
        current_y += 20

        # Forecast results title
        elements.append(
            cls._create_text_element(
                20, current_y, "Resultados del pronóstico", font_weight="bold"
            )
        )
        current_y += 20

        # Forecast table
        forecast_rows, _ = cls._create_table_row(
            20, current_y, ["Característica", "Valor"], True
        )
        elements.append(forecast_rows)
        current_y += 20

        model_row, current_y = cls._create_table_row(
            20, current_y, ["Tipo de modelo", model_type]
        )
        elements.append(model_row)

        max_row, _ = cls._create_table_row(
            20, current_y, ["Máximo valor pronosticado", formatted_max]
        )
        elements.append(max_row)

        # Combine all elements into final SVG
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            {"".join(elements)}
        </svg>"""


class TimeSeriesAnalyzer:
    """Analyzes time series data"""

    def __init__(self, output_dir: str, reports_dir: str):
        self.plot_saver = PlotSaver(output_dir)
        self.result_saver = ResultSaver(reports_dir)
        self.date_col = "time"

    @staticmethod
    def get_column_description(column: str) -> Tuple[str, str]:
        """Get description and unit for a column"""
        # Use the target information provided in the configuration
        if column == "diff_disp_total_abs":
            return ("Desplazamiento total absoluto", "cm")
        
        descriptions = {
            "diff_vert_abs": ("Desplazamiento vertical", "cm"),
            "mean_vel_rel": ("Velocidad media", "cm/día"),
            "inv_mean_vel_rel": ("Inversa de velocidad", "día/cm"),
        }
        return descriptions.get(column, (column, ""))

    def analyze(self, data: TimeSeriesData) -> None:
        """Analyze time series data by processing each time series individually"""
        df = data.series_df
        column = data.target_column
        self._analyze_single_series(df, column)

    def _analyze_single_series(self, df: pd.DataFrame, column: str) -> None:
        # Skip if not enough data
        n_points_min = 15
        if len(df) < n_points_min:
            print(
                f"Advertencia: Serie '{column}' tiene menos de {n_points_min} observaciones. Análisis omitido."
            )
        else:
            # Perform decomposition
            self._plot_decomposition(df, column)

            # Perform forecasting
            self._forecast_series(df, column)

    def _plot_decomposition(self, df: pd.DataFrame, column: str) -> None:
        ts = df.set_index(self.date_col)[column]
        description, unit = self.get_column_description(column)
        decomposition = seasonal_decompose(ts, model="additive", period=7)
        fig = decomposition.plot()
        fig.set_size_inches(8, 6)
        
        # Añadir título con descripción y unidad a cada subplot
        axes = fig.get_axes()
        title_text = f"{description} ({unit})" if unit else description
        
        # Títulos para cada componente
        component_titles = ["Observado", "Tendencia", "Estacionalidad", "Residual"]
        
        # Aplicar títulos a cada subplot
        for i, ax in enumerate(axes):
            if i < len(component_titles):
                ax.set_title(f"{component_titles[i]} - {title_text}")
                ax.set_ylabel(component_titles[i])
            ax.tick_params(axis="x", rotation=90)
            
        plt.tight_layout()
        self.plot_saver.save_plot(f"decomposition_{column}")

    def _is_stationary(self, ts: pd.Series) -> Tuple[bool, pd.Series]:
        adf_result = adfuller(ts.dropna(), autolag="AIC")
        output = pd.Series(
            adf_result[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "No. of Lags used",
                "Number of observations used",
            ],
        )
        for key, value in adf_result[4].items():
            output[f"Critical Value ({key})"] = value
        is_stationary = output["p-value"] < 0.05
        return is_stationary, output

    def _forecast_series(self, df: pd.DataFrame, column: str) -> None:
        ts = df.set_index(self.date_col)[column]

        # Perform stationarity test
        is_stationary, adf_output = self._is_stationary(ts)

        # Forecast based on stationarity and generate combined report
        if is_stationary:
            self._forecast_sarima(ts, column, adf_output, is_stationary)
        else:
            self._forecast_sarima(ts, column, adf_output, is_stationary)

    def _forecast_sarima(
        self, ts: pd.Series, column: str, adf_output: pd.Series, is_stationary: bool
    ) -> None:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        # Generate historical predictions
        historical_forecast = results.get_prediction(
            start=ts.index[0], end=ts.index[-1]
        )
        historical_pred_mean = historical_forecast.predicted_mean

        # Generate future dates for the forecast
        future_dates = pd.date_range(
            start=ts.index[-1], periods=Config.FORECAST_HORIZON + 1, freq="M"
        )[1:]
        forecast = results.get_forecast(steps=Config.FORECAST_HORIZON)
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()

        # Calculate the maximum value considering the prediction range
        max_forecast_value = max(pred_ci.iloc[:, 1].max(), pred_mean.max())

        # Generate and save combined HTML report
        combined_svg = SVGReportGenerator.generate_combined_report(
            column, adf_output, is_stationary, max_forecast_value, "SARIMA"
        )
        self.result_saver.save_result(f"analysis_{column}.svg", combined_svg)

        # Plot forecast
        plt.figure(figsize=(8, 6))
        plt.plot(ts.index, ts, color="blue", label="Datos históricos")
        plt.plot(
            ts.index,
            historical_pred_mean,
            color="gray",
            linestyle="--",
            label="Ajuste de modelo SARIMA",
        )
        plt.plot(
            future_dates,
            pred_mean,
            color="red",
            linestyle="--",
            label="Predicción SARIMA",
        )
        plt.fill_between(
            future_dates,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1],
            color="red",
            alpha=0.1,
            label="Intervalo de confianza 95%",
        )
        description, unit = self.get_column_description(
            column
        )  # Usar el método estático
        plt.title(f"Pronóstico SARIMA (p,d,q)=(1,1,1) - {description} ({unit})")
        plt.xlabel("Fecha de registro")
        plt.ylabel(f"{description} ({unit})")
        plt.legend()
        self.plot_saver.save_plot(f"forecast_{column}")


def process_pct_data(structure: str, num_item: int = 1) -> int:
    """Process PCT data for the given structure.

    Args:
        structure (str): Name of the structure to analyze
        num_item (int): Initial item number for PDF reports
        
    Returns:
        int: Number of PDFs created (to update the global counter)
    """
    # Set Matplotlib style
    # plt.style.use('ticks')

    # Set up directories
    output_dir = f"outputs/forecast/{structure}/plots"
    reports_dir = f"outputs/forecast/{structure}/reports"
    pdf_dir = f"outputs/forecast/{structure}/pdf_reports"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(output_dir, reports_dir)

    # Read sensor data
    sensor_df = pd.read_csv(
        "var/sample_client/sample_project/processed_data/operativity.csv", sep=";"
    )
    sensor_df = sensor_df[
        (sensor_df["sensor_type"] == "PCT") & (sensor_df["operativiy"] == True)
    ]

    # Process structure data
    df_structure = sensor_df.groupby("structure").get_group(structure).copy()
    df_structure.dropna(subset=["first_record", "last_record"], inplace=True)

    # Get structure display name from dictionary
    structure_display_name = structure_names[structure]
    
    initial_num_item = num_item  # Store initial value to calculate PDFs created
    appendix = "F"

    # Process each group in structure
    for group, df_group in df_structure.groupby("group"):
        # Get sensor information
        names = df_group["code"].tolist()

        # Process each sensor
        for code in names:
            csv_path = f"var/sample_client/sample_project/filtered_data/PCT/{structure}.{code}.csv"
            df_temp = pd.read_csv(csv_path, sep=";")
            num_min = 30
            if len(df_temp) >= num_min:
                print(
                    f"Advertencia: Serie '{code}' tiene menos de {num_min} observaciones. Análisis omitido."
                )

                data = TimeSeriesData(
                    csv_path,
                    date_col="time",
                    target_col="diff_disp_total_abs",
                )
                analyzer.analyze(data)

                # Generate PDF report
                decomp_files = sorted(
                    glob.glob(os.path.join(output_dir, "decomposition_*.svg"))
                )
                for decomp_file in decomp_files:
                    base_name = Path(decomp_file).stem.split("_", 1)[1]
                    forecast_file = os.path.join(
                        output_dir, f"forecast_{base_name}.svg"
                    )
                    report_file = os.path.join(reports_dir, f"analysis_{base_name}.svg")

                    if os.path.exists(forecast_file) and os.path.exists(report_file):
                        # Load SVGs
                        decomposition_draw = load_svg(decomp_file, 1)
                        forecast_draw = load_svg(forecast_file, 1)
                        report_draw = load_svg(report_file, 1)

                        # Create grid
                        grid = PlotMerger(fig_size=(6, 8))
                        grid.create_grid(
                            2, 2, row_ratios=[0.55, 0.45], col_ratios=[0.7, 0.3]
                        )

                        # Add objects
                        grid.add_object(decomposition_draw, (0, 0))
                        grid.add_object(report_draw, (0, 1))
                        grid.add_object(forecast_draw, (1, 0), span=(1, 2))

                        # Build final svg2rlg object
                        chart_svg = grid.build(color_border="white", cell_spacing=5)

                        # PDF generation parameters
                        params = {
                            "sample": "chart_portrait_a4_type_02",
                            "project_code": "1410.28.0054-0000",
                            "company_name": "Shahuindo SAC",
                            "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
                            "date": "09-04-25",
                            "revision": "B",
                            "elaborated_by": "J.A.",
                            "approved_by": "R.L.",
                            "doc_title": "SIG-AND",
                            "chart_title": f"Análisis estadístico de desplazamiento total absoluto - {code} / {structure_display_name}",
                            "theme_color": "#0069AA",
                            "theme_color_font": "white",
                            "num_item": f"{appendix}.{num_item}",
                        }

                        # Load logo
                        logo_path = "data/logo/logo_main_anddes.svg"
                        logo_cell = load_svg(logo_path, 0.65)

                        # Generate PDF
                        pdf_generator = ReportBuilder(
                            logo_cell=logo_cell, chart_cell=chart_svg, **params
                        )

                        output_filename = f"{appendix}_{num_item}_{code}.pdf"
                        pdf_generator.generate_pdf(
                            pdf_path=os.path.join(pdf_dir, output_filename)
                        )
                        num_item += 1
    
    # Return the number of PDFs created (difference between final and initial num_item)
    return num_item - initial_num_item

if __name__ == "__main__":
    
    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }
    
    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }
    
    # Inicializar contador global de items para los PDFs
    global_num_item = 1
    
    # Process all structures or a specific one
    for structure_code in structure_names.keys():
        print(f"Processing {structure_names[structure_code]}...")
        try:
            # Pasar el contador actual y recibir el número de PDFs creados
            pdfs_created = process_pct_data(structure_code, global_num_item)
            # Actualizar el contador global para la siguiente estructura
            global_num_item += pdfs_created
            print(f"Created {pdfs_created} PDFs for {structure_code}. Next item number: {global_num_item}")
        except Exception as e:
            print(f"Error processing {structure_code}: {e}")
    
    print("Processing complete! Total PDFs created: {}".format(global_num_item - 1))
=======
import os
import sys
import locale
import glob

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from darts import TimeSeries
from modules.reporter.plot_builder import PlotMerger
from modules.reporter.report_builder import ReportBuilder, load_svg
from libs.utils.df_helpers import read_df_on_time_from_csv
from libs.utils.config_plot import PlotConfig

PlotConfig.setup_matplotlib()

# Configuración global de formato numérico
locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")


@dataclass
class Config:
    """Configuration settings"""
    FORECAST_HORIZON: int = 3
    FORMAT_TYPE: str = "svg"


class TimeSeriesData:
    """Data handling class for a single time series"""

    def __init__(self, file_path: str, date_col: str = "time", target_col: str = None):
        """Initialize TimeSeriesData with a single target column.

        Args:
            file_path (str): Path to the CSV file
            date_col (str): Name of the date column
            target_col (str): Target column to analyze
            separator (str): CSV separator character
        """
        self.df = read_df_on_time_from_csv(file_path, set_index=False)
        self._prepare_data(date_col, target_col)

    def to_timeseries(self) -> TimeSeries:
        """Convert the prepared data to Darts TimeSeries format"""
        return TimeSeries.from_dataframe(
            self.series_df, time_col=self.date_col, value_cols=self.target_column
        )

    def _prepare_data(self, date_col: str, target_col: str = None) -> None:
        """Prepare time series data for analysis.

        Args:
            date_col (str): Name of the date column
            target_col (str): Name of the target column
        """
        if target_col is None:
            raise ValueError("Target column must be specified")

        if target_col not in self.df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in columns: {list(self.df.columns)}"
            )

        self.target_column = target_col
        self.date_col = date_col
        self.series_df = self.df[[date_col, target_col]].dropna()

        # Filtro de anomalías
        window_size = 7  # Ventana rolling de 7 días
        rolling_mean = (
            self.series_df[self.target_column]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        rolling_std = (
            self.series_df[self.target_column]
            .rolling(window=window_size, min_periods=1)
            .std()
        )

        # Umbrales dinámicos ±2 sigma
        upper_bound = rolling_mean + 2 * rolling_std
        lower_bound = rolling_mean - 2 * rolling_std

        # Identificar y reemplazar anomalías
        anomalies_mask = (self.series_df[self.target_column] > upper_bound) | (
            self.series_df[self.target_column] < lower_bound
        )
        self.series_df.loc[anomalies_mask, self.target_column] = np.nan

        # Interpolación lineal para valores faltantes
        self.series_df[self.target_column] = self.series_df[
            self.target_column
        ].interpolate(method="linear")


class PlotSaver:
    """Handles plot saving operations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, name: str, suffix: str = "") -> None:
        filename = (
            f"{name}_{suffix}.{Config.FORMAT_TYPE}"
            if suffix
            else f"{name}.{Config.FORMAT_TYPE}"
        )
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()


class ResultSaver:
    """Handles saving statistical test results and interpretations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_result(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)


class SVGReportGenerator:
    """Generates SVG reports with proper styling"""

    @classmethod
    def _create_text_element(
        cls,
        x: float,
        y: float,
        text: str,
        font_size: int = 10,
        font_weight: str = "normal",
        fill: str = "black",
    ) -> str:
        return (
            f'<text x="{x}" y="{y}" font-family="Arial" font-size="{font_size}" '
            f'font-weight="{font_weight}" fill="{fill}">{text}</text>'
        )

    @classmethod
    def _create_table_row(
        cls, x: float, y: float, cells: list, header: bool = False
    ) -> tuple:
        # Ajustar anchos de columnas: primera columna más ancha para métricas
        first_col_width = 150  # Más espacio para métricas
        second_col_width = 60  # Más compacto para valores
        total_width = first_col_width + second_col_width
        cell_height = 20
        row_elements = []

        # Background for header
        if header:
            row_elements.append(
                f'<rect x="{x}" y="{y-15}" width="{total_width}" '
                f'height="{cell_height}" fill="#0069AA"/>'
            )
        else:
            row_elements.append(
                f'<line x1="{x}" y1="{y+5}" x2="{x+total_width}" '
                f'y2="{y+5}" stroke="#ddd" stroke-width="1"/>'
            )

        # Add text for each cell with adjusted positions
        widths = [first_col_width, second_col_width]
        x_positions = [x, x + first_col_width]

        for i, cell in enumerate(cells):
            text_color = "white" if header else "black"
            # Ajustar alineación: izquierda para métricas, derecha para valores
            text_x = x_positions[i] + (5 if i == 0 else widths[i] - 5)
            text_anchor = "start" if i == 0 else "end"

            row_elements.append(
                f'<text x="{text_x}" y="{y}" font-family="Arial" font-size="10" font-weight="normal" fill="{text_color}" text-anchor="{text_anchor}">{cell}</text>'
            )

        return "\n".join(row_elements), y + cell_height

    @classmethod
    def _series_to_table_rows(
        cls, series: pd.Series, start_x: float, start_y: float
    ) -> tuple:
        translations = {
            "Test Statistic": "Estadístico de prueba",
            "p-value": "p-value",
            "No. of Lags used": "N° de lags usados",
            "Number of observations used": "Observaciones totales",
            "Critical Value (1%)": "Valor crítico (1%)",
            "Critical Value (5%)": "Valor crítico (5%)",
            "Critical Value (10%)": "Valor crítico (10%)",
        }

        elements = []
        current_y = start_y

        # Header
        header_row, current_y = cls._create_table_row(
            start_x, current_y, ["Métrica", "Valor"], True
        )
        elements.append(header_row)

        # Data rows
        for k, v in series.items():
            k = translations.get(k, k)
            if k in ["N° de lags usados", "Observaciones totales"]:
                formatted_value = str(int(v))
            else:
                formatted_value = (
                    locale.format_string("%.2f", v) if isinstance(v, float) else str(v)
                )

            row, current_y = cls._create_table_row(
                start_x, current_y, [k, formatted_value]
            )
            elements.append(row)

        return "\n".join(elements), current_y

    @classmethod
    def generate_combined_report(
        cls,
        column: str,
        adf_output: pd.Series,
        is_stationary: bool,
        max_value: float,
        model_type: str,
    ) -> str:
        description, _ = TimeSeriesAnalyzer.get_column_description(column)
        formatted_max = locale.format_string("%.2f", max_value)

        # SVG dimensions (4x6 inches converted to pixels at 96 DPI)
        width = 250  # 4 inches * 96 DPI
        height = 500  # 6 inches * 96 DPI

        elements = []
        current_y = 30

        # Stationarity analysis title
        elements.append(
            cls._create_text_element(
                20, current_y, "Análisis de estacionariedad", font_weight="bold"
            )
        )
        current_y += 20

        # Stationarity table
        table_elements, current_y = cls._series_to_table_rows(adf_output, 20, current_y)
        elements.append(table_elements)
        current_y += 20

        # Stationarity result
        elements.append(
            cls._create_text_element(
                20,
                current_y,
                f"La serie temporal es {'estacionaria' if is_stationary else 'no estacionaria'}.",
                fill="black",
            )
        )
        current_y += 20

        # Forecast results title
        elements.append(
            cls._create_text_element(
                20, current_y, "Resultados del pronóstico", font_weight="bold"
            )
        )
        current_y += 20

        # Forecast table
        forecast_rows, _ = cls._create_table_row(
            20, current_y, ["Característica", "Valor"], True
        )
        elements.append(forecast_rows)
        current_y += 20

        model_row, current_y = cls._create_table_row(
            20, current_y, ["Tipo de modelo", model_type]
        )
        elements.append(model_row)

        max_row, _ = cls._create_table_row(
            20, current_y, ["Máximo valor pronosticado", formatted_max]
        )
        elements.append(max_row)

        # Combine all elements into final SVG
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            {"".join(elements)}
        </svg>"""


class TimeSeriesAnalyzer:
    """Analyzes time series data"""

    def __init__(self, output_dir: str, reports_dir: str):
        self.plot_saver = PlotSaver(output_dir)
        self.result_saver = ResultSaver(reports_dir)
        self.date_col = "time"

    @staticmethod
    def get_column_description(column: str) -> Tuple[str, str]:
        """Get description and unit for a column"""
        # Use the target information provided in the configuration
        if column == "diff_disp_total_abs":
            return ("Desplazamiento total absoluto", "cm")
        
        descriptions = {
            "diff_vert_abs": ("Desplazamiento vertical", "cm"),
            "mean_vel_rel": ("Velocidad media", "cm/día"),
            "inv_mean_vel_rel": ("Inversa de velocidad", "día/cm"),
        }
        return descriptions.get(column, (column, ""))

    def analyze(self, data: TimeSeriesData) -> None:
        """Analyze time series data by processing each time series individually"""
        df = data.series_df
        column = data.target_column
        self._analyze_single_series(df, column)

    def _analyze_single_series(self, df: pd.DataFrame, column: str) -> None:
        # Skip if not enough data
        n_points_min = 15
        if len(df) < n_points_min:
            print(
                f"Advertencia: Serie '{column}' tiene menos de {n_points_min} observaciones. Análisis omitido."
            )
        else:
            # Perform decomposition
            self._plot_decomposition(df, column)

            # Perform forecasting
            self._forecast_series(df, column)

    def _plot_decomposition(self, df: pd.DataFrame, column: str) -> None:
        ts = df.set_index(self.date_col)[column]
        description, unit = self.get_column_description(column)
        decomposition = seasonal_decompose(ts, model="additive", period=7)
        fig = decomposition.plot()
        fig.set_size_inches(8, 6)
        
        # Añadir título con descripción y unidad a cada subplot
        axes = fig.get_axes()
        title_text = f"{description} ({unit})" if unit else description
        
        # Títulos para cada componente
        component_titles = ["Observado", "Tendencia", "Estacionalidad", "Residual"]
        
        # Aplicar títulos a cada subplot
        for i, ax in enumerate(axes):
            if i < len(component_titles):
                ax.set_title(f"{component_titles[i]} - {title_text}")
                ax.set_ylabel(component_titles[i])
            ax.tick_params(axis="x", rotation=90)
            
        plt.tight_layout()
        self.plot_saver.save_plot(f"decomposition_{column}")

    def _is_stationary(self, ts: pd.Series) -> Tuple[bool, pd.Series]:
        adf_result = adfuller(ts.dropna(), autolag="AIC")
        output = pd.Series(
            adf_result[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "No. of Lags used",
                "Number of observations used",
            ],
        )
        for key, value in adf_result[4].items():
            output[f"Critical Value ({key})"] = value
        is_stationary = output["p-value"] < 0.05
        return is_stationary, output

    def _forecast_series(self, df: pd.DataFrame, column: str) -> None:
        ts = df.set_index(self.date_col)[column]

        # Perform stationarity test
        is_stationary, adf_output = self._is_stationary(ts)

        # Forecast based on stationarity and generate combined report
        if is_stationary:
            self._forecast_sarima(ts, column, adf_output, is_stationary)
        else:
            self._forecast_sarima(ts, column, adf_output, is_stationary)

    def _forecast_sarima(
        self, ts: pd.Series, column: str, adf_output: pd.Series, is_stationary: bool
    ) -> None:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        # Generate historical predictions
        historical_forecast = results.get_prediction(
            start=ts.index[0], end=ts.index[-1]
        )
        historical_pred_mean = historical_forecast.predicted_mean

        # Generate future dates for the forecast
        future_dates = pd.date_range(
            start=ts.index[-1], periods=Config.FORECAST_HORIZON + 1, freq="M"
        )[1:]
        forecast = results.get_forecast(steps=Config.FORECAST_HORIZON)
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()

        # Calculate the maximum value considering the prediction range
        max_forecast_value = max(pred_ci.iloc[:, 1].max(), pred_mean.max())

        # Generate and save combined HTML report
        combined_svg = SVGReportGenerator.generate_combined_report(
            column, adf_output, is_stationary, max_forecast_value, "SARIMA"
        )
        self.result_saver.save_result(f"analysis_{column}.svg", combined_svg)

        # Plot forecast
        plt.figure(figsize=(8, 6))
        plt.plot(ts.index, ts, color="blue", label="Datos históricos")
        plt.plot(
            ts.index,
            historical_pred_mean,
            color="gray",
            linestyle="--",
            label="Ajuste de modelo SARIMA",
        )
        plt.plot(
            future_dates,
            pred_mean,
            color="red",
            linestyle="--",
            label="Predicción SARIMA",
        )
        plt.fill_between(
            future_dates,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1],
            color="red",
            alpha=0.1,
            label="Intervalo de confianza 95%",
        )
        description, unit = self.get_column_description(
            column
        )  # Usar el método estático
        plt.title(f"Pronóstico SARIMA (p,d,q)=(1,1,1) - {description} ({unit})")
        plt.xlabel("Fecha de registro")
        plt.ylabel(f"{description} ({unit})")
        plt.legend()
        self.plot_saver.save_plot(f"forecast_{column}")


def process_pct_data(structure: str, num_item: int = 1) -> int:
    """Process PCT data for the given structure.

    Args:
        structure (str): Name of the structure to analyze
        num_item (int): Initial item number for PDF reports
        
    Returns:
        int: Number of PDFs created (to update the global counter)
    """
    # Set Matplotlib style
    # plt.style.use('ticks')

    # Set up directories
    output_dir = f"outputs/forecast/{structure}/plots"
    reports_dir = f"outputs/forecast/{structure}/reports"
    pdf_dir = f"outputs/forecast/{structure}/pdf_reports"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(output_dir, reports_dir)

    # Read sensor data
    sensor_df = pd.read_csv(
        "var/sample_client/sample_project/processed_data/operativity.csv", sep=";"
    )
    sensor_df = sensor_df[
        (sensor_df["sensor_type"] == "PCT") & (sensor_df["operativiy"] == True)
    ]

    # Process structure data
    df_structure = sensor_df.groupby("structure").get_group(structure).copy()
    df_structure.dropna(subset=["first_record", "last_record"], inplace=True)

    # Get structure display name from dictionary
    structure_display_name = structure_names[structure]
    
    initial_num_item = num_item  # Store initial value to calculate PDFs created
    appendix = "F"

    # Process each group in structure
    for group, df_group in df_structure.groupby("group"):
        # Get sensor information
        names = df_group["code"].tolist()

        # Process each sensor
        for code in names:
            csv_path = f"var/sample_client/sample_project/filtered_data/PCT/{structure}.{code}.csv"
            df_temp = pd.read_csv(csv_path, sep=";")
            num_min = 30
            if len(df_temp) >= num_min:
                print(
                    f"Advertencia: Serie '{code}' tiene menos de {num_min} observaciones. Análisis omitido."
                )

                data = TimeSeriesData(
                    csv_path,
                    date_col="time",
                    target_col="diff_disp_total_abs",
                )
                analyzer.analyze(data)

                # Generate PDF report
                decomp_files = sorted(
                    glob.glob(os.path.join(output_dir, "decomposition_*.svg"))
                )
                for decomp_file in decomp_files:
                    base_name = Path(decomp_file).stem.split("_", 1)[1]
                    forecast_file = os.path.join(
                        output_dir, f"forecast_{base_name}.svg"
                    )
                    report_file = os.path.join(reports_dir, f"analysis_{base_name}.svg")

                    if os.path.exists(forecast_file) and os.path.exists(report_file):
                        # Load SVGs
                        decomposition_draw = load_svg(decomp_file, 1)
                        forecast_draw = load_svg(forecast_file, 1)
                        report_draw = load_svg(report_file, 1)

                        # Create grid
                        grid = PlotMerger(fig_size=(6, 8))
                        grid.create_grid(
                            2, 2, row_ratios=[0.55, 0.45], col_ratios=[0.7, 0.3]
                        )

                        # Add objects
                        grid.add_object(decomposition_draw, (0, 0))
                        grid.add_object(report_draw, (0, 1))
                        grid.add_object(forecast_draw, (1, 0), span=(1, 2))

                        # Build final svg2rlg object
                        chart_svg = grid.build(color_border="white", cell_spacing=5)

                        # PDF generation parameters
                        params = {
                            "sample": "chart_portrait_a4_type_02",
                            "project_code": "1410.28.0054-0000",
                            "company_name": "Shahuindo SAC",
                            "project_name": "Ingeniero de Registro (EoR), Monitoreo y Análisis Geotécnico de los Pads 1&2 y DMEs Choloque y Sur",
                            "date": "09-04-25",
                            "revision": "B",
                            "elaborated_by": "J.A.",
                            "approved_by": "R.L.",
                            "doc_title": "SIG-AND",
                            "chart_title": f"Análisis estadístico de desplazamiento total absoluto - {code} / {structure_display_name}",
                            "theme_color": "#0069AA",
                            "theme_color_font": "white",
                            "num_item": f"{appendix}.{num_item}",
                        }

                        # Load logo
                        logo_path = "data/logo/logo_main_anddes.svg"
                        logo_cell = load_svg(logo_path, 0.65)

                        # Generate PDF
                        pdf_generator = ReportBuilder(
                            logo_cell=logo_cell, chart_cell=chart_svg, **params
                        )

                        output_filename = f"{appendix}_{num_item}_{code}.pdf"
                        pdf_generator.generate_pdf(
                            pdf_path=os.path.join(pdf_dir, output_filename)
                        )
                        num_item += 1
    
    # Return the number of PDFs created (difference between final and initial num_item)
    return num_item - initial_num_item

if __name__ == "__main__":
    
    structure_names = {
        "PAD_1A": "Pad 1A",
        "PAD_2A": "Pad 2A",
        "PAD_2B_2C": "Pad 2B-2C",
        "DME_SUR": "DME Sur",
        "DME_CHO": "DME Choloque",
    }
    
    sensor_names = {
        "PCV": "Piezómetro de cuerda vibrante",
        "PTA": "Piezómetro de tubo abierto",
        "PCT": "Punto de control topográfico",
        "SACV": "Celda de asentamiento de cuerda vibrante",
        "CPCV": "Celda de presión de cuerda vibrante",
    }
    
    # Inicializar contador global de items para los PDFs
    global_num_item = 1
    
    # Process all structures or a specific one
    for structure_code in structure_names.keys():
        print(f"Processing {structure_names[structure_code]}...")
        try:
            # Pasar el contador actual y recibir el número de PDFs creados
            pdfs_created = process_pct_data(structure_code, global_num_item)
            # Actualizar el contador global para la siguiente estructura
            global_num_item += pdfs_created
            print(f"Created {pdfs_created} PDFs for {structure_code}. Next item number: {global_num_item}")
        except Exception as e:
            print(f"Error processing {structure_code}: {e}")
    
    print("Processing complete! Total PDFs created: {}".format(global_num_item - 1))
>>>>>>> 118aabc (update | Independizacion del locale del sistema operativo)
