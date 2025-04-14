import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import locale
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Configuración global de formato numérico
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
pd.options.display.float_format = lambda x: locale.format_string('%.4f', x, grouping=True)

@dataclass
class Config:
    """Configuration settings"""

    structure = "dd_abra"
    OUTPUT_DIR: str = f"{structure}/plots"
    REPORTS_DIR: str = f"{structure}/reports"  # Nueva carpeta para reportes de texto
    DATA_FILE: str = f"{structure}.csv"
    FORECAST_HORIZON: int = 6
    ROLLING_WINDOW: int = 6
    FORMAT_TYPE : str = "svg"


class TimeSeriesData:
    """Data handling class"""

    def __init__(self, file_path: str, date_col: str = "date", separator: str = ";"):
        self.df = pd.read_csv(file_path, sep=separator)
        self.df[date_col] = pd.to_datetime(self.df[date_col], dayfirst=True, format='mixed')
        self._prepare_data(date_col)
        
    def _prepare_data(self, date_col: str) -> None:
        # Filtrar columnas excluyendo las que terminan en _VER
        self.series_columns = [
            col for col in self.df.columns 
            if col != date_col and not col.endswith('_VER')
        ]
        self.series_dfs = [
            self.df[[date_col, col]].dropna() for col in self.series_columns
        ]


class PlotSaver:
    """Handles plot saving operations"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, name: str, suffix: str = "") -> None:
        filename = f"{name}_{suffix}.{Config.FORMAT_TYPE}" if suffix else f"{name}.{Config.FORMAT_TYPE}"
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
    def _create_text_element(cls, x: float, y: float, text: str, font_size: int = 10, 
                           font_weight: str = "normal", fill: str = "black") -> str:
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{font_size}" ' \
               f'font-weight="{font_weight}" fill="{fill}">{text}</text>'

    @classmethod
    def _create_table_row(cls, x: float, y: float, cells: list, header: bool = False) -> tuple:
        # Ajustar anchos de columnas: primera columna más ancha para métricas
        first_col_width = 150  # Más espacio para métricas
        second_col_width = 60  # Más compacto para valores
        total_width = first_col_width + second_col_width
        cell_height = 20
        row_elements = []
        
        # Background for header
        if header:
            row_elements.append(f'<rect x="{x}" y="{y-15}" width="{total_width}" ' \
                              f'height="{cell_height}" fill="#0069AA"/>')
        else:
            row_elements.append(f'<line x1="{x}" y1="{y+5}" x2="{x+total_width}" ' \
                              f'y2="{y+5}" stroke="#ddd" stroke-width="1"/>')

        # Add text for each cell with adjusted positions
        widths = [first_col_width, second_col_width]
        x_positions = [x, x + first_col_width]
        
        for i, cell in enumerate(cells):
            text_color = "white" if header else "black"
            # Ajustar alineación: izquierda para métricas, derecha para valores
            text_x = x_positions[i] + (5 if i == 0 else widths[i] - 5)
            text_anchor = "start" if i == 0 else "end"
            
            row_elements.append(f'<text x="{text_x}" y="{y}" font-family="Arial" font-size="10" font-weight="normal" fill="{text_color}" text-anchor="{text_anchor}">{cell}</text>')
        
        return "\n".join(row_elements), y + cell_height

    @classmethod
    def _series_to_table_rows(cls, series: pd.Series, start_x: float, start_y: float) -> tuple:
        translations = {
            "Test Statistic": "Estadístico de prueba",
            "p-value": "p-value",
            "No. of Lags used": "N° de lags usados",
            "Number of observations used": "Observaciones totales",
            "Critical Value (1%)": "Valor crítico (1%)",
            "Critical Value (5%)": "Valor crítico (5%)",
            "Critical Value (10%)": "Valor crítico (10%)"
        }
        
        elements = []
        current_y = start_y
        
        # Header
        header_row, current_y = cls._create_table_row(start_x, current_y, 
                                                     ["Métrica", "Valor"], True)
        elements.append(header_row)
        
        # Data rows
        for k, v in series.items():
            k = translations.get(k, k)
            if k in ["N° de lags usados", "Observaciones totales"]:
                formatted_value = str(int(v))
            else:
                formatted_value = locale.format_string('%.4f', v) if isinstance(v, float) else str(v)
            
            row, current_y = cls._create_table_row(start_x, current_y, [k, formatted_value])
            elements.append(row)
            
        return "\n".join(elements), current_y

    @classmethod
    def generate_combined_report(cls, column: str, adf_output: pd.Series, is_stationary: bool, 
                               max_value: float, model_type: str) -> str:
        description, _ = TimeSeriesAnalyzer.get_column_description(column)
        formatted_max = locale.format_string('%.4f', max_value)
        
        # SVG dimensions (4x6 inches converted to pixels at 96 DPI)
        width = 250  # 4 inches * 96 DPI
        height = 500  # 6 inches * 96 DPI
        
        elements = []
        current_y = 30
        
        # Stationarity analysis title
        elements.append(cls._create_text_element(20, current_y, "Análisis de estacionariedad", 
                                               font_weight="bold"))
        current_y += 20
        
        # Stationarity table
        table_elements, current_y = cls._series_to_table_rows(adf_output, 20, current_y)
        elements.append(table_elements)
        current_y += 20
        
        # Stationarity result
        elements.append(cls._create_text_element(
            20, current_y, 
            f"La serie temporal es {'estacionaria' if is_stationary else 'no estacionaria'}.",
            fill="black"
        ))
        current_y += 20
        
        # Forecast results title
        elements.append(cls._create_text_element(20, current_y, "Resultados del pronóstico", 
                                               font_weight="bold"))
        current_y += 20
        
        # Forecast table
        forecast_rows, _ = cls._create_table_row(20, current_y, ["Característica", "Valor"], True)
        elements.append(forecast_rows)
        current_y += 20
        
        model_row, current_y = cls._create_table_row(20, current_y, ["Tipo de modelo", model_type])
        elements.append(model_row)
        
        max_row, _ = cls._create_table_row(
            20, current_y, 
            ["Máximo valor pronosticado", formatted_max]
        )
        elements.append(max_row)
        
        # Combine all elements into final SVG
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            {"".join(elements)}
        </svg>'''


class TimeSeriesAnalyzer:
    """Main analysis class"""

    @staticmethod
    def get_column_description(column: str) -> Tuple[str, str]:
        """Obtiene la descripción y unidad según el sufijo de la columna."""
        if column.endswith('_TOT'):
            return "Desplazamiento total absoluto", "Desplazamiento total (cm)"
        elif column.endswith('_HOR'):
            return "Desplazamiento horizontal absoluto", "Desplazamiento horizontal (cm)"
        return column, "Valor"

    def __init__(self, output_dir: str, reports_dir: str):
        self.plot_saver = PlotSaver(output_dir)
        self.result_saver = ResultSaver(reports_dir)

    def analyze(self, data: TimeSeriesData) -> None:
        for df in data.series_dfs:
            column = df.columns[1]
            self._analyze_single_series(df, column)

    def _analyze_single_series(self, df: pd.DataFrame, column: str) -> None:
        # Skip if not enough data (need at least 15 points for 2 complete cycles)
        if len(df) < 15:
            print(f"Advertencia: Serie '{column}' tiene menos de 15 observaciones. Análisis omitido.")
            return

        # Perform decomposition
        self._plot_decomposition(df, column)

        # Perform forecasting
        self._forecast_series(df, column)

    def _plot_decomposition(self, df: pd.DataFrame, column: str) -> None:
        ts = df.set_index("date")[column]
        decomposition = seasonal_decompose(ts, model="additive", period=6)

        fig = decomposition.plot()
        fig.set_size_inches(10, 10)
        
        # Rotar las etiquetas del eje X en cada subplot
        for ax in fig.get_axes():
            ax.tick_params(axis='x', rotation=90)
            
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
        ts = df.set_index("date")[column]
        
        # Perform stationarity test
        is_stationary, adf_output = self._is_stationary(ts)
        
        # Forecast based on stationarity and generate combined report
        if is_stationary:
            self._forecast_sarima(ts, column, adf_output, is_stationary)
        else:
            self._forecast_prophet(df, column, adf_output, is_stationary)

    def _forecast_sarima(self, ts: pd.Series, column: str, adf_output: pd.Series, 
                        is_stationary: bool) -> None:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

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
        plt.figure(figsize=(15, 10))
        plt.plot(ts.index, ts, color="blue", label="Datos históricos")
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
        description, y_label = self.get_column_description(column)  # Usar el método estático
        plt.title(f"Pronóstico SARIMA (p,d,q)=(1,1,1)\n{column} - {description}")
        plt.xlabel("Fecha")
        plt.ylabel(y_label)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        self.plot_saver.save_plot(f"forecast_{column}")

    def _forecast_prophet(self, df: pd.DataFrame, column: str, adf_output: pd.Series, 
                         is_stationary: bool) -> None:
        prophet_df = df.rename(columns={"date": "ds", column: "y"})
        model = Prophet(
            yearly_seasonality=True, interval_width=0.95, changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=Config.FORECAST_HORIZON, freq="M")
        forecast = model.predict(future)

        # Calculate the maximum value considering the prediction range
        max_forecast_value = max(forecast["yhat_upper"].max(), forecast["yhat"].max())

        # Generate and save combined HTML report
        combined_svg = SVGReportGenerator.generate_combined_report(
            column, adf_output, is_stationary, max_forecast_value, "Prophet"
        )
        self.result_saver.save_result(f"analysis_{column}.svg", combined_svg)

        # Plot forecast
        plt.figure(figsize=(15, 10))
        plt.plot(df["date"], df[column], color="blue", label="Datos históricos")
        forecast_dates = pd.to_datetime(forecast["ds"])
        plt.plot(
            forecast_dates,
            forecast["yhat"],
            color="red",
            linestyle="--",
            label="Predicción Prophet",
        )
        plt.fill_between(
            forecast_dates,
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="red",
            alpha=0.1,
            label="Intervalo de confianza 95%",
        )
        description, y_label = self.get_column_description(column)  # Usar el método estático
        plt.title(f"Pronóstico Prophet (Bayesiano)\n{column} - {description}")
        plt.xlabel("Fecha")
        plt.ylabel(y_label)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        self.plot_saver.save_plot(f"forecast_{column}")


def main():
    """Main execution function"""
    structures = ["tc_cashaucro", "dd_brunilda"]
    
    PlotConfig.setup_matplotlib()
    
    sns.set(style="ticks")

    # Initialize data and analyzer
    for structure in structures:
        data_file = f"{structure}.csv"
        data = TimeSeriesData(data_file)
        output_dir = f"{structure}/plots"
        reports_dir = f"{structure}/reports"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        analyzer = TimeSeriesAnalyzer(output_dir, reports_dir)
        analyzer.analyze(data)


if __name__ == "__main__":
    main()
