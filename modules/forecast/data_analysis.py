import numpy as np
import pandas as pd

from typing import List
from datetime import datetime

from pykalman import KalmanFilter

from scipy.signal import medfilt
from scipy.stats import linregress


class JumpDetector:
    """Detección de saltos en series temporales usando Filtro de Kalman.

    Atributos:
        n_iterations: Número de iteraciones para convergencia del filtro
        std_threshold: Umbral de detección en desviaciones estándar
        position_choice: Selección de posición del registro ('pre'=anterior al salto, 'post'=posterior al salto)
    """

    def __init__(
        self,
        n_iterations: int = 10,
        std_threshold: float = 15.0,
        position: str = "post",
    ):
        if position not in ["pre", "post"]:
            raise ValueError("position_choice debe ser 'pre' o 'post'")

        self.n_iterations = n_iterations
        self.std_threshold = std_threshold
        self.position = position

    def detect(self, df: pd.DataFrame, target_column: str) -> List[datetime]:
        """Detecta saltos en los datos temporales.

        Args:
            df: DataFrame con datos temporales
            target_column: Columna a analizar

        Returns:
            Lista de timestamps con saltos detectados, según posición seleccionada (pre/post)
        """
        if target_column not in df.columns:
            raise ValueError(f"Columna {target_column} no encontrada")

        series = df[target_column].values
        kf = KalmanFilter(initial_state_mean=series[0], n_dim_obs=1, n_dim_state=1)
        kf = kf.em(series, n_iter=self.n_iterations)
        filtered_state_means, _ = kf.filter(series)

        residuals = np.abs(series - filtered_state_means.ravel())
        threshold = np.mean(residuals) + self.std_threshold * np.std(residuals)
        jump_indices = np.where(residuals > threshold)[0]

        # Ajustar índices según selección de posición
        adjusted_indices = []
        for idx in jump_indices:
            if self.position == "pre" and idx > 0:
                adjusted_indices.append(idx - 1)
            elif self.position == "post" and idx < len(df) - 1:
                adjusted_indices.append(idx + 1)
            else:
                adjusted_indices.append(idx)

        return df["time"].iloc[adjusted_indices].tolist()


class AnomalyDetector:
    """Detección de anomalías enfocada en picos usando:
    - Filtro de mediana para suavizar tendencias
    - Umbral dinámico basado en percentiles
    - Puntuación combinada de magnitud y curvatura
    - Validación multi-ventana para evitar falsos positivos

    Atributos:
        window_size: Tamaño de ventana para el filtro
        min_threshold: Umbral mínimo absoluto basado en percentil
        sensitivity: Factor de sensibilidad (percentil superior)
        validation_windows: Lista de ventanas para validación
    """

    def __init__(
        self,
        window_size: int = 5,
        min_threshold: float = 95,
        sensitivity: float = 99,
        validation_windows: list = [3, 5, 7],
    ):
        self.window_size = window_size
        self.min_threshold = min_threshold  # Percentil mínimo (95 por defecto)
        self.sensitivity = sensitivity  # Percentil alto para picos (99)
        self.validation_windows = validation_windows

    def detect(self, df: pd.DataFrame, target_column: str) -> List[pd.Timestamp]:
        if target_column not in df.columns:
            raise ValueError(f"Columna {target_column} no encontrada")

        series = df[target_column].values

        # 🔹 Filtro de mediana para suavizar tendencias
        smoothed = medfilt(series, kernel_size=self.window_size)
        residuals = np.abs(series - smoothed)

        # 🔹 Umbral dinámico basado en percentiles (mejor que std)
        min_thresh_val = np.percentile(residuals, self.min_threshold)
        max_thresh_val = np.percentile(residuals, self.sensitivity)

        # Detección inicial de picos
        anomalies = np.where(residuals > max_thresh_val)[0]

        # 🔹 Validación multi-ventana para reducir falsos positivos
        validated_anomalies = [
            idx
            for idx in anomalies
            if self._validate_anomaly(series, idx, min_thresh_val)
        ]

        # 🔹 Sistema de scoring basado en curvatura + magnitud
        scores = self._calculate_anomaly_scores(
            residuals, validated_anomalies, max_thresh_val
        )
        final_indices = [
            idx for idx, score in zip(validated_anomalies, scores) if score > 0.75
        ]  # Filtro de confianza

        return df["time"].iloc[final_indices].tolist()

    def _validate_anomaly(
        self, series: np.ndarray, index: int, min_thresh: float
    ) -> bool:
        """Validación en múltiples ventanas usando media y mediana local."""
        for window in self.validation_windows:
            start = max(0, index - window)
            end = min(len(series), index + window + 1)

            local_median = np.median(series[start:end])
            local_std = np.std(series[start:end])

            # Si el valor no es suficientemente alto en comparación con la variabilidad local, descartarlo
            if (
                abs(series[index] - local_median) < 2 * local_std
                or abs(series[index] - local_median) < min_thresh
            ):
                return False
        return True

    def _calculate_anomaly_scores(
        self, residuals: np.ndarray, indices: list, max_thresh: float
    ) -> np.ndarray:
        """Calcula scores combinando magnitud y curvatura de la señal."""
        scores = []
        for idx in indices:
            # Score por magnitud (valores extremos)
            mag_score = (residuals[idx] - np.min(residuals)) / (
                max_thresh - np.min(residuals)
            )

            # Score por curvatura (segunda derivada local)
            start = max(0, idx - 2)
            end = min(len(residuals), idx + 3)
            x = np.arange(start, end)
            y = residuals[start:end]

            if len(x) > 2:
                slope, _, _, _, _ = linregress(x, y)
                curve_score = np.abs(slope) * 0.5  # Aumento de la pendiente

                # Score combinado
                scores.append(min(1.0, 0.7 * mag_score + 0.3 * curve_score))
            else:
                scores.append(
                    mag_score
                )  # Si no hay suficiente contexto, usar solo magnitud

        return np.array(scores)


class DataHandler:
    """Procesamiento posterior de datos para manejo de saltos y anomalías."""

    @staticmethod
    def filter_jumps(
        jumps: List[datetime], anomalies: List[datetime]
    ) -> List[datetime]:
        """Filtra saltos que coinciden con anomalías.

        Args:
            jumps: Lista de saltos detectados
            anomalies: Lista de anomalías detectadas

        Returns:
            Lista de saltos filtrados
        """
        return [jump for jump in jumps if jump not in set(anomalies)]

    @staticmethod
    def update_baseline(df: pd.DataFrame, jumps: List[datetime]) -> pd.DataFrame:
        """Actualiza columna base_line con saltos válidos.

        Args:
            df: DataFrame original
            jumps: Saltos filtrados

        Returns:
            DataFrame actualizado
        """
        df_updated = df.copy()
        df_updated.loc[df_updated["time"].isin(jumps), "base_line"] = True
        return df_updated

    @staticmethod
    def smooth_trend(
        df: pd.DataFrame,
        target_column: str,
        jumps: List[datetime],
        window_size: int = 5,
    ) -> pd.DataFrame:
        """Suaviza tendencia entre saltos usando regresión lineal.

        Args:
            df: DataFrame con datos
            target_column: Columna a suavizar
            jumps: Saltos detectados
            window_size: Tamaño de ventana

        Returns:
            DataFrame con tendencia suavizada
        """
        smoothed_df = df.copy()
        jump_indices = df.index[df["time"].isin(jumps)].tolist()

        for i in range(len(jump_indices) - 1):
            start, end = jump_indices[i], jump_indices[i + 1]
            if end - start > window_size:
                x = np.arange(start, start + window_size)
                y = df[target_column].iloc[start : start + window_size]
                slope, intercept, *_ = linregress(x, y)

                for j in range(start + window_size, end):
                    smoothed_df.at[j, target_column] = slope * j + intercept

        return smoothed_df
