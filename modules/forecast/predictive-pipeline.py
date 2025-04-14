import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
from yellowbrick.regressor import prediction_error, residuals_plot
import shap

# 1. Preparación de Datos
def prepare_data(df):
    # Feature engineering temporal
    print(df.columns)
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.days
    df['inv_mean_vel_rel_x_time'] = df['inv_mean_vel_rel'] * df['time_numeric']
    
    # Crear lags para la feature clave
    for lag in [1, 2]:
        df[f'inv_mean_vel_rel_lag{lag}'] = df['inv_mean_vel_rel'].shift(lag)
    
    # Eliminar filas con NaN generadas por los lags
    df = df.dropna().reset_index(drop=True)
    
    # Selección final de features
    features = ['inv_mean_vel_rel', 'time_numeric', 'inv_mean_vel_rel_x_time',
                'inv_mean_vel_rel_lag1', 'inv_mean_vel_rel_lag2']
    
    X = df[features]
    y = df['diff_disp_total_abs']
    
    return X, y

# 2. División temporal de datos
def temporal_split(X, y):
    # Conservar orden temporal
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Definición de Modelos
def create_model(model_type, params=None):
    base_pipe = Pipeline([
        ('scaler', RobustScaler())
    ])
    
    models = {
        'ridge': Ridge(alpha=1.0),
        'gpr': GaussianProcessRegressor(n_restarts_optimizer=10),
        'lgbm': LGBMRegressor(max_depth=2, n_estimators=50)
    }
    
    if model_type not in models:
        raise ValueError("Modelo no soportado")
    
    return Pipeline([
        ('preprocessing', base_pipe),
        ('regressor', models[model_type].set_params(**params) if params else models[model_type])
    ])

# 4. Optimización con Optuna
def objective(trial, model_type, X_train, y_train):
    params = {
        'ridge': {'alpha': trial.suggest_float('alpha', 0.1, 10.0)},
        'gpr': {'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True)},
        'lgbm': {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 2, 8)
        }
    }
    
    model = create_model(model_type, params[model_type])
    cv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_cv_train, y_cv_train)
        preds = model.predict(X_cv_val)
        scores.append(mean_squared_error(y_cv_val, preds, squared=False))
    
    return np.mean(scores)

# 5. Entrenamiento y Evaluación
def train_and_evaluate(df):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = temporal_split(X, y)
    
    # Entrenar y optimizar modelos
    study = {}
    best_models = {}
    for model_type in ['ridge', 'gpr', 'lgbm']:
        study[model_type] = optuna.create_study(direction='minimize')
        study[model_type].optimize(lambda trial: objective(trial, model_type, X_train, y_train), n_trials=30)
        
        best_models[model_type] = create_model(model_type, study[model_type].best_params)
        best_models[model_type].fit(X_train, y_train)
    
    # Evaluar en test
    results = {}
    for name, model in best_models.items():
        preds = model.predict(X_test)
        results[name] = {
            'RMSE': np.round(mean_squared_error(y_test, preds, squared=False), 3),
            'MAE': np.round(mean_absolute_error(y_test, preds), 3),
            'R2': np.round(r2_score(y_test, preds), 3)
        }
    
    # Seleccionar champion
    champion = min(results, key=lambda x: results[x]['RMSE'])
    
    # Visualizaciones
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error de predicción
    prediction_error(best_models[champion], X_train, y_train, X_test, y_test, ax=ax[0, 0])
    
    # Importancia de características (solo para LGBM)
    if champion == 'lgbm':
        shap_values = shap.TreeExplainer(best_models[champion].named_steps['regressor']).shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type='bar', show=False, plot_size=(5,5))
        ax[0, 1].set_title('Importancia de Features (SHAP)')
    else:
        # Usar coeficientes para Ridge
        if champion == 'ridge':
            coefs = pd.Series(best_models[champion].named_steps['regressor'].coef_, index=X.columns)
            coefs.sort_values().plot(kind='barh', ax=ax[0, 1])
            ax[0, 1].set_title('Coeficientes del Modelo')
    
    # Residuales vs Tiempo
    residuals_plot(best_models[champion], X_train, y_train, X_test, y_test, ax=ax[1, 0])
    
    # Predicciones vs Real
    ax[1, 1].scatter(y_test, best_models[champion].predict(X_test))
    ax[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax[1, 1].set_xlabel('Real')
    ax[1, 1].set_ylabel('Predicho')
    
    plt.tight_layout()
    plt.show()
    
    # Print resultados clave
    print(f"Modelo Champion: {champion.upper()}")
    print("Parámetros Óptimos:")
    print(study[champion].best_params)
    print("\nMétricas en Test:")
    print(pd.DataFrame(results).T)
    
    return best_models[champion]

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    
    from libs.utils.df_helpers import read_df_on_time_from_csv

    
    # Cargar datos
    df = read_df_on_time_from_csv(
        r"C:\Users\Joel Efraín\Desktop\_workspace\geo_sentry\outputs\forecast\input\PCT\DME_CHO.CH-41.csv",
        set_index=False
    )

    champion_model = train_and_evaluate(df)