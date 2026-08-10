"""Métricas de evaluación para regresión/forecasting (fase 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% de veces que el signo del cambio predicho coincide con el real."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    change_true = np.diff(y_true)
    change_pred = np.diff(y_pred)
    if len(change_true) == 0:
        return np.nan
    return float(np.mean(np.sign(change_true) == np.sign(change_pred)) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       horizon: int = 1) -> dict[str, float]:
    """Conjunto completo de métricas de regresión/forecasting."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "horizon": int(horizon),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "n": int(len(y_true)),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Tabla comparativa de métricas."""
    df = pd.DataFrame(rows)
    cols = ["horizon", "mae", "rmse", "r2", "smape", "mape", "directional_accuracy", "n"]
    return df[[c for c in cols if c in df.columns]]
