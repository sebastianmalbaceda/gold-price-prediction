"""Métricas de evaluación para regresión/forecasting (fase 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    sMAPE = 100 * mean( 2*|y - yhat| / (|y| + |yhat|) )

    Si el denominador es 0 (ambos valores 0), la contribución se define
    como 0 (error nulo) para evitar división por cero.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    num = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, num / denom, 0.0)
    return float(np.mean(ratio) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% de veces que el signo del cambio predicho coincide con el real.

    DA = 100 * mean( sign(y_t - y_{t-1}) == sign(yhat_t - yhat_{t-1}) )
    """
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
    """Tabla comparativa de métricas.

    Conserva columnas adicionales (p.ej. "model") y ordena las métricas
    estándar primero.
    """
    df = pd.DataFrame(rows)
    metric_cols = ["horizon", "mae", "rmse", "r2", "smape", "mape",
                   "directional_accuracy", "n"]
    present = [c for c in metric_cols if c in df.columns]
    extra = [c for c in df.columns if c not in metric_cols]
    return df[extra + present]


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> dict[str, float]:
    """Métricas de clasificación binaria para el predictor de dirección.

    y_true: etiquetas reales (0/1).
    y_prob: probabilidad predicha de la clase positiva (sube).
    threshold: umbral de decisión (por defecto 0.5).

    Devuelve AUC-ROC, PR-AUC, accuracy, precision, recall, F1 y Brier.
    En problemas de dirección de mercado, AUC es la métrica primaria
    honesta (el accuracy depende del umbral y del balance de clases).
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 brier_score_loss, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUC solo si hay ambas clases
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }
