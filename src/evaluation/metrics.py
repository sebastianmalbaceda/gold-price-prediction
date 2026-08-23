"""Metricas de evaluacion para regresion, clasificacion y forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size == 0 or pred.size == 0:
        raise ValueError("Las metricas requieren al menos una observacion")
    if true.size != pred.size:
        raise ValueError("y_true e y_pred deben tener la misma longitud")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Las metricas no aceptan NaN ni infinitos")
    return true, pred


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true, y_pred = _numeric_pair(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.divide(
        np.abs(y_true - y_pred),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(np.mean(ratio) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Si todos los valores reales son cero, el MAPE no esta definido y se
    devuelve ``nan`` de forma explicita, sin generar una advertencia de NumPy.
    """
    y_true, y_pred = _numeric_pair(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Porcentaje de cambios consecutivos cuyo signo predicho coincide con el real.

    ADVERTENCIA metodologica: esta metrica compara ``diff(y_true)`` con
    ``diff(y_pred)``, es decir, la variacion entre dos observaciones
    consecutivas de la serie objetivo. Para forecasting a horizonte ``h`` la
    pregunta relevante es otra (sube o baja respecto al nivel conocido hoy);
    para eso debe usarse :func:`forecast_directional_accuracy`.

    Las observaciones con variacion real nula se excluyen del denominador: en
    caso contrario ``np.sign(0) == np.sign(0)`` las contabiliza como aciertos
    y una serie plana devolveria 100 % de forma espuria.
    """
    y_true, y_pred = _numeric_pair(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    change_true = np.diff(y_true)
    change_pred = np.diff(y_pred)
    moved = change_true != 0
    if not moved.any():
        return float("nan")
    hits = np.sign(change_true[moved]) == np.sign(change_pred[moved])
    return float(np.mean(hits) * 100)


def forecast_directional_accuracy(
    y_current: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Acierto direccional real de un forecast a horizonte ``h``.

    Compara el signo de ``y_true - y_current`` (movimiento observado desde el
    ultimo nivel conocido) con el de ``y_pred - y_current`` (movimiento
    pronosticado). Es la definicion coherente con ``make_direction_targets``,
    donde los empates se tratan como "no sube".

    Parameters
    ----------
    y_current:
        Nivel observable al emitir la prediccion (``gold_spot`` en ``t``).
    y_true:
        Nivel realizado en ``t + h`` (``target_h``).
    y_pred:
        Nivel pronosticado para ``t + h``.
    """
    current = np.asarray(y_current, dtype=float).reshape(-1)
    true, pred = _numeric_pair(y_true, y_pred)
    if current.size != true.size:
        raise ValueError("y_current debe tener la misma longitud que y_true")
    if not np.isfinite(current).all():
        raise ValueError("y_current no acepta NaN ni infinitos")
    return float(np.mean((true > current) == (pred > current)) * 100)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
    y_current: np.ndarray | None = None,
) -> dict[str, float]:
    """Conjunto completo de metricas de regresion/forecasting.

    Si se proporciona ``y_current`` (nivel conocido en ``t``), la clave
    ``directional_accuracy`` se calcula con
    :func:`forecast_directional_accuracy`, que es la definicion correcta para
    forecasting. Ademas se incluye ``directional_accuracy_definition`` para
    dejar constancia del criterio empleado. Sin ``y_current`` se conserva el
    comportamiento historico por compatibilidad hacia atras.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon debe ser un entero positivo")
    y_true, y_pred = _numeric_pair(y_true, y_pred)
    if len(y_true) < 2:
        raise ValueError("R2 requiere al menos dos observaciones")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if y_current is None:
        da = directional_accuracy(y_true, y_pred)
        da_definition = "consecutive_diff"
    else:
        da = forecast_directional_accuracy(y_current, y_true, y_pred)
        da_definition = "vs_current_level"
    return {
        "horizon": int(horizon),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": da,
        "directional_accuracy_definition": da_definition,
        "n": int(len(y_true)),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Tabla comparativa de metricas, conservando columnas adicionales."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [
        "horizon",
        "mae",
        "rmse",
        "r2",
        "smape",
        "mape",
        "directional_accuracy",
        "directional_accuracy_definition",
        "n",
    ]
    present = [c for c in metric_cols if c in df.columns]
    extra = [c for c in df.columns if c not in metric_cols]
    return df[extra + present]


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Metricas de clasificacion binaria para probabilidades de clase positiva."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.size == 0 or y_true.size != y_prob.size:
        raise ValueError("y_true e y_prob deben tener la misma longitud no vacia")
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob debe contener probabilidades finitas entre 0 y 1")
    if not np.isfinite(y_true.astype(float)).all() or not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true debe contener exclusivamente las clases 0 y 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1")

    y_true = y_true.astype(np.int8)
    y_pred = (y_prob >= threshold).astype(np.int8)
    has_both_classes = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")
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
