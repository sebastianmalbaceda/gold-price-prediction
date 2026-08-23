"""Clasificador de direccion del precio del oro (fase 16 extendida)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import get_config, path_from_root
from src.utils import atomic_write_joblib, atomic_write_text


def _validate_feature_list(feature_list: list[str]) -> None:
    if not feature_list or any(not isinstance(c, str) or not c for c in feature_list):
        raise ValueError("feature_list debe contener al menos una columna valida")
    if len(set(feature_list)) != len(feature_list):
        raise ValueError("feature_list contiene columnas duplicadas")


# Alias retrocompatibles: la implementacion unica vive en ``src.utils``.
_atomic_joblib = atomic_write_joblib
_atomic_text = atomic_write_text


def make_direction_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Crea targets binarios: 1 si ``gold(t+h) > gold(t)``."""
    if "gold_spot" not in df.columns:
        raise ValueError("Falta gold_spot en el dataframe")
    if not horizons or any(isinstance(h, bool) or int(h) != h or int(h) < 1 for h in horizons):
        raise ValueError("horizons debe contener enteros positivos")
    out = df.copy()
    if out["gold_spot"].isna().any():
        raise ValueError("gold_spot contiene nulos: no se puede etiquetar la direccion")
    for horizon in horizons:
        h = int(horizon)
        col = f"target_{h}"
        if col not in out.columns:
            raise ValueError(f"Falta {col}: ejecute primero make_targets (fase 5)")
        # Sin esta validacion un target nulo se convierte silenciosamente en 0
        # ("baja"), porque ``NaN > x`` es False. Eso inventa etiquetas negativas
        # en las ultimas filas de la serie o ante huecos de datos.
        if out[col].isna().any():
            n_null = int(out[col].isna().sum())
            raise ValueError(
                f"{col} contiene {n_null} nulos: eliminelos antes de etiquetar la "
                "direccion (make_targets ya descarta las filas sin futuro)"
            )
        # Los empates se etiquetan como 0 ("no sube"), criterio que deben
        # replicar las metricas direccionales.
        out[f"dir_{h}"] = (out[col] > out["gold_spot"]).astype(np.int8)
    return out


def fit_direction_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    horizon: int = 1,
    n_splits: int = 3,
    temporal_calibration: bool = True,
):
    """Entrena un RandomForest calibrado por validacion cruzada TEMPORAL.

    ``CalibratedClassifierCV(cv=3)`` usa ``StratifiedKFold``, que mezcla
    fechas: cada calibrador aprende su transformacion de probabilidad viendo
    observaciones posteriores a su propio pliegue de validacion. En series
    temporales eso es fuga. Por defecto se usa
    ``TimeSeriesSplit(n_splits, gap=horizon)``, con un hueco igual al horizonte
    para que la etiqueta del pliegue de entrenamiento no alcance al de
    calibracion.

    ``temporal_calibration=False`` restaura el comportamiento anterior; solo es
    razonable con datos sin orden temporal.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).reshape(-1)
    if X_train.ndim != 2 or len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben ser compatibles")
    if not np.isfinite(X_train).all() or not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("Datos de entrenamiento invalidos para clasificacion")
    counts = np.bincount(y_train.astype(np.int8), minlength=2)
    if counts.min() < 3:
        raise ValueError("Cada clase necesita al menos 3 observaciones para calibrar")

    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=seed,
        n_jobs=-1,
    )
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon debe ser un entero positivo")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits debe ser un entero >= 2")
    if temporal_calibration:
        min_rows = (n_splits + 1) * (horizon + 1)
        if len(X_train) < min_rows:
            raise ValueError(
                f"Se necesitan al menos {min_rows} filas para calibrar con "
                f"TimeSeriesSplit(n_splits={n_splits}, gap={horizon}); "
                f"recibidas {len(X_train)}"
            )
        cv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    else:
        cv = n_splits
    clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)
    clf.fit(X_train, y_train.astype(np.int8))
    return clf


def save_classifier_artifacts(
    model,
    preprocessor,
    feature_list: list[str],
    metrics: dict | None = None,
    horizon: int = 1,
    cfg: dict | None = None,
) -> dict:
    """Guarda el clasificador, su preprocesador, features y metricas."""
    if model is None or preprocessor is None:
        raise ValueError("model y preprocessor son obligatorios")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon debe ser un entero positivo")
    _validate_feature_list(feature_list)
    cfg = cfg or get_config()
    m = cfg["model"]
    base = path_from_root(m["models_dir"])
    paths = {
        "classifier": base / "direction_classifier.joblib",
        "classifier_preprocessor": base / "direction_preprocessor.joblib",
        "classifier_features": base / "direction_feature_list.json",
        "classifier_metrics": base / "direction_metrics.json",
    }
    for p in set(paths.values()):
        p.parent.mkdir(parents=True, exist_ok=True)

    _atomic_joblib(model, paths["classifier"])
    _atomic_joblib(preprocessor, paths["classifier_preprocessor"])
    _atomic_text(
        json.dumps(feature_list, indent=2, ensure_ascii=False), paths["classifier_features"]
    )
    if metrics is not None:
        _atomic_text(
            json.dumps(metrics, indent=2, default=str, ensure_ascii=False),
            paths["classifier_metrics"],
        )
    print(f"[save] clasificador -> {paths['classifier']}")
    print(f"[save] preprocesador -> {paths['classifier_preprocessor']}")
    print(f"[save] features ({len(feature_list)}) -> {paths['classifier_features']}")
    return {k: str(v) for k, v in paths.items()}


def load_classifier_artifacts(cfg: dict | None = None) -> tuple:
    """Carga clasificador + preprocesador + features guardados.

    Los artefactos joblib deben ser locales y confiables porque su carga
    ejecuta deserializacion de Python.
    """
    cfg = cfg or get_config()
    base = path_from_root(cfg["model"]["models_dir"])
    model = joblib.load(base / "direction_classifier.joblib")
    preprocessor = joblib.load(base / "direction_preprocessor.joblib")
    features = json.loads((base / "direction_feature_list.json").read_text(encoding="utf-8"))
    _validate_feature_list(features)
    return model, preprocessor, features
