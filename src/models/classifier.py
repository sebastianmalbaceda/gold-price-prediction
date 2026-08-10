"""Clasificador de dirección del precio del oro (fase 16 extendida).

Predice si el precio spot del oro SUBE (1) o BAJA (0) en t+h días hábiles:

    y_dir = 1  si  gold_spot(t+h) > gold_spot(t)
    y_dir = 0  en caso contrario

Reutiliza TODO el pipeline existente:
- Features ya construidas (data/processed/features.parquet)
- Lista de features seleccionadas (models/feature_list.json)
- Preprocesador RobustScaler ajustado solo con train
- Split temporal estricto y TimeSeriesSplit

La señal de dirección en mercados eficientes es débil (AUC ~0.53-0.56),
por lo que la métrica primaria honesta es AUC-ROC, no accuracy.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import get_config, path_from_root


def make_direction_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Crea targets binarios de dirección: 1 si gold(t+h) > gold(t).

    Se deriva de los targets de regresión ya existentes (target_h) sin
    información futura adicional: solo compara el nivel futuro con el actual.
    """
    out = df.copy()
    for h in horizons:
        col = f"target_{h}"
        if col not in out.columns:
            raise ValueError(f"Falta {col}: ejecute primero make_targets (fase 5)")
        out[f"dir_{h}"] = (out[col] > out["gold_spot"]).astype(np.int8)
    return out


def fit_direction_classifier(X_train: np.ndarray, y_train: np.ndarray,
                             seed: int = 42):
    """Entrena el clasificador de dirección (RandomForest, calibrado por CV)."""
    from sklearn.calibration import CalibratedClassifierCV

    base = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        max_features=0.5, random_state=seed, n_jobs=-1)
    # Calibración isotónica con CV interna (mejora la calidad de las
    # probabilidades, que es lo que se consume en producción)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)
    return clf


def save_classifier_artifacts(model, preprocessor, feature_list: list[str],
                              metrics: dict | None = None,
                              horizon: int = 1,
                              cfg: dict | None = None) -> dict:
    """Guarda el clasificador, su preprocesador y las features usadas."""
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

    joblib.dump(model, paths["classifier"])
    joblib.dump(preprocessor, paths["classifier_preprocessor"])
    with open(paths["classifier_features"], "w") as f:
        json.dump(feature_list, f, indent=2)
    if metrics is not None:
        with open(paths["classifier_metrics"], "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    print(f"[save] clasificador -> {paths['classifier']}")
    print(f"[save] preprocesador -> {paths['classifier_preprocessor']}")
    print(f"[save] features ({len(feature_list)}) -> {paths['classifier_features']}")
    return {k: str(v) for k, v in paths.items()}


def load_classifier_artifacts(cfg: dict | None = None) -> tuple:
    """Carga clasificador + preprocesador + features guardados."""
    cfg = cfg or get_config()
    base = path_from_root(cfg["model"]["models_dir"])
    model = joblib.load(base / "direction_classifier.joblib")
    preprocessor = joblib.load(base / "direction_preprocessor.joblib")
    with open(base / "direction_feature_list.json") as f:
        features = json.load(f)
    return model, preprocessor, features
