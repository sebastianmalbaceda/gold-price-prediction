"""Pipeline de preprocesamiento entrenable (fases 8-9)."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def make_preprocessor() -> Pipeline:
    """Escalado robusto, menos sensible a outliers que StandardScaler."""
    return Pipeline(
        [
            ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
        ]
    )


def fit_preprocessor(X_train: np.ndarray) -> Pipeline:
    """Ajusta el preprocesador exclusivamente con una matriz de train valida."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2 or X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError("X_train debe ser una matriz 2D no vacia")
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contiene NaN o infinitos")
    pp = make_preprocessor()
    pp.fit(X_train)
    return pp
