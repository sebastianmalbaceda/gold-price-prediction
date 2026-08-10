"""Pipeline de preprocesamiento entrenable (fase 8-9).

Preprocessing que aprende estadísticas se ajusta SOLO con train y se
aplica a validation/test con transform (nunca un nuevo fit).

Para este proyecto: las features son numéricas continuas (precios, tipos,
retornos). El único transformador con estado es RobustScaler (ajustado en
train). No hay imputación con estado porque la limpieza forward-fill ya
se hizo sobre datos pasados, sin futuro.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def make_preprocessor() -> Pipeline:
    """Escalado robusto (menos sensible a outliers que StandardScaler)."""
    return Pipeline([
        ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
    ])


def fit_preprocessor(X_train: np.ndarray):
    """Ajusta el preprocesador solo con train y lo devuelve."""
    pp = make_preprocessor()
    pp.fit(X_train)
    return pp
