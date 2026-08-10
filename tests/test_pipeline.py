"""Tests unitarios (pytest): validación de inputs, split y métricas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.split import temporal_split
from src.evaluation.metrics import regression_metrics, smape

client = TestClient(app)


def _make_df(n: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=n)
    gold = 300 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({"date": dates, "gold_spot": gold, "dummy": np.linspace(0, 1, n)})


# --- Fase 7: split temporal ---
def test_temporal_split_strict_order():
    df = _make_df(800)
    cfg = {"split": {"train": ["2000-01-03", "2001-01-01"],
                     "val": ["2001-01-02", "2002-01-01"],
                     "test": ["2002-01-02", "2004-01-01"]}}
    parts = temporal_split(df, cfg)
    assert parts["train"]["date"].max() < parts["val"]["date"].min()
    assert parts["val"]["date"].max() < parts["test"]["date"].min()
    assert len(parts["train"]) + len(parts["val"]) + len(parts["test"]) == len(df)


def test_split_no_overlap():
    df = _make_df(800)
    cfg = {"split": {"train": ["2000-01-03", "2002-01-01"],
                     "val": ["2002-01-02", "2003-01-01"],
                     "test": ["2003-01-02", "2004-01-01"]}}
    parts = temporal_split(df, cfg)
    dates = [set(p["date"]) for p in parts.values()]
    assert dates[0].isdisjoint(dates[1])
    assert dates[1].isdisjoint(dates[2])


# --- Fase 6: métricas ---
def test_smape_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert smape(y, y) == 0.0


def test_smape_bounds():
    y = np.array([100.0])
    p = np.array([300.0])
    # |100-300| / ((100+300)/2) = 200/200 = 1.0 -> 100%
    assert smape(y, p) == 100.0


def test_smape_asymmetric_cases():
    # errores de igual magnitud en valores distintos no dan el mismo sMAPE
    a = smape(np.array([100.0]), np.array([110.0]))
    b = smape(np.array([200.0]), np.array([210.0]))
    assert a > b
    assert 0 < a < 20


def test_regression_metrics_keys():
    y = np.array([100.0, 102.0, 101.0, 105.0])
    p = np.array([99.0, 103.0, 100.0, 106.0])
    m = regression_metrics(y, p, horizon=1)
    assert m["horizon"] == 1
    assert m["mae"] >= 0 and m["rmse"] >= 0
    assert -1 <= m["r2"] <= 1
    assert 0 <= m["directional_accuracy"] <= 100


# --- Fase 21: API ---
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_invalid_schema():
    # Sin modelo entrenado, la API debe fallar limpiamente (500 o 503),
    # nunca devolver 200 con predicción inventada
    r = client.post("/predict", json={"date": "2025-01-01", "features": {}})
    assert r.status_code in (422, 500, 503)


def test_predict_invalid_date():
    r = client.post("/predict", json={"date": "no-es-fecha", "features": {"a": 1.0}})
    assert r.status_code in (422, 500, 503)
