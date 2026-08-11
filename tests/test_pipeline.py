"""Tests unitarios (pytest): validación de inputs, split y métricas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.load_data import clean_daily_series
from src.data.split import drop_warmup, temporal_split
from src.evaluation.metrics import (
    classification_metrics,
    directional_accuracy,
    mape,
    regression_metrics,
    smape,
)
from src.features.build_features import build_features, get_feature_columns, make_targets
from src.models.classifier import make_direction_targets

client = TestClient(app)


def _make_df(n: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=n)
    gold = 300 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({"date": dates, "gold_spot": gold, "dummy": np.linspace(0, 1, n)})


# --- Fase 7: split temporal ---
def test_temporal_split_strict_order():
    df = _make_df(800)
    cfg = {
        "split": {
            "train": ["2000-01-03", "2001-01-01"],
            "val": ["2001-01-02", "2002-01-01"],
            "test": ["2002-01-02", "2004-01-01"],
        }
    }
    parts = temporal_split(df, cfg)
    assert parts["train"]["date"].max() < parts["val"]["date"].min()
    assert parts["val"]["date"].max() < parts["test"]["date"].min()
    assert len(parts["train"]) + len(parts["val"]) + len(parts["test"]) == len(df)


def test_split_no_overlap():
    df = _make_df(800)
    cfg = {
        "split": {
            "train": ["2000-01-03", "2002-01-01"],
            "val": ["2002-01-02", "2003-01-01"],
            "test": ["2003-01-02", "2004-01-01"],
        }
    }
    parts = temporal_split(df, cfg)
    dates = [set(p["date"]) for p in parts.values()]
    assert dates[0].isdisjoint(dates[1])
    assert dates[1].isdisjoint(dates[2])


# --- Fase 7: drop_warmup ---
def test_drop_warmup_no_nan():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    assert len(drop_warmup(df, 0)) == 4  # sin NaN no recorta
    assert len(drop_warmup(df, 260)) == 0  # warmup grande vacía


def test_drop_warmup_with_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    out = drop_warmup(df, 0)
    assert out["a"].iloc[0] == 3.0  # elimina filas sin lag


def test_drop_warmup_empty():
    assert len(drop_warmup(pd.DataFrame(), 260)) == 0


# --- Fase 6: métricas ---
def test_smape_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert smape(y, y) == 0.0


def test_smape_bounds():
    y = np.array([100.0])
    p = np.array([300.0])
    # |100-300| / ((100+300)/2) = 200/200 = 1.0 -> 100%
    assert smape(y, p) == 100.0


def test_smape_zero_denominator():
    # ambos valores 0 -> contribución 0 (sin división por cero)
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


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


def test_directional_accuracy_perfect():
    y = np.array([100.0, 101.0, 102.0, 103.0])
    p = np.array([99.0, 100.0, 101.0, 102.0])
    assert directional_accuracy(y, p) == 100.0


def test_mape_zero_guard():
    y = np.array([0.0, 100.0])
    p = np.array([0.0, 110.0])
    assert mape(y, p) == 10.0


# --- Fase 9: feature engineering ---
def test_build_features_creates_lags_and_targets():
    df = _make_df(300)
    cfg = {
        "features": {
            "target_lags": [1, 2],
            "rolling_windows": [5],
            "exogenous_lags": [1],
            "calendar": True,
        },
        "target": {"horizons": [1, 5]},
    }
    feats = build_features(df, cfg, raw=None)
    feats = make_targets(feats, [1, 5])
    assert "gold_spot_lag1" in feats.columns
    assert "gold_ret_lag1" in feats.columns
    assert "target_1" in feats.columns
    assert "target_5" in feats.columns
    assert "year" in feats.columns


def test_make_targets_no_future_leakage():
    df = _make_df(300)
    cfg = {
        "features": {
            "target_lags": [1],
            "rolling_windows": [5],
            "exogenous_lags": [1],
            "calendar": True,
        },
        "target": {"horizons": [1]},
    }
    feats = build_features(df, cfg, raw=None)
    feats = make_targets(feats, [1])
    # target_1(t) == gold_spot(t+1)
    assert np.allclose(feats["target_1"].to_numpy()[:-1], feats["gold_spot"].to_numpy()[1:])


def test_get_feature_columns_excludes_targets():
    df = _make_df(100)
    cfg = {
        "features": {
            "target_lags": [1],
            "rolling_windows": [5],
            "exogenous_lags": [1],
            "calendar": True,
        },
        "target": {"horizons": [1]},
    }
    feats = build_features(df, cfg, raw=None)
    feats = make_targets(feats, [1])
    cols = get_feature_columns(feats, [1])
    assert "target_1" not in cols
    assert "date" not in cols
    assert "gold_spot" not in cols


# --- Fase 16b: clasificación de dirección ---
def test_make_direction_targets():
    df = _make_df(100)
    cfg = {
        "features": {
            "target_lags": [1],
            "rolling_windows": [5],
            "exogenous_lags": [1],
            "calendar": True,
        },
        "target": {"horizons": [1]},
    }
    feats = build_features(df, cfg, raw=None)
    feats = make_targets(feats, [1])
    out = make_direction_targets(feats, [1])
    assert "dir_1" in out.columns
    # dir_1 = 1 si target_1 > gold_spot
    expected = (feats["target_1"] > feats["gold_spot"]).astype(int)
    assert (out["dir_1"] == expected).all()


def test_classification_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert 0 <= m["brier"] <= 1


def test_classification_metrics_single_class():
    y = np.array([1, 1, 1])
    p = np.array([0.9, 0.8, 0.7])
    m = classification_metrics(y, p)
    assert np.isnan(m["auc"])  # AUC no definido con una sola clase


# --- Fase 2-3: limpieza ---
def test_clean_daily_series_removes_weekends():
    cfg = {
        "data": {
            "start_date": "2000-01-01",
            "end_date": "2000-02-01",
            "excluded_features": [],
            "max_missing_after_ffill": 0.99,
        }
    }
    df = _make_df(60)
    # añadir un fin de semana artificial
    wk = pd.DataFrame({"date": [pd.Timestamp("2000-01-08")], "gold_spot": [np.nan], "dummy": [0.5]})
    df = pd.concat([df, wk]).sort_values("date")
    clean = clean_daily_series(df, cfg)
    assert (clean["date"].dt.dayofweek < 5).all()
    assert clean["gold_spot"].notna().all()


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


def test_predict_extra_keys_rejected():
    r = client.post("/predict", json={"date": "2025-01-01", "features": {"a": 1.0, "extra": 2.0}})
    assert r.status_code in (422, 500, 503)


def test_predict_direction_invalid_schema():
    r = client.post("/predict_direction", json={"date": "2025-01-01", "features": {}})
    assert r.status_code in (422, 500, 503)


def test_predict_direction_invalid_date():
    r = client.post("/predict_direction", json={"date": "no-es-fecha", "features": {"a": 1.0}})
    assert r.status_code in (422, 500, 503)
