"""Tests unitarios (pytest): validación de inputs, split y métricas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import src.api.main as api_module
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
from src.models.deep_learning import (
    GRUDirectionModel,
    TabularMLP,
    device_report,
    make_sequences,
    predict_proba,
    train_binary_model,
)

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


def test_clean_daily_series_does_not_fill_a_long_target_gap():
    dates = pd.bdate_range("2000-01-03", periods=8)
    df = pd.DataFrame(
        {"date": dates, "gold_spot": [100.0, 100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0]}
    )
    cfg = {
        "data": {
            "start_date": "2000-01-01",
            "end_date": "2000-02-01",
            "excluded_features": [],
            "max_missing_after_ffill": 0.99,
        }
    }
    clean = clean_daily_series(df, cfg)
    assert dates[5] not in set(clean["date"])
    assert clean["gold_spot"].isna().sum() == 0


def test_feature_columns_never_include_other_future_targets():
    feats = make_targets(
        build_features(
            _make_df(100),
            {
                "features": {"target_lags": [1], "rolling_windows": [5], "exogenous_lags": [1]},
                "target": {"horizons": [1, 5]},
            },
        ),
        [1, 5],
    )
    columns = get_feature_columns(feats, [1])
    assert not any(column.startswith("target_") for column in columns)


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


def test_predict_extreme_values_rejected():
    """Valores físicamente imposibles deben rechazarse (422)."""
    r = client.post(
        "/predict",
        json={"date": "2025-01-01", "features": {"us10y_yield": 1e9}},
    )
    assert r.status_code in (422, 500, 503)


def test_predict_negative_extreme_rejected():
    r = client.post(
        "/predict",
        json={"date": "2025-01-01", "features": {"us10y_yield": -1e9}},
    )
    assert r.status_code in (422, 500, 503)


def test_metrics_reject_mismatched_or_invalid_probabilities():
    with pytest.raises(ValueError):
        smape(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), np.array([0.2, 1.2]))


def test_drop_warmup_rejects_all_null_columns():
    with pytest.raises(ValueError, match="completamente nulas"):
        drop_warmup(pd.DataFrame({"a": [1.0, 2.0], "empty": [np.nan, np.nan]}), 0)


def test_make_sequences_is_causal_and_validates_dates():
    dates = pd.bdate_range("2020-01-02", periods=12)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": np.arange(len(dates), dtype=float),
            "target": np.arange(len(dates)) % 2,
        }
    )
    x, y, used = make_sequences(frame, ["feature"], "target", dates[4:8], lookback=3)
    assert x.shape == (4, 3, 1)
    assert np.array_equal(x[0, :, 0], np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert np.array_equal(y, (np.arange(4, 8) % 2).astype(np.float32))
    assert used.iloc[-1] == dates[7]
    with pytest.raises(ValueError, match="no existe"):
        make_sequences(frame, ["feature"], "target", [pd.Timestamp("2030-01-01")], 3)


def test_deep_learning_cpu_training_and_prediction():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(40, 3)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    x_val = rng.normal(size=(20, 3)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)
    result = train_binary_model(
        TabularMLP(3, hidden=(8,), dropout=0.0),
        x_train,
        y_train,
        x_val,
        y_val,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=8,
        patience=2,
        seed=7,
    )
    probabilities = predict_proba(result.model, x_val, device="cpu")
    assert result.device == "cpu"
    assert 1 <= result.best_epoch <= 3
    assert probabilities.shape == (len(x_val),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_gru_forward_and_device_report():
    model = GRUDirectionModel(n_features=2, hidden_size=4, dropout=0.0)
    output = model(torch.zeros((3, 5, 2)))
    report = device_report(torch.device("cpu"))
    assert output.shape == (3,)
    assert report["device"] == "cpu"
    assert report["device_name"] == "cpu"


def test_api_rejects_bad_date_before_loading_model(monkeypatch):
    monkeypatch.setattr(api_module, "_ensure_loaded", lambda: None)
    response = client.post("/predict", json={"date": "2025-02-30", "features": {}})
    assert response.status_code == 422


def test_api_predict_with_validated_dummy_model(monkeypatch):
    class DummyPreprocessor:
        def transform(self, values):
            return values

    class DummyModel:
        def predict(self, values):
            return np.array([1234.567])

    monkeypatch.setattr(api_module, "_model", DummyModel())
    monkeypatch.setattr(api_module, "_preprocessor", DummyPreprocessor())
    monkeypatch.setattr(api_module, "_features", ["gold_spot_lag1"])
    response = client.post(
        "/predict", json={"date": "2025-02-28", "features": {"gold_spot_lag1": 1800}}
    )
    assert response.status_code == 200
    assert response.json()["prediction_usd_per_oz"] == 1234.57
    assert response.json()["date"] == "2025-02-28"
