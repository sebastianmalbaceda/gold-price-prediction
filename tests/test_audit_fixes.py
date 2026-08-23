"""Pruebas de regresion de las correcciones de la auditoria integral.

Cada prueba de este modulo fija el comportamiento de un defecto concreto
detectado en la auditoria, de modo que una regresion futura falle de forma
inmediata y localizable. El nombre de cada prueba describe el defecto que
protege.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.config import _declares_paths, load_yaml
from src.data.load_data import clean_daily_series
from src.data.split import purge_label_overlap, temporal_split
from src.evaluation.metrics import (
    directional_accuracy,
    forecast_directional_accuracy,
)
from src.models.classifier import make_direction_targets
from src.utils import atomic_write, atomic_write_text, save_json, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Carga un script de ``scripts/`` como modulo aislado."""
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fuga de etiquetas entre particiones (purge_label_overlap)
# ---------------------------------------------------------------------------


def _split_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=900)
    return pd.DataFrame({"date": dates, "gold_spot": np.arange(len(dates), dtype=float)})


def _cfg_for(frame: pd.DataFrame) -> dict:
    return {
        "split": {
            "train": ["2018-01-01", "2019-12-31"],
            "val": ["2020-01-01", "2020-12-31"],
            "test": ["2021-01-01", "2021-12-31"],
        },
        "target": {"horizons": [1, 5, 21], "primary_horizon": 1},
    }


def test_purge_label_overlap_recorta_el_solape_de_etiquetas():
    frame = _split_frame()
    cfg = _cfg_for(frame)
    parts = temporal_split(frame, cfg)
    purged = purge_label_overlap(parts, horizon=21)

    # train y val pierden exactamente h filas por la cola; test no se toca.
    assert len(purged["train"]) == len(parts["train"]) - 21
    assert len(purged["val"]) == len(parts["val"]) - 21
    assert len(purged["test"]) == len(parts["test"])

    # La etiqueta a h dias de la ultima fila de train ya no cae dentro de val.
    last_train = purged["train"]["date"].max()
    first_val = purged["val"]["date"].min()
    label_date = parts["train"]["date"].iloc[-1]
    assert last_train < first_val
    assert last_train < label_date


def test_purge_label_overlap_no_muta_la_entrada():
    frame = _split_frame()
    parts = temporal_split(frame, _cfg_for(frame))
    original = {name: len(part) for name, part in parts.items()}
    purge_label_overlap(parts, horizon=5)
    assert {name: len(part) for name, part in parts.items()} == original


def test_purge_label_overlap_usa_el_horizonte_maximo_de_una_lista():
    frame = _split_frame()
    parts = temporal_split(frame, _cfg_for(frame))
    por_lista = purge_label_overlap(parts, horizon=[1, 5, 21])
    por_maximo = purge_label_overlap(parts, horizon=21)
    assert len(por_lista["train"]) == len(por_maximo["train"])


def test_purge_label_overlap_toma_el_horizonte_de_la_configuracion():
    frame = _split_frame()
    cfg = _cfg_for(frame)
    parts = temporal_split(frame, cfg)
    desde_cfg = purge_label_overlap(parts, cfg=cfg)
    assert len(desde_cfg["train"]) == len(parts["train"]) - 21


def test_purge_label_overlap_rechaza_entradas_invalidas():
    with pytest.raises(ValueError):
        purge_label_overlap({}, horizon=1)


# ---------------------------------------------------------------------------
# Metricas direccionales
# ---------------------------------------------------------------------------


def test_directional_accuracy_no_cuenta_los_empates_como_acierto():
    """Una serie plana devolvia 100 % porque ``sign(0) == sign(0)``.

    Ahora las variaciones nulas se excluyen del denominador, de modo que sin
    ningun movimiento real la metrica es indefinida (NaN) en lugar de perfecta.
    """
    plana = np.array([100.0, 100.0, 100.0, 100.0])
    assert np.isnan(directional_accuracy(plana, plana))


def test_directional_accuracy_excluye_los_empates_del_denominador():
    # Tres variaciones: +1 (acertada), 0 (empate, excluida), -1 (fallada).
    y_true = np.array([100.0, 101.0, 101.0, 100.0])
    y_pred = np.array([100.0, 102.0, 102.0, 103.0])
    # Solo cuentan 2 movimientos, 1 acierto -> 50 %, no 66.7 %.
    assert directional_accuracy(y_true, y_pred) == 50.0


def test_directional_accuracy_requiere_al_menos_dos_puntos():
    assert np.isnan(directional_accuracy(np.array([1.0]), np.array([1.0])))


def test_directional_accuracy_detecta_signo_correcto():
    y_true = np.array([100.0, 101.0, 102.0])
    y_pred = np.array([100.0, 101.5, 103.0])
    assert directional_accuracy(y_true, y_pred) == 100.0


def test_forecast_directional_accuracy_compara_contra_el_nivel_actual():
    """La DA de un forecast a h dias se mide frente al ultimo nivel conocido."""
    y_current = np.array([100.0, 100.0, 100.0, 100.0])
    y_true = np.array([105.0, 95.0, 105.0, 95.0])
    y_pred = np.array([101.0, 99.0, 99.0, 101.0])  # acierta las dos primeras
    assert forecast_directional_accuracy(y_current, y_true, y_pred) == 50.0


def test_forecast_directional_accuracy_difiere_de_la_version_consecutiva():
    """Fija la diferencia entre ambas definiciones para evitar confundirlas."""
    y_current = np.array([100.0, 110.0, 120.0])
    y_true = np.array([110.0, 120.0, 130.0])
    y_pred = np.array([109.0, 121.0, 119.0])
    consecutiva = directional_accuracy(y_true, y_pred)
    a_horizonte = forecast_directional_accuracy(y_current, y_true, y_pred)
    assert consecutiva != a_horizonte


def test_forecast_directional_accuracy_valida_longitudes_y_no_finitos():
    with pytest.raises(ValueError):
        forecast_directional_accuracy(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        forecast_directional_accuracy(
            np.array([np.nan, 1.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )


# ---------------------------------------------------------------------------
# Targets de direccion
# ---------------------------------------------------------------------------


def test_make_direction_targets_falla_en_vez_de_inventar_bajadas():
    """Antes ``NaN > x`` era False y el target nulo se etiquetaba como 0."""
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "gold_spot": [100.0, 101.0, 102.0, 103.0],
            "target_1": [101.0, 102.0, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="contiene 2 nulos"):
        make_direction_targets(frame, horizons=[1])


def test_make_direction_targets_rechaza_gold_spot_nulo():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "gold_spot": [100.0, np.nan],
            "target_1": [101.0, 102.0],
        }
    )
    with pytest.raises(ValueError, match="gold_spot contiene nulos"):
        make_direction_targets(frame, horizons=[1])


def test_make_direction_targets_trata_el_empate_como_no_sube():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "gold_spot": [100.0, 100.0],
            "target_1": [100.0, 100.0],
        }
    )
    out = make_direction_targets(frame, horizons=[1])
    assert set(out["dir_1"].unique()) == {0}


def test_make_direction_targets_exige_target_previo_y_horizontes_validos():
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=2), "gold_spot": [100.0, 101.0]}
    )
    with pytest.raises(ValueError, match="Falta target_1"):
        make_direction_targets(frame, horizons=[1])
    with pytest.raises(ValueError, match="enteros positivos"):
        make_direction_targets(frame, horizons=[0])


# ---------------------------------------------------------------------------
# Limpieza de series: filtro de cobertura real
# ---------------------------------------------------------------------------


def test_clean_daily_series_descarta_columnas_sin_cobertura_real():
    """El filtro de cobertura debe medirse ANTES del ffill, no despues."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    frame = pd.DataFrame(
        {
            "date": dates,
            "gold_spot": np.linspace(1500, 1900, len(dates)),
            "densa": np.linspace(1.0, 2.0, len(dates)),
        }
    )
    # Una sola observacion al inicio: tras el ffill parece completa, pero su
    # cobertura real es del 0.25 %.
    escasa = np.full(len(dates), np.nan)
    escasa[0] = 42.0
    frame["escasa"] = escasa

    base_cfg = {
        "data": {
            "start_date": str(dates.min().date()),
            "end_date": str(dates.max().date()),
        }
    }

    # Comportamiento historico preservado: el filtro esta desactivado por
    # defecto, de modo que la correccion no altera resultados ya publicados.
    por_defecto = clean_daily_series(frame, base_cfg)
    assert "escasa" in por_defecto.columns

    # Con el umbral activado la columna se descarta pese a que tras el ffill no
    # tiene ningun nulo: su cobertura real es del 0.25 %.
    estricta = {"data": {**base_cfg["data"], "min_coverage_before_ffill": 0.5}}
    out = clean_daily_series(frame, estricta)
    assert "densa" in out.columns
    assert "escasa" not in out.columns


def test_clean_daily_series_valida_el_umbral_de_cobertura():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=10),
            "gold_spot": np.linspace(1500, 1600, 10),
            "x": np.linspace(1.0, 2.0, 10),
        }
    )
    cfg = {
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2020-01-20",
            "min_coverage_before_ffill": 1.5,
        }
    }
    with pytest.raises(ValueError, match="min_coverage_before_ffill"):
        clean_daily_series(frame, cfg)


# ---------------------------------------------------------------------------
# Escritura atomica unificada
# ---------------------------------------------------------------------------


def test_atomic_write_no_deja_temporales_ni_ficheros_parciales(tmp_path):
    destino = tmp_path / "sub" / "salida.json"
    atomic_write_text('{"a": 1}', destino)
    assert json.loads(destino.read_text(encoding="utf-8")) == {"a": 1}
    assert not list(destino.parent.glob(".*tmp"))


def test_atomic_write_preserva_el_fichero_previo_si_falla_la_escritura(tmp_path):
    destino = tmp_path / "salida.txt"
    destino.write_text("contenido-bueno", encoding="utf-8")

    def writer_que_falla(path: Path) -> None:
        path.write_text("basura-parcial", encoding="utf-8")
        raise RuntimeError("fallo simulado a mitad del volcado")

    with pytest.raises(RuntimeError):
        atomic_write(destino, writer_que_falla)

    # El destino original sigue intacto y no queda ningun temporal huerfano.
    assert destino.read_text(encoding="utf-8") == "contenido-bueno"
    assert not list(destino.parent.glob(".*tmp"))


def test_atomic_write_usa_un_temporal_unico_por_proceso(tmp_path):
    """Un nombre temporal fijo hacia que dos procesos se pisasen el fichero."""
    nombres: list[str] = []
    atomic_write(tmp_path / "x.bin", lambda p: (nombres.append(p.name), p.write_bytes(b"1"))[1])
    assert nombres[0] != ".x.bin.tmp"
    assert nombres[0].startswith(".x.bin.")


def test_save_json_rechaza_rutas_con_subdirectorios():
    with pytest.raises(ValueError):
        save_json({"a": 1}, "../fuera.json")


def test_set_seed_es_reproducible_y_valida_el_tipo():
    set_seed(123)
    primero = np.random.rand(5)
    set_seed(123)
    assert np.allclose(primero, np.random.rand(5))
    with pytest.raises(TypeError):
        set_seed(1.5)
    with pytest.raises(TypeError):
        set_seed(True)


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------


def test_load_yaml_resuelve_rutas_por_contenido_no_por_nombre():
    """Antes la resolucion dependia de que el fichero se llamase config.yaml."""
    cfg = load_yaml("config.yaml")
    assert Path(cfg["data"]["raw_path"]).is_absolute()
    assert _declares_paths(cfg) is True
    assert _declares_paths({"tuning": {"n_trials": 50}}) is False


def test_load_yaml_confina_la_carga_a_configs():
    for nombre in ("../setup.py", "/etc/passwd", "sub/dir.yaml"):
        with pytest.raises(ValueError):
            load_yaml(nombre)


def test_get_params_no_resuelve_rutas_inexistentes():
    params = load_yaml("params.yaml")
    assert isinstance(params, dict)


# ---------------------------------------------------------------------------
# Correccion por contrastes multiples en monitor_drift
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_controla_los_falsos_positivos():
    monitor = _load_script("monitor_drift.py")
    # 83 contrastes con p-valores uniformes: sin correccion, varios caerian por
    # debajo de 0.01 por puro azar.
    rng = np.random.default_rng(7)
    p_uniformes = rng.uniform(0.005, 1.0, size=83)
    rechazos, ajustados = monitor.benjamini_hochberg(list(p_uniformes), 0.01)
    assert rechazos.sum() <= (p_uniformes < 0.01).sum()
    assert np.all(ajustados >= p_uniformes - 1e-12)
    assert np.all(ajustados <= 1.0)


def test_benjamini_hochberg_detecta_una_senal_clara():
    monitor = _load_script("monitor_drift.py")
    p = [1e-12, 1e-10, 0.4, 0.6, 0.9]
    rechazos, _ = monitor.benjamini_hochberg(p, 0.01)
    assert list(rechazos) == [True, True, False, False, False]


def test_benjamini_hochberg_es_monotono_y_admite_lista_vacia():
    monitor = _load_script("monitor_drift.py")
    rechazos, ajustados = monitor.benjamini_hochberg([], 0.05)
    assert rechazos.size == 0 and ajustados.size == 0
    p = [0.01, 0.02, 0.03, 0.04]
    _, ajustados = monitor.benjamini_hochberg(p, 0.05)
    assert np.all(np.diff(ajustados) >= -1e-12)


# ---------------------------------------------------------------------------
# Script de prediccion batch
# ---------------------------------------------------------------------------


def test_predict_valida_la_ruta_de_salida():
    predict = _load_script("predict.py")
    with pytest.raises(ValueError, match="debe terminar en .csv"):
        predict._resolve_output_path("data/processed/features.parquet")
    assert predict._resolve_output_path("reports/salida.csv").suffix == ".csv"


def test_predict_row_rechaza_filas_no_finitas_y_features_ausentes():
    predict = _load_script("predict.py")

    class Modelo:
        def predict(self, X):
            return np.array([float(np.sum(X))])

    class Pre:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    fila_ok = pd.Series({"a": 1.0, "b": 2.0})
    assert predict.predict_row(fila_ok, Modelo(), Pre(), ["a", "b"]) == 3.0

    with pytest.raises(ValueError, match="no finitas"):
        predict.predict_row(pd.Series({"a": np.nan, "b": 2.0}), Modelo(), Pre(), ["a", "b"])
    with pytest.raises(ValueError, match="Faltan features"):
        predict.predict_row(pd.Series({"a": 1.0}), Modelo(), Pre(), ["a", "b"])


def test_predict_rechaza_fechas_mal_formadas():
    predict = _load_script("predict.py")
    for mala in ("2025-9-12", "12-09-2025", "hoy", "2025-13-01"):
        with pytest.raises(ValueError):
            predict._parse_date(mala)
    assert predict._parse_date("2025-09-12") == pd.Timestamp("2025-09-12")


# ---------------------------------------------------------------------------
# Controles de seguridad de la API
# ---------------------------------------------------------------------------


client = TestClient(api_module.app)


def test_api_limita_el_numero_de_features():
    """El limite lo aplica Pydantic, antes de recorrer el diccionario."""
    payload = {
        "date": "2025-01-02",
        "features": {f"f{i}": 1.0 for i in range(api_module.MAX_FEATURES + 1)},
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_api_rechaza_cuerpos_demasiado_grandes():
    r = client.post(
        "/predict",
        content=b'{"date": "2025-01-02", "features": {}}',
        headers={
            "content-type": "application/json",
            "content-length": str(api_module.MAX_BODY_BYTES + 1),
        },
    )
    assert r.status_code == 413


def test_api_rechaza_content_length_invalido():
    r = client.post(
        "/predict",
        content=b'{"date": "2025-01-02", "features": {}}',
        headers={"content-type": "application/json", "content-length": "no-es-numero"},
    )
    assert r.status_code == 400


def test_api_exige_clave_cuando_esta_configurada(monkeypatch):
    monkeypatch.setattr(api_module, "_API_KEY", "clave-secreta")
    payload = {"date": "2025-01-02", "features": {"a": 1.0}}

    assert client.post("/predict", json=payload).status_code == 401
    assert (
        client.post("/predict", json=payload, headers={"X-API-Key": "incorrecta"}).status_code
        == 401
    )
    # Con la clave correcta ya no es 401: el fallo pasa a ser la ausencia de
    # artefactos de modelo, que es el comportamiento esperado en CI.
    r = client.post("/predict", json=payload, headers={"X-API-Key": "clave-secreta"})
    assert r.status_code != 401

    assert client.post("/predict_direction", json=payload).status_code == 401


def test_api_queda_abierta_sin_clave_configurada(monkeypatch):
    """Comportamiento historico preservado: sin GOLD_API_KEY no hay 401."""
    monkeypatch.setattr(api_module, "_API_KEY", "")
    r = client.post("/predict", json={"date": "2025-01-02", "features": {"a": 1.0}})
    assert r.status_code != 401


def test_api_health_no_requiere_clave(monkeypatch):
    monkeypatch.setattr(api_module, "_API_KEY", "clave-secreta")
    assert client.get("/health").status_code == 200


def test_api_devuelve_el_horizonte_configurado(monkeypatch):
    """El horizonte de la respuesta ya no esta codificado a 1."""
    api_module._primary_horizon.cache_clear()
    monkeypatch.setattr(api_module, "get_config", lambda: {"target": {"primary_horizon": 5}})
    assert api_module._primary_horizon() == 5
    api_module._primary_horizon.cache_clear()


def test_api_model_version_se_cachea(monkeypatch):
    """Antes se releia el YAML en cada peticion."""
    api_module._model_version.cache_clear()
    llamadas = {"n": 0}

    def contador():
        llamadas["n"] += 1
        return {"model": {"version": "9.9.9"}, "target": {"primary_horizon": 1}}

    monkeypatch.setattr(api_module, "get_config", contador)
    assert api_module._model_version() == "9.9.9"
    api_module._model_version()
    api_module._model_version()
    assert llamadas["n"] == 1
    api_module._model_version.cache_clear()


def test_api_model_version_degrada_sin_seccion_model(monkeypatch):
    """Una configuracion incompleta ya no provoca un HTTP 500 por KeyError."""
    api_module._model_version.cache_clear()
    monkeypatch.setattr(api_module, "get_config", lambda: {"target": {"primary_horizon": 1}})
    assert api_module._model_version() == "unknown"
    api_module._model_version.cache_clear()


def test_api_rechaza_fecha_invalida_con_422_exacto():
    """Antes el test aceptaba 422, 500 o 503 y no distinguia un fallo real."""
    r = client.post("/predict", json={"date": "no-es-fecha", "features": {"a": 1.0}})
    assert r.status_code == 422


def test_api_rechaza_claves_extra_con_422_exacto():
    r = client.post(
        "/predict",
        json={"date": "2025-01-02", "features": {"a": 1.0}, "sobrante": 1},
    )
    assert r.status_code == 422
