"""Script CLI de prediccion batch (fase 21).

Uso:
    python scripts/predict.py --date 2025-09-12
    python scripts/predict.py --csv data/processed/test.parquet --out reports/batch_predictions.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import get_config, path_from_root  # noqa: E402
from src.data.load_data import clean_daily_series, load_raw  # noqa: E402
from src.data.split import drop_warmup  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.models.train_model import load_model_artifacts  # noqa: E402

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ALLOWED_OUT_SUFFIXES = {".csv"}


def _parse_date(value: str) -> pd.Timestamp:
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise ValueError("La fecha debe tener formato YYYY-MM-DD")
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"Fecha invalida: {value}") from exc


def _resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else path_from_root(value)


def _resolve_output_path(value: str) -> Path:
    """Valida la ruta de salida antes de escribir nada.

    Comprobar el sufijo y la existencia del directorio padre evita sobrescribir
    por accidente un artefacto del proyecto (por ejemplo un ``.parquet`` de
    ``data/processed``) al teclear mal ``--out``.
    """
    path = _resolve_input_path(value)
    if path.suffix.lower() not in _ALLOWED_OUT_SUFFIXES:
        raise ValueError(f"--out debe terminar en .csv, recibido: {path.name}")
    if path.is_dir():
        raise ValueError(f"--out apunta a un directorio: {path}")
    return path


def predict_row(features_row: pd.Series, model, pp, sel_cols, horizon: int = 1) -> float:
    """Predice el target a ``horizon`` dias a partir de una fila de features."""
    del horizon  # Se conserva en la API por compatibilidad con el CLI.
    missing = [column for column in sel_cols if column not in features_row.index]
    if missing:
        raise ValueError(f"Faltan features: {missing[:10]}...")
    values = pd.to_numeric(features_row[sel_cols], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("La fila contiene features no finitas")
    X = pp.transform(values.reshape(1, -1))
    prediction = float(model.predict(X)[0])
    if not np.isfinite(prediction):
        raise ValueError("El modelo devolvio una prediccion no finita")
    return prediction


def predict_for_date(date_str: str, model, pp, sel_cols, cfg: dict) -> float:
    """Reconstruye las features disponibles exactamente hasta una fecha.

    No llama a ``make_targets``: en inferencia el target futuro no existe y
    eliminar las ultimas h filas produciria una prediccion desplazada h dias.
    """
    target_date = _parse_date(date_str)
    raw = load_raw(cfg)
    last_date = raw["date"].max()
    if target_date > last_date:
        raise ValueError(
            f"La fecha {date_str} es posterior al ultimo dato disponible ({last_date.date()})"
        )

    raw_until_date = raw[raw["date"] <= target_date]
    if raw_until_date.empty:
        raise ValueError(f"No hay datos hasta la fecha {date_str}")
    clean = clean_daily_series(raw_until_date, cfg)
    if clean.empty or target_date not in set(clean["date"]):
        raise ValueError(f"La fecha {date_str} no es un dia habil con cotizacion disponible")

    feats = build_features(clean, cfg, raw_until_date)
    feats = drop_warmup(feats, warmup=260)
    if feats.empty or target_date not in set(feats["date"]):
        raise ValueError(f"Datos insuficientes para construir features en {date_str}")
    row = feats.loc[feats["date"] == target_date].iloc[-1]
    return predict_row(row, model, pp, sel_cols, cfg["target"]["primary_horizon"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Prediccion del precio del oro (USD/oz)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=str, help="Fecha YYYY-MM-DD para una prediccion puntual")
    group.add_argument("--csv", type=str, help="CSV/parquet con filas de features a predecir")
    ap.add_argument(
        "--out",
        type=str,
        default="reports/batch_predictions.csv",
        help="Ruta de salida para modo --csv",
    )
    args = ap.parse_args()

    cfg = get_config()
    try:
        model, pp, sel_cols = load_model_artifacts(cfg)
        horizon = cfg["target"]["primary_horizon"]
        if args.date:
            prediction = predict_for_date(args.date, model, pp, sel_cols, cfg)
            print(f"Prediccion gold_spot en {args.date} +{horizon}d: {prediction:.2f} USD/oz")
            return

        csv_path = _resolve_input_path(args.csv)
        if not csv_path.is_file():
            raise ValueError(f"No existe {csv_path}")
        df = (
            pd.read_parquet(csv_path)
            if csv_path.suffix.lower() == ".parquet"
            else pd.read_csv(csv_path)
        )
        if df.empty:
            raise ValueError("El fichero de entrada no contiene filas")

        # Validar la salida ANTES de gastar tiempo prediciendo.
        out_path = _resolve_output_path(args.out)

        # Sobrescribir columnas de entrada destruiria datos del usuario en
        # silencio, asi que se aborta si ya existen.
        clashing = [c for c in ("prediction", "prediction_error") if c in df.columns]
        if clashing:
            raise ValueError(
                f"El fichero de entrada ya contiene las columnas {clashing}; "
                "renombrelas para no perder informacion"
            )

        # Un fallo en una sola fila no debe abortar el lote entero: se anota el
        # motivo por fila y se resume al final. ``df.apply`` propagaba la primera
        # excepcion y descartaba todo el trabajo ya realizado.
        predictions: list[float | None] = []
        errors: list[str | None] = []
        for _, row in df.iterrows():
            try:
                predictions.append(predict_row(row, model, pp, sel_cols, horizon))
                errors.append(None)
            except (TypeError, ValueError, KeyError, IndexError) as row_exc:
                predictions.append(None)
                errors.append(str(row_exc))

        df = df.copy()
        df["prediction"] = predictions
        n_failed = sum(1 for e in errors if e is not None)
        if n_failed:
            df["prediction_error"] = errors

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(
            f"Predicciones guardadas en {out_path} "
            f"({len(df) - n_failed}/{len(df)} filas con prediccion valida)"
        )
        if n_failed:
            print(f"Advertencia: {n_failed} filas fallaron; ver columna 'prediction_error'")
            first = next(e for e in errors if e is not None)
            print(f"Primer error: {first}", file=sys.stderr)
            raise SystemExit(1)
    except (FileNotFoundError, OSError, TypeError, ValueError, IndexError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
