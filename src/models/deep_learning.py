"""Modelos neuronales para la clasificacion de direccion.

Incluye una MLP tabular y una GRU causal para predecir si
``gold_spot(t+h) > gold_spot(t)``. Reutiliza las features, particiones y
preprocesador del flujo principal. Todas las estadisticas se ajustan sobre
train y el dispositivo se selecciona automaticamente (CUDA o CPU).
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import path_from_root
from src.utils import atomic_write, set_seed


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Devuelve CUDA si esta disponible y se solicita; si no, CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _coerce_device(device: torch.device | str | None) -> torch.device:
    selected = torch.device(device) if device is not None else get_device()
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("El dispositivo debe ser cpu o cuda")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicito CUDA, pero no hay un dispositivo CUDA disponible")
        if selected.index is not None and not 0 <= selected.index < torch.cuda.device_count():
            raise ValueError(f"Indice CUDA fuera de rango: {selected.index}")
    return selected


def device_report(device: torch.device | str | None = None) -> dict:
    """Informa del hardware y backend utilizado por deep learning."""
    selected = _coerce_device(device)
    is_cuda = selected.type == "cuda" and torch.cuda.is_available()
    index = selected.index if selected.index is not None else 0
    return {
        "device": str(selected),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(index) if is_cuda else "cpu",
        "torch_version": torch.__version__,
    }


def set_torch_seed(seed: int = 42) -> None:
    """Fija semillas Python, NumPy y PyTorch para reproducibilidad.

    Delega en :func:`src.utils.set_seed`, que ya cubre Python, NumPy y PyTorch
    (incluido cuDNN determinista). Antes ambas funciones mantenian copias
    independientes de la misma logica y podian divergir.
    """
    set_seed(seed)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_dates: Iterable[pd.Timestamp],
    lookback: int = 21,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Construye ventanas causales terminadas en cada fecha objetivo.

    Para un target en ``t`` se utilizan filas ``t-lookback+1 ... t``. Las
    fechas objetivo deben pertenecer al dataframe y contar con historia
    suficiente; no se descartan silenciosamente fechas invalidas.
    """
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 1:
        raise ValueError("lookback debe ser un entero positivo")
    if not feature_cols or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols debe ser una lista no vacia y sin duplicados")
    missing = [c for c in [*feature_cols, target_col, "date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para construir secuencias: {missing}")

    ordered = df.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    if ordered["date"].isna().any() or ordered["date"].duplicated().any():
        raise ValueError("date debe contener valores validos y unicos")
    ordered = ordered.sort_values("date").reset_index(drop=True)
    if not ordered["date"].is_monotonic_increasing:
        raise ValueError("No se pudo ordenar temporalmente el dataframe")

    try:
        X_all = ordered[feature_cols].to_numpy(dtype=np.float32)
        y_all = pd.to_numeric(ordered[target_col], errors="coerce").to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Las features y el target deben ser numericos") from exc
    if not np.isfinite(X_all).all() or not np.isfinite(y_all).all():
        raise ValueError("El dataframe contiene NaN o infinitos")
    if not set(np.unique(y_all)).issubset({0.0, 1.0}):
        raise ValueError("El target de secuencias debe contener solo 0 y 1")

    requested = pd.to_datetime(list(target_dates), errors="coerce")
    if len(requested) == 0 or requested.isna().any():
        raise ValueError("target_dates debe contener fechas validas y no estar vacio")
    if requested.duplicated().any() or not requested.is_monotonic_increasing:
        raise ValueError("target_dates debe estar ordenado y no contener duplicados")

    date_to_idx = {date: i for i, date in enumerate(ordered["date"])}
    sequences, targets, used_dates = [], [], []
    for date in requested:
        idx = date_to_idx.get(date)
        if idx is None:
            raise ValueError(f"La fecha objetivo no existe en el dataframe: {date.date()}")
        if idx < lookback - 1:
            raise ValueError(f"Historia insuficiente para construir la ventana de {date.date()}")
        sequences.append(X_all[idx - lookback + 1 : idx + 1])
        targets.append(y_all[idx])
        used_dates.append(date)

    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.Series(used_dates, name="date"),
    )


class TabularMLP(nn.Module):
    """MLP para comparar aprendizaje neuronal tabular con modelos clasicos."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.25):
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        if not hidden or any(
            not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in hidden
        ):
            raise ValueError("hidden debe contener capas con tamaño positivo")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GRUDirectionModel(nn.Module):
    """GRU causal para ventanas temporales de features."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size debe ser positivo")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers debe ser positivo")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict]
    best_epoch: int
    best_val_auc: float
    device: str


def _loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool
) -> DataLoader:
    if len(X) == 0:
        raise ValueError("No se puede crear un DataLoader vacio")
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def _predict_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device, non_blocking=device.type == "cuda"))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def _validate_training_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    if X_train.ndim not in (2, 3) or X_val.ndim != X_train.ndim:
        raise ValueError("X debe tener forma (n, features) o (n, pasos, features)")
    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("train y validation deben compartir la forma de features")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("X e y deben tener el mismo numero de filas")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("train y validation no pueden estar vacios")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("X contiene NaN o infinitos")
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ValueError("y contiene NaN o infinitos")
    if not set(np.unique(y_train)).issubset({0.0, 1.0}) or not set(np.unique(y_val)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("y debe contener exclusivamente 0 y 1")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("train y validation deben contener las dos clases")


def train_binary_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device | str | None = None,
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42,
) -> TrainingResult:
    """Entrena con early stopping basado exclusivamente en AUC de validation."""
    if not isinstance(model, nn.Module):
        raise ValueError("model debe ser una instancia de torch.nn.Module")
    set_torch_seed(seed)
    selected_device = _coerce_device(device)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
    _validate_training_arrays(X_train, y_train, X_val, y_val)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs debe ser un entero positivo")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size debe ser un entero positivo")
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("patience debe ser un entero positivo")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo y finito")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay debe ser no negativo y finito")

    model = model.to(selected_device)
    pin_memory = selected_device.type == "cuda"
    train_loader = _loader(X_train, y_train, batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = _loader(X_val, y_val, batch_size, shuffle=False, pin_memory=pin_memory)

    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight = torch.tensor(negatives / positives, device=selected_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(selected_device, non_blocking=pin_memory))
            loss = criterion(logits, yb.to(selected_device, non_blocking=pin_memory))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prob = _predict_prob(model, val_loader, selected_device)
        val_auc = float(roc_auc_score(y_val, val_prob))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc})
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No se obtuvo un estado valido del modelo")
    model.load_state_dict(best_state)
    return TrainingResult(model, history, best_epoch, best_auc, str(selected_device))


def predict_proba(
    model: nn.Module, X: np.ndarray, device: torch.device | str | None = None
) -> np.ndarray:
    """Predice probabilidades en CPU o CUDA."""
    if not isinstance(model, nn.Module):
        raise ValueError("model debe ser una instancia de torch.nn.Module")
    selected_device = _coerce_device(device)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim not in (2, 3) or X.shape[0] == 0:
        raise ValueError("X debe tener forma (n, features) o (n, pasos, features) y no estar vacio")
    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o infinitos")
    loader = _loader(
        X,
        np.zeros(len(X), dtype=np.float32),
        256,
        shuffle=False,
        pin_memory=selected_device.type == "cuda",
    )
    model = model.to(selected_device)
    return _predict_prob(model, loader, selected_device)


# Alias retrocompatible: la implementacion unica vive en ``src.utils``.
_atomic_write = atomic_write


def save_deep_artifacts(
    model: nn.Module,
    preprocessor,
    feature_list: list[str],
    model_name: str,
    metadata: dict,
) -> dict[str, str]:
    """Guarda pesos portables, preprocesador, features y metadatos."""
    if not isinstance(model, nn.Module) or preprocessor is None:
        raise ValueError("model debe ser nn.Module y preprocessor es obligatorio")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", model_name):
        raise ValueError("model_name contiene caracteres no permitidos")
    if not feature_list or len(set(feature_list)) != len(feature_list):
        raise ValueError("feature_list debe ser no vacia y no contener duplicados")
    if not isinstance(metadata, dict):
        raise TypeError("metadata debe ser un diccionario")

    base = path_from_root("models")
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "weights": base / f"{model_name}.pt",
        "preprocessor": base / f"{model_name}_preprocessor.joblib",
        "features": base / f"{model_name}_features.json",
        "metadata": base / f"{model_name}_metadata.json",
    }
    # Guardar siempre los pesos en CPU hace que el .pt sea portable a equipos
    # sin CUDA, aunque el entrenamiento se haya realizado en GPU.
    state_cpu = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    _atomic_write(paths["weights"], lambda p: torch.save(state_cpu, p))
    _atomic_write(paths["preprocessor"], lambda p: joblib.dump(preprocessor, p))
    _atomic_write(
        paths["features"],
        lambda p: p.write_text(
            json.dumps(feature_list, indent=2, ensure_ascii=False), encoding="utf-8"
        ),
    )
    _atomic_write(
        paths["metadata"],
        lambda p: p.write_text(
            json.dumps(metadata, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        ),
    )
    return {k: str(v) for k, v in paths.items()}
