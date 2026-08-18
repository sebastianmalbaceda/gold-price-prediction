"""Deep learning para la clasificacion de direccion.

Extension de la fase 16: MLP tabular y GRU temporal para predecir
si gold_spot(t+h) > gold_spot(t). Reutiliza features, split y preprocesador
existentes. Todas las estadisticas se ajustan sobre train.

El modulo selecciona automaticamente CUDA si esta disponible y usa CPU como
fallback. La funcion device_report() deja constancia del dispositivo usado.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import path_from_root


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Devuelve CUDA si esta disponible y se solicita; si no, CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device | None = None) -> dict:
    """Informa de hardware y backend utilizado por deep learning."""
    device = device or get_device()
    return {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_name": (
            torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
    }


def set_torch_seed(seed: int = 42) -> None:
    """Fija semillas Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinismo para que los experimentos sean comparables.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_dates: Iterable[pd.Timestamp],
    lookback: int = 21,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Construye ventanas causales terminadas en cada fecha objetivo.

    Para un target en t se utilizan filas t-lookback+1 ... t. Ninguna
    observacion posterior a t entra en X. Las fechas de target se reciben
    desde la particion correspondiente, por lo que train/val/test mantienen
    orden temporal y no se mezclan.
    """
    ordered = df.sort_values("date").reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(ordered["date"])}
    X_all = ordered[feature_cols].to_numpy(dtype=np.float32)
    y_all = ordered[target_col].to_numpy(dtype=np.float32)

    sequences, targets, used_dates = [], [], []
    for date in pd.to_datetime(list(target_dates)):
        idx = date_to_idx.get(date)
        if idx is None or idx < lookback - 1:
            continue
        window = X_all[idx - lookback + 1 : idx + 1]
        target = y_all[idx]
        if not np.isfinite(window).all() or not np.isfinite(target):
            continue
        sequences.append(window)
        targets.append(target)
        used_dates.append(date)

    if not sequences:
        raise ValueError("No se pudieron construir secuencias validas")
    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.Series(used_dates, name="date"),
    )


class TabularMLP(nn.Module):
    """MLP para comparar DL tabular contra los modelos clasicos."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.25):
        super().__init__()
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


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _predict_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_binary_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device | None = None,
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42,
) -> TrainingResult:
    """Entrena con early stopping basado exclusivamente en validation AUC."""
    set_torch_seed(seed)
    device = device or get_device()
    model = model.to(device)
    train_loader = _loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, batch_size, shuffle=False)

    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
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
            logits = model(xb.to(device, non_blocking=True))
            loss = criterion(logits, yb.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prob = _predict_prob(model, val_loader, device)
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
    return TrainingResult(model, history, best_epoch, best_auc, str(device))


def predict_proba(
    model: nn.Module, X: np.ndarray, device: torch.device | None = None
) -> np.ndarray:
    """Predice probabilidades en CPU o CUDA."""
    device = device or get_device()
    loader = _loader(X.astype(np.float32), np.zeros(len(X), dtype=np.float32), 256, False)
    model = model.to(device)
    return _predict_prob(model, loader, device)


def save_deep_artifacts(
    model: nn.Module,
    preprocessor,
    feature_list: list[str],
    model_name: str,
    metadata: dict,
) -> dict[str, str]:
    """Guarda pesos, preprocesador, features y metadata del modelo DL."""
    base = path_from_root("models")
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "weights": base / f"{model_name}.pt",
        "preprocessor": base / f"{model_name}_preprocessor.joblib",
        "features": base / f"{model_name}_features.json",
        "metadata": base / f"{model_name}_metadata.json",
    }
    torch.save(model.state_dict(), paths["weights"])
    joblib.dump(preprocessor, paths["preprocessor"])
    paths["features"].write_text(json.dumps(feature_list, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}
