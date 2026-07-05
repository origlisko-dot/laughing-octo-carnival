"""Minimal training loop + sequence dataset for the deep classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_trading.models.deep import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:
    torch = None
    DataLoader = None
    TensorDataset = None


def make_sequences(
    channels: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice (T, C) channel matrix into (N, seq_len, C) windows ending at each labeled bar.

    Returns (X, y, sample_idx) where sample_idx maps each window back to its bar index
    (needed to carry label_end_idx into purged CV).
    """
    T = channels.shape[0]
    xs, ys, idx = [], [], []
    for t in range(seq_len - 1, T):
        window = channels[t - seq_len + 1 : t + 1]
        if np.isnan(window).any():
            continue
        xs.append(window)
        ys.append(labels[t])
        idx.append(t)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), np.asarray(idx)


@dataclass
class TrainResult:
    train_loss: list[float]
    val_accuracy: float


def train_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | None = None,
) -> TrainResult:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch not installed; run: uv pip install 'ml-trading[deep]'")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    y_enc = torch.as_tensor(y_train + 1, dtype=torch.long)  # {-1,0,1} -> {0,1,2}
    ds = TensorDataset(torch.as_tensor(X_train), y_enc)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            count += len(yb)
        losses.append(total / max(count, 1))

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_val).to(dev))
        pred = logits.argmax(dim=1).cpu().numpy() - 1
    acc = float((pred == y_val).mean()) if len(y_val) else 0.0
    return TrainResult(train_loss=losses, val_accuracy=acc)
