"""iTransformer-style classifier: inverted attention over *variates*.

Each channel's full history is embedded as one token; self-attention runs
across channels (price, volume, RSI, leader returns, ...), capturing
cross-variate structure — well suited to multi-stock, multi-indicator inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml_trading.models.deep import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from torch import nn
else:
    nn = None


@dataclass
class ITransformerConfig:
    seq_len: int = 128
    n_channels: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_classes: int = 3


class ITransformerClassifier(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, cfg: ITransformerConfig) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not installed; run: uv pip install 'ml-trading[deep]'")
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.seq_len, cfg.d_model)  # whole series -> one token per variate
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        # Concatenate variate tokens (not mean-pool): signal concentrated in one
        # variate must survive into the head.
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model * cfg.n_channels),
            nn.Linear(cfg.d_model * cfg.n_channels, cfg.n_classes),
        )

    def forward(self, x):
        """x: (batch, seq_len, n_channels) -> logits (batch, n_classes)."""
        b = x.shape[0]
        tokens = self.embed(x.permute(0, 2, 1))  # (batch, n_channels, d_model)
        enc = self.encoder(tokens)
        return self.head(enc.reshape(b, -1))
