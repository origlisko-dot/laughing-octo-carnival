"""PatchTST-style classifier for candle sequences.

The input sequence (length L, C channels: OHLCV-derived series) is split into
overlapping patches; each patch is linearly embedded and a Transformer encoder
attends over patch tokens (channel-independent, as in the PatchTST paper).
Head: 3-way classification matching the triple-barrier labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml_trading.models.deep import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import nn
else:  # keep module importable without torch
    torch = None
    nn = None


@dataclass
class PatchTSTConfig:
    seq_len: int = 128
    n_channels: int = 5
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    n_classes: int = 3


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch not installed; run: uv pip install 'ml-trading[deep]'")


class PatchTSTClassifier(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, cfg: PatchTSTConfig) -> None:
        _require_torch()
        super().__init__()
        self.cfg = cfg
        n_patches = (cfg.seq_len - cfg.patch_len) // cfg.stride + 1

        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model * cfg.n_channels),
            nn.Linear(cfg.d_model * cfg.n_channels, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_channels) -> logits (batch, n_classes)."""
        b, L, c = x.shape
        cfg = self.cfg
        # channel-independent: fold channels into the batch dimension
        x = x.permute(0, 2, 1).reshape(b * c, L)
        patches = x.unfold(dimension=1, size=cfg.patch_len, step=cfg.stride)  # (b*c, P, patch_len)
        tok = self.embed(patches) + self.pos
        enc = self.encoder(tok)  # (b*c, P, d_model)
        pooled = enc.mean(dim=1).reshape(b, c * cfg.d_model)
        return self.head(pooled)
