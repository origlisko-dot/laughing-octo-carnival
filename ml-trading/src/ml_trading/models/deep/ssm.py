"""State-space sequence classifier (S4D-style diagonal SSM, simplified).

For very long intraday sequences where quadratic attention is wasteful. This is
a pragmatic diagonal-SSM implementation; if the `mamba-ssm` CUDA package is
installed, swap its Mamba block into `SSMClassifier.block_factory` for the full
selective-scan architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml_trading.models.deep import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import nn
else:
    torch = None
    nn = None


@dataclass
class SSMConfig:
    seq_len: int = 512
    n_channels: int = 5
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 2
    dropout: float = 0.1
    n_classes: int = 3


if TORCH_AVAILABLE:

    class DiagonalSSMBlock(nn.Module):
        """y_t = C * x_t, x_t = A x_{t-1} + B u_t with diagonal (stable) A per feature."""

        def __init__(self, d_model: int, d_state: int) -> None:
            super().__init__()
            self.log_a = nn.Parameter(torch.rand(d_model, d_state) * -2.0 - 0.5)  # A in (0,1)
            self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
            self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
            self.D = nn.Parameter(torch.ones(d_model))
            self.norm = nn.LayerNorm(d_model)
            self.gate = nn.Linear(d_model, d_model)

        def forward(self, u: torch.Tensor) -> torch.Tensor:
            """u: (batch, L, d_model)."""
            b, L, d = u.shape
            a = torch.sigmoid(self.log_a)  # (d, s) in (0,1) => stable
            x = torch.zeros(b, d, a.shape[1], device=u.device, dtype=u.dtype)
            ys = []
            for t in range(L):
                x = a * x + self.B * u[:, t, :, None]
                ys.append((x * self.C).sum(-1) + self.D * u[:, t])
            y = torch.stack(ys, dim=1)
            return self.norm(u + y * torch.sigmoid(self.gate(u)))

    class SSMClassifier(nn.Module):
        block_factory = DiagonalSSMBlock

        def __init__(self, cfg: SSMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.proj = nn.Linear(cfg.n_channels, cfg.d_model)
            self.blocks = nn.ModuleList(
                [self.block_factory(cfg.d_model, cfg.d_state) for _ in range(cfg.n_layers)]
            )
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.d_model, cfg.n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, seq_len, n_channels) -> logits."""
            h = self.proj(x)
            for blk in self.blocks:
                h = self.drop(blk(h))
            return self.head(h[:, -1])  # last-state readout suits streaming inference

else:

    class SSMClassifier:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("torch not installed; run: uv pip install 'ml-trading[deep]'")
