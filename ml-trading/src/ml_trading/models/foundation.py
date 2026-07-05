"""HuggingFace time-series foundation models as zero-shot forecasters.

Two roles only (never direct trading decisions):
  1. an extra baseline every trained model must beat;
  2. forecast quantiles fed as features (with uncertainty width) to our models.

Requires the `foundation` extra (chronos-forecasting). TimesFM/Moirai can be
added behind the same interface.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from chronos import BaseChronosPipeline
except ImportError:  # optional extra
    torch = None
    BaseChronosPipeline = None


class ChronosForecaster:
    def __init__(self, model_name: str = "amazon/chronos-bolt-small", device: str | None = None) -> None:
        if BaseChronosPipeline is None:
            raise RuntimeError(
                "chronos-forecasting not installed; run: uv pip install 'ml-trading[foundation]'"
            )
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=dev)

    def forecast_quantiles(
        self,
        context: np.ndarray,
        horizon: int = 12,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Zero-shot quantile forecast for a 1-D price/return context."""
        q, _ = self.pipeline.predict_quantiles(
            context=torch.as_tensor(context, dtype=torch.float32),
            prediction_length=horizon,
            quantile_levels=list(quantiles),
        )
        arr = q[0].cpu().numpy()  # (horizon, n_quantiles)
        return {lvl: arr[:, i] for i, lvl in enumerate(quantiles)}

    def forecast_features(self, context: np.ndarray, horizon: int = 12) -> dict[str, float]:
        """Compact feature set from the forecast: expected move and uncertainty width."""
        q = self.forecast_quantiles(context, horizon)
        last = float(context[-1])
        med = float(q[0.5][-1])
        return {
            "fm_exp_move": (med - last) / last,
            "fm_uncertainty": float((q[0.9][-1] - q[0.1][-1]) / max(abs(last), 1e-12)),
        }
