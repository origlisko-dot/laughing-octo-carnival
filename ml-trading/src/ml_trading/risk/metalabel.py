"""Meta-labeling (Lopez de Prado): a secondary model that, given a primary
signal fired, predicts whether acting on it will be profitable. Its probability
maps to a size multiplier in [0, 1] — confidence-weighted sizing that can only
shrink the deterministic base size.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


class MetaLabeler:
    def __init__(self, threshold: float = 0.5, **params) -> None:
        self.threshold = threshold
        defaults = dict(objective="binary", n_estimators=200, verbosity=-1)
        self._model = lgb.LGBMClassifier(**{**defaults, **params})

    def fit(self, X: np.ndarray, primary_signal: np.ndarray, outcome: np.ndarray) -> MetaLabeler:
        """Train on bars where the primary model fired (signal != 0).

        outcome: realized label of acting on the signal (+1 profitable, -1 not).
        """
        mask = primary_signal != 0
        meta_y = (outcome[mask] > 0).astype(int)  # {0, 1}
        self._model.fit(X[mask], meta_y)
        return self

    def success_probability(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict_proba(X))[:, 1]

    def size_multiplier(self, X: np.ndarray) -> np.ndarray:
        """0 below threshold; linear ramp threshold->1.0 above it."""
        p = self.success_probability(X)
        ramp = (p - self.threshold) / max(1.0 - self.threshold, 1e-9)
        return np.clip(ramp, 0.0, 1.0)
