"""Signal fusion: combine the technical model and the event model into one
entry score. The fused score is a *candidate* — it always goes through the
RiskEngine before any order exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionConfig:
    w_technical: float = 0.7
    w_event: float = 0.3
    entry_threshold: float = 0.25  # |score| must exceed this to trade
    conflict_veto: bool = True  # opposite-sign strong signals cancel the trade


def technical_score(proba: np.ndarray) -> np.ndarray:
    """Map triple-barrier class probabilities [P(stop), P(timeout), P(profit)] to [-1, 1]."""
    return proba[:, 2] - proba[:, 0]


def fuse(
    tech_score: np.ndarray,
    event_signal: np.ndarray,
    cfg: FusionConfig,
) -> np.ndarray:
    """Return entry signals in {-1, 0, +1} and store the fused score's magnitude
    as a size hint via `fused_scores` (same shape, in [-1, 1])."""
    score = cfg.w_technical * tech_score + cfg.w_event * event_signal
    if cfg.conflict_veto:
        conflict = (np.sign(tech_score) * np.sign(event_signal) < 0) & (
            np.abs(event_signal) > 0.5
        )
        score = np.where(conflict, 0.0, score)
    signals = np.where(score > cfg.entry_threshold, 1, np.where(score < -cfg.entry_threshold, -1, 0))
    return signals.astype(np.int8)


def fused_scores(
    tech_score: np.ndarray,
    event_signal: np.ndarray,
    cfg: FusionConfig,
) -> np.ndarray:
    score = cfg.w_technical * tech_score + cfg.w_event * event_signal
    return np.clip(np.abs(score), 0.0, 1.0)
