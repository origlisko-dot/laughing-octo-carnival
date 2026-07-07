"""Microstructural order-imbalance guard.

Before any entry we approximate buy/sell pressure from the trade stream using
the tick rule (upticks = buyer-initiated, downticks = seller-initiated). A
strongly one-sided flow against our direction vetoes or shrinks the entry —
protection against stepping into a liquidity drop. A full VPIN upgrade needs
tick/order-book data (Databento) behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tick_rule_imbalance(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Signed volume imbalance in [-1, 1] over the window: (buyV - sellV)/totalV."""
    p = np.asarray(prices, dtype=float)
    v = np.asarray(volumes, dtype=float)
    if len(p) < 2 or v[1:].sum() <= 0:
        return 0.0
    direction = np.sign(np.diff(p))
    # zero ticks inherit the previous direction (standard tick rule)
    for i in range(1, len(direction)):
        if direction[i] == 0:
            direction[i] = direction[i - 1]
    signed = direction * v[1:]
    return float(signed.sum() / v[1:].sum())


@dataclass
class ImbalanceGuard:
    veto_threshold: float = 0.6  # |imbalance| against us above this: no trade
    shrink_threshold: float = 0.3  # above this: halve the size

    def size_multiplier(self, side: str, imbalance: float) -> float:
        """1.0 = trade full size, 0.0 = veto. Only flow *against* the trade matters."""
        against = -imbalance if side == "long" else imbalance
        if against >= self.veto_threshold:
            return 0.0
        if against >= self.shrink_threshold:
            return 0.5
        return 1.0
