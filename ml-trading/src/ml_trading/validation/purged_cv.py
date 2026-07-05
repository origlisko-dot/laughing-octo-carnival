"""Purged & embargoed walk-forward cross-validation.

Random splits are forbidden on financial series: labels span multiple bars, so a
random split leaks test information into training. This splitter:

- walks forward in time (train always precedes test),
- *purges* training samples whose label window (`label_end_idx`) overlaps the
  test window,
- applies an *embargo* after each test window before training data may resume
  (only relevant for later folds when `expanding=False` is combined with gaps).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedWalkForward:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 0,
        min_train_size: int = 100,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size

    def split(
        self,
        n_samples: int,
        label_end_idx: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx). Samples are assumed time-ordered.

        label_end_idx[i] = last bar index that influences sample i's label; used for purging.
        """
        if label_end_idx is None:
            label_end_idx = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError("not enough samples for the requested number of splits")

        for k in range(1, self.n_splits + 1):
            test_start = k * fold_size
            test_end = min(test_start + fold_size, n_samples)
            test_idx = np.arange(test_start, test_end)

            train_end = test_start
            candidate = np.arange(0, train_end)
            # Purge: drop train samples whose label resolves inside (or after) the test window start.
            purged = candidate[label_end_idx[candidate] < test_start]
            # Embargo: additionally drop the last `embargo_bars` train samples adjacent to the test set.
            if self.embargo_bars > 0:
                purged = purged[purged < max(test_start - self.embargo_bars, 0)]
            if len(purged) < self.min_train_size:
                continue
            yield purged, test_idx
