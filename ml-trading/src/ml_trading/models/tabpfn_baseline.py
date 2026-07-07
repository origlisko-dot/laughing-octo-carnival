"""TabPFN as a fast zero-shot tabular baseline and early feature-screening tool.

TabPFN is capped in rows/features, so we subsample; it is a research baseline,
never the production model. Requires the `tabpfn` extra.
"""

from __future__ import annotations

import numpy as np

try:
    from tabpfn import TabPFNClassifier
except ImportError:  # optional extra
    TabPFNClassifier = None

MAX_ROWS = 10_000


def tabpfn_baseline_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 7,
) -> float:
    """Zero-shot accuracy of TabPFN on a (subsampled) split."""
    if TabPFNClassifier is None:
        raise RuntimeError("tabpfn not installed; run: uv pip install 'ml-trading[tabpfn]'")
    rng = np.random.default_rng(seed)
    if len(y_train) > MAX_ROWS:
        keep = rng.choice(len(y_train), MAX_ROWS, replace=False)
        X_train, y_train = X_train[keep], y_train[keep]
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    return float((clf.predict(X_test) == y_test).mean())
