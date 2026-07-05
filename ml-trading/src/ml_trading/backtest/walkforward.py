"""Walk-forward research loop: per fold, train the tabular model on purged past
data, emit signals on the untouched test window, and run the cost-aware
simulator there. This is the number that decides whether a model lives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_trading.backtest.costs import CostModel
from ml_trading.backtest.metrics import PerformanceMetrics, performance_metrics
from ml_trading.backtest.simulator import BacktestResult, run_backtest
from ml_trading.config import AppConfig
from ml_trading.features.indicators import atr
from ml_trading.features.pipeline import base_features, feature_columns
from ml_trading.labeling.triple_barrier import triple_barrier_labels
from ml_trading.models.tabular import TabularModel
from ml_trading.validation.purged_cv import PurgedWalkForward


@dataclass
class FoldOutcome:
    fold: int
    test_start: int
    test_end: int
    metrics: PerformanceMetrics
    result: BacktestResult = field(repr=False, default=None)


def walk_forward_backtest(
    bars: pl.DataFrame,
    cfg: AppConfig,
    signal_threshold: float = 0.6,
    model_params: dict | None = None,
) -> list[FoldOutcome]:
    """Single-ticker walk-forward: train -> signal -> simulate per fold."""
    feats = base_features(bars)
    labels = triple_barrier_labels(
        bars,
        profit_mult=cfg.labels.profit_atr_mult,
        stop_mult=cfg.labels.stop_atr_mult,
        max_holding=cfg.labels.max_holding_bars,
        atr_window=cfg.labels.atr_window,
    )
    cols = [c for c in feature_columns(feats) if feats[c].dtype.is_numeric()]
    X_full = feats.select(cols).fill_nan(None).to_numpy().astype(np.float64)
    atr_vals = bars.select(atr(cfg.labels.atr_window)).to_series().to_numpy()

    valid = ~np.isnan(X_full).any(axis=1)
    first_valid = int(np.argmax(valid))  # leading warm-up rows only

    cv = PurgedWalkForward(n_splits=cfg.cv.n_splits, embargo_bars=cfg.cv.embargo_bars)
    outcomes: list[FoldOutcome] = []
    n = len(labels.label)

    for fold, (train_idx, test_idx) in enumerate(cv.split(n, labels.label_end_idx)):
        train_idx = train_idx[train_idx >= first_valid]
        if len(train_idx) < 200:
            continue
        model = TabularModel(params=model_params or {}).fit(
            X_full[train_idx], labels.label[train_idx]
        )
        proba = model.predict_proba(X_full[test_idx])
        long_conf = proba[:, 2]
        short_conf = proba[:, 0]
        signals = np.zeros(n, dtype=np.int8)
        signals[test_idx] = np.where(
            long_conf > signal_threshold, 1, np.where(short_conf > signal_threshold, -1, 0)
        )

        t0, t1 = int(test_idx[0]), int(test_idx[-1]) + 1
        seg = bars[t0:t1]
        result = run_backtest(
            seg,
            signals[t0:t1],
            cfg.risk,
            CostModel(
                commission_per_share=cfg.backtest.commission_per_share,
                min_commission=cfg.backtest.min_commission,
                slippage_bps=cfg.backtest.slippage_bps,
            ),
            atr_vals[t0:t1],
            initial_equity=cfg.backtest.initial_equity,
            stop_atr_mult=cfg.labels.stop_atr_mult,
            target_atr_mult=cfg.labels.profit_atr_mult,
            max_holding_bars=cfg.labels.max_holding_bars,
        )
        outcomes.append(
            FoldOutcome(
                fold=fold, test_start=t0, test_end=t1,
                metrics=performance_metrics(result), result=result,
            )
        )
    return outcomes


def summarize(outcomes: list[FoldOutcome]) -> dict:
    if not outcomes:
        return {}
    keys = outcomes[0].metrics.as_dict().keys()
    return {k: float(np.mean([o.metrics.as_dict()[k] for o in outcomes])) for k in keys}
