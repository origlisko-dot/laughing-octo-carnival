import numpy as np
import polars as pl
import pytest

from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.features.pipeline import base_features, feature_columns
from ml_trading.labeling.triple_barrier import triple_barrier_labels
from ml_trading.models.regime import GaussianHMMRegime, regime_observations
from ml_trading.models.tabular import TabularModel, cv_score, tune_lightgbm
from ml_trading.validation.purged_cv import PurgedWalkForward


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return SyntheticProvider(seed=5).fetch("TEST", "5m", "2025-01-06", "2025-03-28")


@pytest.fixture(scope="module")
def labeled(bars):
    res = triple_barrier_labels(bars, profit_mult=2.0, stop_mult=1.0, max_holding=24)
    return bars, res


def test_triple_barrier_sane(labeled) -> None:
    bars, res = labeled
    assert set(np.unique(res.label)).issubset({-1, 0, 1})
    # all three outcomes should occur on months of data
    assert (res.label == 1).sum() > 10
    assert (res.label == -1).sum() > 10
    assert (res.label == 0).sum() > 10
    # label end index is always >= sample index and within horizon
    idx = np.arange(len(res.label))
    assert (res.label_end_idx >= idx).all()
    assert (res.label_end_idx - idx <= 24).all()
    # profit labels have positive realized return, stops negative
    assert (res.realized_return[res.label == 1] > 0).all()
    assert (res.realized_return[res.label == -1] < 0).all()


def test_purged_cv_no_leakage(labeled) -> None:
    bars, res = labeled
    n = len(res.label)
    cv = PurgedWalkForward(n_splits=4, embargo_bars=24)
    folds = list(cv.split(n, res.label_end_idx))
    assert len(folds) >= 3
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()  # walk-forward
        # purge: no train label may resolve at/after the test window start
        assert (res.label_end_idx[train_idx] < test_idx.min()).all()
        # embargo: gap between last train sample and test start
        assert test_idx.min() - train_idx.max() >= 24


def test_tabular_model_learns_signal() -> None:
    """On separable synthetic data the model must beat chance by a wide margin."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.standard_normal((n, 6))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0.5, 1, np.where(X[:, 0] < -0.5, -1, 0))
    m = TabularModel().fit(X[:2000], y[:2000])
    acc = (m.predict(X[2000:]) == y[2000:]).mean()
    assert acc > 0.8
    proba = m.predict_proba(X[2000:])
    assert proba.shape == (1000, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_cv_score_and_optuna_smoke(labeled) -> None:
    bars, res = labeled
    feats = base_features(bars, ffd_d=0.4)
    cols = [c for c in feature_columns(feats) if feats[c].dtype.is_numeric()]
    X = feats.select(cols).fill_nan(None).to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    X, y, end = X[ok], res.label[ok], res.label_end_idx[ok]

    cv = PurgedWalkForward(n_splits=3, embargo_bars=24)
    base = cv_score({}, X, y, end, cv)
    assert 0.0 <= base <= 1.0

    study = tune_lightgbm(X[:800], y[:800], end[:800], cv, n_trials=2)
    assert study.best_value >= 0.0


def test_regime_hmm_recovers_states() -> None:
    rng = np.random.default_rng(1)
    # two alternating vol regimes
    r1 = rng.standard_normal(500) * 0.001
    r2 = rng.standard_normal(500) * 0.02
    rets = np.concatenate([r1, r2, r1])
    obs = regime_observations(rets, vol_window=20)
    hmm = GaussianHMMRegime(n_states=2, n_iter=30).fit(obs)
    states = hmm.predict(obs)
    # middle (high-vol) segment must be dominated by a different state than the flanks
    a = np.bincount(states[100:400], minlength=2).argmax()
    b = np.bincount(states[520:900], minlength=2).argmax()
    assert a != b


def test_deep_models_require_torch_or_work() -> None:
    from ml_trading.models.deep import TORCH_AVAILABLE

    if not TORCH_AVAILABLE:
        from ml_trading.models.deep.ssm import SSMClassifier, SSMConfig

        with pytest.raises(RuntimeError):
            SSMClassifier(SSMConfig())
        return

    import torch

    from ml_trading.models.deep.itransformer import ITransformerClassifier, ITransformerConfig
    from ml_trading.models.deep.patchtst import PatchTSTClassifier, PatchTSTConfig
    from ml_trading.models.deep.ssm import SSMClassifier, SSMConfig
    from ml_trading.models.deep.trainer import make_sequences, train_classifier

    x = torch.randn(4, 128, 5)
    assert PatchTSTClassifier(PatchTSTConfig()).forward(x).shape == (4, 3)
    assert ITransformerClassifier(ITransformerConfig(n_channels=5)).forward(x).shape == (4, 3)
    assert SSMClassifier(SSMConfig(seq_len=128)).forward(x).shape == (4, 3)

    # tiny end-to-end training run: label depends on the window mean of channel 0
    rng = np.random.default_rng(2)
    T = 4000
    seq = 32
    chans = rng.standard_normal((T, 5)).astype(np.float32)
    win_mean = np.convolve(chans[:, 0], np.ones(seq) / seq, mode="full")[: T]
    thr = np.std(win_mean) * 0.5
    labels = np.where(win_mean > thr, 1, np.where(win_mean < -thr, -1, 0))
    X, y, idx = make_sequences(chans, labels, seq_len=seq)
    cut = int(len(y) * 0.8)
    model = ITransformerClassifier(ITransformerConfig(seq_len=32, n_channels=5, d_model=32))
    res = train_classifier(
        model, X[:cut], y[:cut], X[cut:], y[cut:], epochs=15, lr=1e-3, device="cpu"
    )
    assert res.val_accuracy > 0.6
    assert res.train_loss[-1] < res.train_loss[0]
