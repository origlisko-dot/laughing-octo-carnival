import numpy as np
import polars as pl
import pytest

from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.data.store import resample
from ml_trading.features.candles import candle_features
from ml_trading.features.fracdiff import adf_stat, frac_diff_ffd, ffd_weights, min_ffd_order
from ml_trading.features.indicators import indicator_features
from ml_trading.features.leadlag import LeadLagPair, granger_f_stat, lead_lag_matrix, leader_features
from ml_trading.features.mtf import align_multi_timeframe
from ml_trading.features.pipeline import build_feature_frame, feature_columns


@pytest.fixture(scope="module")
def bars_1m() -> pl.DataFrame:
    return SyntheticProvider(seed=3).fetch("TEST", "1m", "2025-01-06", "2025-01-31")


def test_indicators_no_lookahead(bars_1m: pl.DataFrame) -> None:
    """Truncating the input must not change earlier feature values."""
    full = indicator_features(bars_1m)
    trunc = indicator_features(bars_1m.head(500))
    for col in ("rsi_14", "atr_14", "macd_hist", "vwap", "trend_strength"):
        a = full[col].head(500).to_numpy()
        b = trunc[col].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-10, err_msg=col)


def test_rsi_bounds(bars_1m: pl.DataFrame) -> None:
    rsi = indicator_features(bars_1m)["rsi_14"].drop_nulls()
    assert rsi.min() >= 0.0 and rsi.max() <= 100.0


def test_candles_bullish_engulfing() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2025, 1, 6, 14, 0),
                pl.datetime(2025, 1, 6, 14, 1),
                interval="1m",
                time_zone="UTC",
                eager=True,
            ),
            "open": [10.0, 9.0],
            "high": [10.2, 10.8],
            "low": [8.9, 8.9],
            "close": [9.2, 10.5],  # bear bar then bull bar engulfing it
            "volume": [1.0, 1.0],
        }
    )
    out = candle_features(df)
    assert out["cdl_engulfing"][1] == 1


def test_ffd_weights_and_memory() -> None:
    w = ffd_weights(0.45)
    assert w[0] == 1.0 and w[1] == pytest.approx(-0.45)
    rng = np.random.default_rng(0)
    price = np.cumsum(rng.standard_normal(3000) * 0.01) + 10
    d = min_ffd_order(price)
    assert 0.0 < d <= 1.0
    series = frac_diff_ffd(price, d)
    assert adf_stat(series) < -2.86  # stationary
    valid = ~np.isnan(series)
    corr = np.corrcoef(series[valid], price[valid])[0, 1]
    assert abs(corr) > 0.1  # retains memory of the level (d=1 would give ~0)


def test_mtf_alignment_is_causal(bars_1m: pl.DataFrame) -> None:
    hour = indicator_features(resample(bars_1m, "1h"))
    base = bars_1m.sort("ts")
    out = align_multi_timeframe(base, {"1h": hour}, context_cols=["ret_1"])
    # For a sample inside hour H, the joined hourly return must come from a *completed* hour < H.
    sample = out.filter(pl.col("ts").dt.hour() == 15).drop_nulls("ret_1_1h").row(0, named=True)
    hour_rows = hour.with_columns(pl.col("ts").alias("bar_open"))
    match = hour_rows.filter(pl.col("ret_1") == sample["ret_1_1h"])
    assert match.height >= 1
    import datetime as dt

    assert all(
        b + dt.timedelta(hours=1) <= sample["ts"] for b in match["bar_open"].to_list()
    ), "joined hourly bar closed after the base sample: lookahead!"


def test_lead_lag_detects_planted_relationship() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    lead_ret = rng.standard_normal(n) * 0.01
    lag_ret = 0.6 * np.roll(lead_ret, 2) + 0.4 * rng.standard_normal(n) * 0.01
    lag_ret[:2] = 0.0
    ts = pl.datetime_range(
        pl.datetime(2025, 1, 1), pl.datetime(2025, 1, 1) + pl.duration(minutes=n - 1), "1m",
        time_zone="UTC", eager=True,
    )

    def frame(rets: np.ndarray) -> pl.DataFrame:
        close = 100 * np.exp(np.cumsum(rets))
        return pl.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close,
                             "volume": np.ones(n)})

    pairs = lead_lag_matrix({"BIG": frame(lead_ret), "SMALL": frame(lag_ret)}, max_lag=5)
    assert pairs, "planted lead-lag not detected"
    top = pairs[0]
    assert (top.leader, top.lagger, top.lag) == ("BIG", "SMALL", 2)
    assert granger_f_stat(lead_ret, lag_ret, 2) > granger_f_stat(lag_ret, lead_ret, 2)

    feat = leader_features(frame(lag_ret), frame(lead_ret), top)
    assert f"lead_{top.leader}_z" in feat.columns


def test_full_pipeline(bars_1m: pl.DataFrame) -> None:
    bars = {
        "1m": bars_1m,
        "1h": resample(bars_1m, "1h"),
        "1d": resample(bars_1m, "1d"),
    }
    frame = build_feature_frame(bars, base_interval="1m", context_intervals=["1h", "1d"])
    cols = feature_columns(frame)
    assert frame.height == bars_1m.height
    assert "rsi_14" in cols and "rsi_14_1h" in cols and "rsi_14_1d" in cols
    assert "ffd_close" in cols and "cdl_engulfing" in cols and "t_sin_day_1" in cols
