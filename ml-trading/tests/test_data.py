from datetime import UTC, datetime

import polars as pl
import pytest

from ml_trading.config import AppConfig
from ml_trading.data.ingest import ingest_universe
from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.data.store import BarStore, resample
from ml_trading.data.validation import filter_rth, validate_ohlcv


@pytest.fixture
def bars_1m() -> pl.DataFrame:
    return SyntheticProvider(seed=1).fetch("TEST", "1m", "2025-01-06", "2025-01-10")


def test_synthetic_provider_valid(bars_1m: pl.DataFrame) -> None:
    rep = validate_ohlcv(bars_1m, "TEST", "1m")
    assert rep.n_rows > 1000
    assert rep.invalid_ohlc == 0
    assert rep.nonpositive_price == 0
    assert rep.duplicated_ts == 0


def test_validation_catches_bad_ohlc() -> None:
    df = pl.DataFrame(
        {
            "ts": [datetime(2025, 1, 6, 14, m, tzinfo=UTC) for m in range(3)],
            "open": [10.0, 10.0, 10.0],
            "high": [9.0, 11.0, 11.0],  # first bar: high < open
            "low": [8.0, 9.5, -1.0],  # last bar: negative price
            "close": [8.5, 10.5, 10.0],
            "volume": [100.0, 100.0, -5.0],
        }
    )
    rep = validate_ohlcv(df, "BAD", "1m")
    assert rep.invalid_ohlc >= 1
    assert rep.nonpositive_price == 1
    assert rep.negative_volume == 1
    assert not rep.ok


def test_store_roundtrip_and_upsert(tmp_path, bars_1m: pl.DataFrame) -> None:
    store = BarStore(tmp_path)
    n1 = store.write("TEST", "1m", bars_1m)
    n2 = store.write("TEST", "1m", bars_1m.tail(50))  # overlapping upsert
    assert n1 == n2 == bars_1m.height
    out = store.read("TEST", "1m")
    assert out.height == bars_1m.height
    assert out["ts"].is_sorted()

    windowed = store.read("TEST", "1m", start="2025-01-07", end="2025-01-08")
    assert windowed.height > 0
    assert windowed["ts"].dt.date().unique().to_list() == [datetime(2025, 1, 7).date()]


def test_duckdb_sql(tmp_path, bars_1m: pl.DataFrame) -> None:
    store = BarStore(tmp_path)
    store.write("TEST", "1m", bars_1m)
    res = store.sql("SELECT ticker, interval, count(*) AS n FROM bars GROUP BY 1, 2")
    assert res["n"][0] == bars_1m.height


def test_resample_consistency(bars_1m: pl.DataFrame) -> None:
    five = resample(bars_1m, "5m")
    assert five.height < bars_1m.height
    # Aggregates must preserve totals/extremes.
    assert five["volume"].sum() == pytest.approx(bars_1m["volume"].sum())
    assert five["high"].max() == pytest.approx(bars_1m["high"].max())
    assert five["low"].min() == pytest.approx(bars_1m["low"].min())


def test_filter_rth(bars_1m: pl.DataFrame) -> None:
    assert filter_rth(bars_1m).height == bars_1m.height  # synthetic already RTH-only


def test_ingest_universe(tmp_path) -> None:
    cfg = AppConfig()
    cfg.data.universe = ["AAA", "BBB"]
    store = BarStore(tmp_path)
    reports = ingest_universe(
        SyntheticProvider(), store, cfg, "2025-01-06", "2025-01-08", intervals=["5m", "15m"]
    )
    assert len(reports) == 4
    assert store.tickers("5m") == ["AAA", "BBB"]
