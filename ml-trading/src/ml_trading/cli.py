"""Command-line interface: ingest, train, backtest, paper-trade."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import typer

from ml_trading.backtest.report import render_report
from ml_trading.backtest.walkforward import summarize, walk_forward_backtest
from ml_trading.config import AppConfig
from ml_trading.data.ingest import ingest_universe
from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.data.store import BarStore
from ml_trading.execution.broker import DryRunBroker
from ml_trading.execution.monitor import TradeLog
from ml_trading.execution.runner import PaperTrader
from ml_trading.features.pipeline import base_features, feature_columns
from ml_trading.labeling.triple_barrier import triple_barrier_labels
from ml_trading.models.tabular import TabularModel

app = typer.Typer(help="ML intraday trading research CLI", no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _provider(name: str):
    if name == "synthetic":
        return SyntheticProvider()
    if name == "yfinance":
        from ml_trading.data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    if name == "alpaca":
        from ml_trading.data.providers.alpaca_provider import AlpacaProvider

        return AlpacaProvider()
    raise typer.BadParameter(f"unknown provider {name!r}")


@app.command()
def ingest(
    provider: str = typer.Option("synthetic", help="synthetic | yfinance | alpaca"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    config: Path | None = typer.Option(None, help="YAML config path"),
    intervals: str = typer.Option("", help="comma-separated subset, e.g. '1m,1h,1d'"),
) -> None:
    """Fetch, validate and store OHLCV bars for the configured universe."""
    cfg = AppConfig.load(config)
    store = BarStore(cfg.data.root)
    ivs = [s.strip() for s in intervals.split(",") if s.strip()] or None
    reports = ingest_universe(_provider(provider), store, cfg, start, end, intervals=ivs)
    bad = [r for r in reports if not r.ok]
    typer.echo(f"ingested {len(reports) - len(bad)} clean frames, {len(bad)} with issues")


@app.command("train-technical")
def train_technical(
    ticker: str = typer.Option("AAPL"),
    interval: str = typer.Option("5m"),
    config: Path | None = typer.Option(None),
    threshold: float = typer.Option(0.6, help="signal probability threshold"),
    report_path: Path = typer.Option(Path("reports/walkforward.html")),
) -> None:
    """Walk-forward train + cost-aware evaluation of the technical model."""
    cfg = AppConfig.load(config)
    bars = BarStore(cfg.data.root).read(ticker, interval)
    if bars.is_empty():
        raise typer.BadParameter(f"no bars for {ticker} {interval}; run `mlt ingest` first")
    outcomes = walk_forward_backtest(bars, cfg, signal_threshold=threshold)
    if not outcomes:
        typer.echo("not enough data for a single fold")
        raise typer.Exit(1)
    path = render_report(outcomes, f"walk-forward {ticker} {interval}", report_path)
    for k, v in summarize(outcomes).items():
        typer.echo(f"{k:>20}: {v:,.4f}")
    typer.echo(f"report: {path}")


@app.command("paper-trade")
def paper_trade(
    ticker: str = typer.Option("AAPL"),
    interval: str = typer.Option("5m"),
    config: Path | None = typer.Option(None),
    dry_run: bool = typer.Option(True, "--dry-run/--live-paper"),
    iterations: int = typer.Option(0, help="0 = run until interrupted"),
    poll_seconds: float = typer.Option(60.0),
) -> None:
    """Paper-trading loop. --dry-run logs orders locally; --live-paper sends to Alpaca Paper."""
    cfg = AppConfig.load(config)
    store = BarStore(cfg.data.root)
    history = store.read(ticker, interval)
    if history.height < 300:
        raise typer.BadParameter("need at least 300 stored bars for feature warm-up")

    feats = base_features(history)
    labels = triple_barrier_labels(history, max_holding=cfg.labels.max_holding_bars)
    cols = [c for c in feature_columns(feats) if feats[c].dtype.is_numeric()]
    X = feats.select(cols).fill_nan(None).to_numpy()
    model = TabularModel().fit(X, labels.label)

    if dry_run:
        broker = DryRunBroker()
    else:
        from ml_trading.execution.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()

    trader = PaperTrader(cfg, model, broker, TradeLog(Path("reports/trades.jsonl")))
    i = 0
    while iterations == 0 or i < iterations:
        latest = store.read(ticker, interval).tail(600)
        decision = trader.on_bar(ticker, latest)
        typer.echo(f"[{i}] {decision.action} {decision.detail}")
        i += 1
        if iterations == 0 or i < iterations:
            time.sleep(poll_seconds)


@app.command()
def backtest(
    ticker: str = typer.Option("AAPL"),
    interval: str = typer.Option("5m"),
    config: Path | None = typer.Option(None),
    threshold: float = typer.Option(0.6),
) -> None:
    """Alias for train-technical without persisting an HTML report path override."""
    train_technical(ticker=ticker, interval=interval, config=config, threshold=threshold,
                    report_path=Path("reports/backtest.html"))


@app.command("min-ffd")
def min_ffd(
    ticker: str = typer.Option("AAPL"),
    interval: str = typer.Option("1d"),
    config: Path | None = typer.Option(None),
) -> None:
    """Report the minimal fractional-differentiation order for a stored series."""
    from ml_trading.features.fracdiff import min_ffd_order

    cfg = AppConfig.load(config)
    bars = BarStore(cfg.data.root).read(ticker, interval)
    if bars.is_empty():
        raise typer.BadParameter("no data stored")
    d = min_ffd_order(np.log(bars["close"].to_numpy()), max_width=300)
    typer.echo(f"minimal FFD order for {ticker} {interval}: d = {d:.2f}")


if __name__ == "__main__":
    app()
