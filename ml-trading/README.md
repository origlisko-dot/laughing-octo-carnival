# ml-trading

ML system for intraday US-equity trading. Research-first: nothing here touches real money —
the pipeline is historical data → training → backtest → walk-forward → paper trading (Alpaca Paper).

Three cooperating models:

1. **Multi-timeframe technical model** — candlestick/indicator/cyclical features computed on 1m…1M
   bars and aligned across timeframes without lookahead; triple-barrier labels; a model ladder of
   LightGBM/CatBoost (+Optuna), TabPFN baseline, PatchTST/iTransformer, and Mamba-style SSM;
   validated exclusively with purged & embargoed walk-forward CV.
2. **Risk engine** — a deterministic core (fixed-fraction position sizing, ATR-based SL/TP,
   minimum risk/reward enforcement, exposure caps, daily-loss kill switch) that ML may inform but
   never override; volatility forecasting, meta-labeling, and an optional DRL sizer clipped to the
   hard limits.
3. **Event-driven catalyst model** — near-real-time news + SEC 8-K ingestion, catalyst
   classification (rule-based → FinBERT → LLM), and an impact model predicting abnormal
   volatility/returns after events.

Signals from models 1 and 3 are fused, then pass through the risk engine before any order exists.

## Layout

```
src/ml_trading/
  config.py        central Pydantic config; secrets from env only
  data/            providers (Alpaca/yfinance/synthetic), Parquet+DuckDB warehouse, validation
  features/        candles, indicators, fractional differentiation, lead-lag, MTF alignment
  labeling/        triple-barrier labeling
  models/          tabular (LightGBM/CatBoost/Optuna), deep (PatchTST/iTransformer), regime HMM,
                   meta-labeling, HF foundation-model wrappers
  validation/      purged & embargoed walk-forward CV
  risk/            deterministic risk core, volatility forecasting, DRL sizing env
  events/          EDGAR/news ingestion, catalyst classification, impact model
  backtest/        event-driven simulator, costs, metrics, signal fusion, reports
  execution/       Alpaca paper trading, order-imbalance guard, monitoring
rust/trading-core/ PyO3 acceleration crate (backtest hot loop, candle patterns, risk math)
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'              # core (Polars, LightGBM, Optuna, DuckDB, ...)
uv pip install -e '.[deep,brokers]'     # optional: torch models, Alpaca/yfinance
```

Optional extras: `deep`, `catboost`, `tabpfn`, `foundation` (Chronos), `events-nlp` (FinBERT),
`drl` (PPO sizing), `vol` (GARCH), `brokers`, `dashboard`, `rust`.

Secrets are environment variables only: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`FINNHUB_API_KEY`, `POLYGON_API_KEY`.

## CLI

```bash
mlt ingest --provider synthetic --start 2025-01-01 --end 2025-06-01   # fill the warehouse
mlt train-technical                                                   # walk-forward train + eval
mlt backtest                                                          # simulate with costs
mlt paper-trade --dry-run                                             # paper loop (no orders in dry-run)
```

## Rust core

```bash
cd rust/trading-core
maturin develop --release    # builds the `trading_core` Python module in the active venv
```

Python implementations remain the reference; `tests/test_rust_parity.py` cross-checks the Rust
port bar-for-bar against them whenever the module is importable.

## Tests

```bash
pytest
```

## Non-negotiables

- No lookahead: features may only use data available at bar close; CV is purged and embargoed.
- Every backtest includes commissions and slippage.
- The deterministic risk limits in `config.RiskLimits` bound every ML/DRL decision.
- Deep models must beat the tuned tabular baseline out-of-sample after costs, or they are dropped.
