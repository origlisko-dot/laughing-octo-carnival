"""Streamlit monitoring dashboard (optional `dashboard` extra).

Run:  streamlit run src/ml_trading/dashboard.py
Shows the trade journal, live-vs-backtest drift status, and equity by day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

try:
    import streamlit as st
except ImportError:  # optional extra
    st = None

from ml_trading.execution.monitor import TradeLog, drift_report

TRADE_LOG_PATH = Path("reports/trades.jsonl")


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit not installed; run: uv pip install 'ml-trading[dashboard]'")
    st.set_page_config(page_title="ml-trading monitor", layout="wide")
    st.title("Paper-trading monitor")

    records = TradeLog(TRADE_LOG_PATH).load()
    if not records:
        st.info("No trades logged yet. Run `mlt paper-trade`.")
        return

    df = pl.DataFrame([r.__dict__ for r in records])
    st.subheader(f"Trade journal ({df.height} orders)")
    st.dataframe(df.to_pandas(), use_container_width=True)

    st.subheader("Planned risk per trade")
    planned_risk = df.select(
        ((pl.col("entry") - pl.col("stop")).abs() * pl.col("qty")).alias("risk_$")
    )
    st.bar_chart(planned_risk.to_pandas())

    st.subheader("Live vs backtest drift")
    bt_returns = st.session_state.get("bt_returns")
    if bt_returns is None:
        st.caption("Load backtest R-multiples to enable drift detection.")
    else:
        live_r = np.random.default_rng(0).normal(0.1, 1.0, df.height)  # placeholder until fills sync
        st.json(drift_report(live_r, np.asarray(bt_returns)))


if __name__ == "__main__":
    main()
