//! trading-core: Rust acceleration for ml-trading.
//!
//! Ports of the performance-critical Python reference implementations:
//!   - `run_backtest`     — the event-driven simulator hot loop
//!   - `candle_patterns`  — candlestick anatomy + pattern detection
//!   - `evaluate_trade`   — deterministic risk-engine sizing math
//!
//! Semantics intentionally mirror the Python code operation-for-operation so the
//! cross-parity tests (`tests/test_rust_parity.py`) can compare bit-for-bit.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// (entry_idx, exit_idx, side(+1/-1), qty, entry_price, exit_price, pnl, exit_reason)
type TradeTuple = (i64, i64, i8, i64, f64, f64, f64, u8);

// ---------------------------------------------------------------------------
// Risk engine math (mirror of ml_trading.risk.engine.RiskEngine.evaluate for
// the single-ticker simulator path: no open positions, sector exposure zero).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct RiskLimits {
    pub risk_per_trade_pct: f64,
    pub min_risk_reward: f64,
    pub max_daily_loss_pct: f64,
    pub max_position_pct: f64,
    pub loss_streak_cooldown: i64,
    pub loss_streak_size_factor: f64,
    pub kelly_cap_fraction: f64,
}

pub struct RiskDecisionCore {
    pub approved: bool,
    pub qty: i64,
    pub risk_amount: f64,
    pub rr: f64,
}

fn kill_switch_active(equity: f64, day_start_equity: f64, max_daily_loss_pct: f64) -> bool {
    let dd = (day_start_equity - equity) / day_start_equity * 100.0;
    dd >= max_daily_loss_pct
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_core(
    is_long: bool,
    entry: f64,
    stop: f64,
    target: f64,
    size_multiplier: f64,
    equity: f64,
    day_start_equity: f64,
    consecutive_losses: i64,
    lim: &RiskLimits,
) -> RiskDecisionCore {
    let (risk_per_share, reward_per_share) = if is_long {
        (entry - stop, target - entry)
    } else {
        (stop - entry, entry - target)
    };
    let rejected = RiskDecisionCore { approved: false, qty: 0, risk_amount: 0.0, rr: 0.0 };
    if risk_per_share <= 0.0 || reward_per_share <= 0.0 {
        return rejected;
    }
    let rr = reward_per_share / risk_per_share;
    if rr < lim.min_risk_reward {
        return RiskDecisionCore { rr, ..rejected };
    }
    if kill_switch_active(equity, day_start_equity, lim.max_daily_loss_pct) {
        return RiskDecisionCore { rr, ..rejected };
    }

    let mut risk_fraction = lim.risk_per_trade_pct / 100.0;
    let kelly = ((0.5 * (rr + 1.0) - 1.0) / rr).max(0.0);
    if kelly > 0.0 {
        risk_fraction = risk_fraction.min(lim.kelly_cap_fraction * kelly);
    }
    if consecutive_losses >= lim.loss_streak_cooldown {
        risk_fraction *= lim.loss_streak_size_factor;
    }
    let mult = size_multiplier.clamp(0.0, 1.0);
    let risk_amount = equity * risk_fraction * mult;

    let mut qty = (risk_amount / risk_per_share) as i64;
    let max_qty_by_value = (equity * lim.max_position_pct / 100.0 / entry) as i64;
    if qty > max_qty_by_value {
        qty = max_qty_by_value;
    }
    if qty <= 0 {
        return RiskDecisionCore { rr, ..rejected };
    }
    RiskDecisionCore { approved: true, qty, risk_amount: qty as f64 * risk_per_share, rr }
}

fn atr_levels_core(entry: f64, atr: f64, is_long: bool, stop_mult: f64, target_mult: f64) -> (f64, f64) {
    if is_long {
        (entry - stop_mult * atr, entry + target_mult * atr)
    } else {
        (entry + stop_mult * atr, entry - target_mult * atr)
    }
}

fn commission(qty: i64, per_share: f64, min_commission: f64) -> f64 {
    (qty.unsigned_abs() as f64 * per_share).max(min_commission)
}

fn fill_price(price: f64, is_buy: bool, slippage_bps: f64) -> f64 {
    let slip = price * slippage_bps / 10_000.0;
    if is_buy {
        price + slip
    } else {
        price - slip
    }
}

// ---------------------------------------------------------------------------
// Backtest hot loop (mirror of ml_trading.backtest.simulator.run_backtest)
// ---------------------------------------------------------------------------

pub struct TradeCore {
    pub entry_idx: i64,
    pub exit_idx: i64,
    pub is_long: bool,
    pub qty: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub exit_reason: u8, // 0=stop 1=target 2=timeout 3=end
}

#[allow(clippy::too_many_arguments)]
pub fn run_backtest_core(
    open_: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    days: &[i64],
    signals: &[i8],
    atr_values: &[f64],
    size_multipliers: &[f64],
    lim: &RiskLimits,
    commission_per_share: f64,
    min_commission: f64,
    slippage_bps: f64,
    initial_equity: f64,
    stop_atr_mult: f64,
    target_atr_mult: f64,
    max_holding_bars: i64,
) -> (Vec<f64>, Vec<TradeCore>) {
    let n = close.len();
    let mut equity = initial_equity;
    let mut day_start_equity = initial_equity;
    let mut consecutive_losses: i64 = 0;
    let mut equity_curve = vec![0.0f64; n];
    let mut trades: Vec<TradeCore> = Vec::new();

    let mut pos_qty: i64 = 0;
    let mut pos_long = true;
    let mut pos_entry = 0.0f64;
    let mut pos_stop = 0.0f64;
    let mut pos_target = 0.0f64;
    let mut pos_entry_idx: i64 = -1;
    let mut pending_signal: i8 = 0;
    let mut pending_mult = 1.0f64;

    for i in 0..n {
        if i > 0 && days[i] != days[i - 1] {
            day_start_equity = equity;
        }

        // 1) execute pending entry at this bar's open
        if pending_signal != 0 && pos_qty == 0 {
            let is_long = pending_signal > 0;
            let raw_entry = open_[i];
            let a = if i > 0 { atr_values[i - 1] } else { f64::NAN };
            if a.is_finite() && a > 0.0 {
                let (stop, target) = atr_levels_core(raw_entry, a, is_long, stop_atr_mult, target_atr_mult);
                let decision = evaluate_core(
                    is_long, raw_entry, stop, target, pending_mult,
                    equity, day_start_equity, consecutive_losses, lim,
                );
                if decision.approved {
                    let fill = fill_price(raw_entry, is_long, slippage_bps);
                    equity -= commission(decision.qty, commission_per_share, min_commission);
                    pos_qty = decision.qty;
                    pos_long = is_long;
                    pos_entry = fill;
                    pos_stop = stop;
                    pos_target = target;
                    pos_entry_idx = i as i64;
                }
            }
            pending_signal = 0;
        }

        // 2) manage the open position
        if pos_qty > 0 {
            let mut exit_reason: i32 = -1;
            let mut exit_price = 0.0f64;
            if pos_long {
                if low[i] <= pos_stop {
                    exit_reason = 0;
                    exit_price = pos_stop;
                } else if high[i] >= pos_target {
                    exit_reason = 1;
                    exit_price = pos_target;
                }
            } else if high[i] >= pos_stop {
                exit_reason = 0;
                exit_price = pos_stop;
            } else if low[i] <= pos_target {
                exit_reason = 1;
                exit_price = pos_target;
            }
            if exit_reason < 0 && (i as i64 - pos_entry_idx) >= max_holding_bars {
                exit_reason = 2;
                exit_price = close[i];
            }
            if exit_reason < 0 && i == n - 1 {
                exit_reason = 3;
                exit_price = close[i];
            }

            if exit_reason >= 0 {
                let fill = fill_price(exit_price, !pos_long, slippage_bps);
                let sign = if pos_long { 1.0 } else { -1.0 };
                let pnl = sign * (fill - pos_entry) * pos_qty as f64
                    - commission(pos_qty, commission_per_share, min_commission);
                equity += pnl;
                consecutive_losses = if pnl > 0.0 { 0 } else { consecutive_losses + 1 };
                trades.push(TradeCore {
                    entry_idx: pos_entry_idx,
                    exit_idx: i as i64,
                    is_long: pos_long,
                    qty: pos_qty,
                    entry_price: pos_entry,
                    exit_price: fill,
                    pnl,
                    exit_reason: exit_reason as u8,
                });
                pos_qty = 0;
            }
        }

        // 3) queue a new signal decided at this bar's close
        if pos_qty == 0
            && signals[i] != 0
            && !kill_switch_active(equity, day_start_equity, lim.max_daily_loss_pct)
        {
            pending_signal = signals[i];
            pending_mult = size_multipliers[i];
        }

        // 4) mark to market
        let unrealized = if pos_qty > 0 {
            let sign = if pos_long { 1.0 } else { -1.0 };
            sign * (close[i] - pos_entry) * pos_qty as f64
        } else {
            0.0
        };
        equity_curve[i] = equity + unrealized;
    }
    (equity_curve, trades)
}

// ---------------------------------------------------------------------------
// Candlestick patterns (mirror of ml_trading.features.candles.candle_features)
// ---------------------------------------------------------------------------

pub struct CandleOutput {
    pub body_frac: Vec<f64>,
    pub upper_frac: Vec<f64>,
    pub lower_frac: Vec<f64>,
    pub direction: Vec<i8>,
    pub doji: Vec<i8>,
    pub hammer_star: Vec<i8>,
    pub engulfing: Vec<i8>,
    pub marubozu: Vec<i8>,
    pub piercing_dcc: Vec<i8>,
    pub harami: Vec<i8>,
    pub gap: Vec<f64>,
}

pub fn candle_patterns_core(open_: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> CandleOutput {
    let n = close.len();
    let mut out = CandleOutput {
        body_frac: vec![0.0; n],
        upper_frac: vec![0.0; n],
        lower_frac: vec![0.0; n],
        direction: vec![0; n],
        doji: vec![0; n],
        hammer_star: vec![0; n],
        engulfing: vec![0; n],
        marubozu: vec![0; n],
        piercing_dcc: vec![0; n],
        harami: vec![0; n],
        gap: vec![f64::NAN; n],
    };

    const TREND_WINDOW: usize = 10;

    for i in 0..n {
        let (o, h, l, c) = (open_[i], high[i], low[i], close[i]);
        let body = (c - o).abs();
        let range = (h - l).max(1e-12);
        let upper = h - o.max(c);
        let lower = o.min(c) - l;
        let bull = c > o;
        let bear = c < o;

        out.body_frac[i] = body / range;
        out.upper_frac[i] = upper / range;
        out.lower_frac[i] = lower / range;
        out.direction[i] = if bull { 1 } else if bear { -1 } else { 0 };
        out.doji[i] = i8::from(body / range < 0.1);
        let maru = body / range > 0.95;
        out.marubozu[i] = if maru && bull { 1 } else if maru && bear { -1 } else { 0 };

        if i >= 1 {
            let (po, pc) = (open_[i - 1], close[i - 1]);
            let prev_body = (pc - po).abs();
            let prev_bull = pc > po;
            let prev_bear = pc < po;
            let prev_mid = (po + pc) / 2.0;

            out.gap[i] = (o - pc) / range;

            let engulf_bull = bull && prev_bear && c >= po && o <= pc && body > prev_body;
            let engulf_bear = bear && prev_bull && c <= po && o >= pc && body > prev_body;
            out.engulfing[i] = if engulf_bull { 1 } else if engulf_bear { -1 } else { 0 };

            let piercing = bull && prev_bear && o < pc && c > prev_mid && c < po;
            let dark_cloud = bear && prev_bull && o > pc && c < prev_mid && c > po;
            out.piercing_dcc[i] = if piercing { 1 } else if dark_cloud { -1 } else { 0 };

            let harami_bull = bull && prev_bear && o > pc && c < po && body < prev_body;
            let harami_bear = bear && prev_bull && o < pc && c > po && body < prev_body;
            out.harami[i] = if harami_bull { 1 } else if harami_bear { -1 } else { 0 };
        }

        // hammer / shooting star gated by the prior 10-bar trend of prev close
        if i >= TREND_WINDOW {
            let prev_close = close[i - 1];
            let mean: f64 = close[i - TREND_WINDOW..i].iter().sum::<f64>() / TREND_WINDOW as f64;
            let downtrend = prev_close < mean;
            let uptrend = prev_close > mean;
            let hammer = lower > 2.0 * body && upper < body;
            let shooting = upper > 2.0 * body && lower < body;
            out.hammer_star[i] = if hammer && downtrend {
                1
            } else if shooting && uptrend {
                -1
            } else {
                0
            };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

fn limits_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RiskLimits> {
    let get_f = |k: &str| -> PyResult<f64> {
        d.get_item(k)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()))?
            .extract()
    };
    let get_i = |k: &str| -> PyResult<i64> {
        d.get_item(k)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()))?
            .extract()
    };
    Ok(RiskLimits {
        risk_per_trade_pct: get_f("risk_per_trade_pct")?,
        min_risk_reward: get_f("min_risk_reward")?,
        max_daily_loss_pct: get_f("max_daily_loss_pct")?,
        max_position_pct: get_f("max_position_pct")?,
        loss_streak_cooldown: get_i("loss_streak_cooldown")?,
        loss_streak_size_factor: get_f("loss_streak_size_factor")?,
        kelly_cap_fraction: get_f("kelly_cap_fraction")?,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest<'py>(
    py: Python<'py>,
    open_: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    days: PyReadonlyArray1<'py, i64>,
    signals: PyReadonlyArray1<'py, i8>,
    atr_values: PyReadonlyArray1<'py, f64>,
    size_multipliers: PyReadonlyArray1<'py, f64>,
    limits: Bound<'py, PyDict>,
    commission_per_share: f64,
    min_commission: f64,
    slippage_bps: f64,
    initial_equity: f64,
    stop_atr_mult: f64,
    target_atr_mult: f64,
    max_holding_bars: i64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Vec<TradeTuple>)> {
    let lim = limits_from_dict(&limits)?;
    let (curve, trades) = run_backtest_core(
        open_.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        days.as_slice()?,
        signals.as_slice()?,
        atr_values.as_slice()?,
        size_multipliers.as_slice()?,
        &lim,
        commission_per_share,
        min_commission,
        slippage_bps,
        initial_equity,
        stop_atr_mult,
        target_atr_mult,
        max_holding_bars,
    );
    let trade_tuples = trades
        .into_iter()
        .map(|t| {
            (
                t.entry_idx,
                t.exit_idx,
                if t.is_long { 1i8 } else { -1i8 },
                t.qty,
                t.entry_price,
                t.exit_price,
                t.pnl,
                t.exit_reason,
            )
        })
        .collect();
    Ok((curve.into_pyarray_bound(py), trade_tuples))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn evaluate_trade(
    is_long: bool,
    entry: f64,
    stop: f64,
    target: f64,
    size_multiplier: f64,
    equity: f64,
    day_start_equity: f64,
    consecutive_losses: i64,
    limits: Bound<'_, PyDict>,
) -> PyResult<(bool, i64, f64, f64)> {
    let lim = limits_from_dict(&limits)?;
    let d = evaluate_core(
        is_long, entry, stop, target, size_multiplier,
        equity, day_start_equity, consecutive_losses, &lim,
    );
    Ok((d.approved, d.qty, d.risk_amount, d.rr))
}

#[pyfunction]
fn candle_patterns<'py>(
    py: Python<'py>,
    open_: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = candle_patterns_core(open_.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?);
    let d = PyDict::new_bound(py);
    d.set_item("cdl_body_frac", out.body_frac.into_pyarray_bound(py))?;
    d.set_item("cdl_upper_frac", out.upper_frac.into_pyarray_bound(py))?;
    d.set_item("cdl_lower_frac", out.lower_frac.into_pyarray_bound(py))?;
    d.set_item("cdl_direction", out.direction.into_pyarray_bound(py))?;
    d.set_item("cdl_doji", out.doji.into_pyarray_bound(py))?;
    d.set_item("cdl_hammer_star", out.hammer_star.into_pyarray_bound(py))?;
    d.set_item("cdl_engulfing", out.engulfing.into_pyarray_bound(py))?;
    d.set_item("cdl_marubozu", out.marubozu.into_pyarray_bound(py))?;
    d.set_item("cdl_piercing_dcc", out.piercing_dcc.into_pyarray_bound(py))?;
    d.set_item("cdl_harami", out.harami.into_pyarray_bound(py))?;
    d.set_item("cdl_gap", out.gap.into_pyarray_bound(py))?;
    Ok(d)
}

#[pymodule]
fn trading_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_trade, m)?)?;
    m.add_function(wrap_pyfunction!(candle_patterns, m)?)?;
    Ok(())
}
