"""Metrics engine.

Computes every metric the reporting requirements mandate, from an equity curve
and a trade list. Conventions:
- 252 trading days per year; risk-free rate 0 (stated on every report)
- drawdown is peak-to-trough on the mark-to-market equity curve
- trade metrics are net of all modeled fees and slippage
- expectancy is mean net P&L per trade in dollars (and as % of initial capital)
- guards: empty inputs return NaN rather than crashing or fabricating zeros
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestResult, Trade

TRADING_DAYS = 252


def _nan() -> float:
    return float("nan")


def sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return _nan()
    sd = float(daily_returns.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return _nan()
    return float(daily_returns.mean()) / sd * math.sqrt(TRADING_DAYS)


def sortino(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return _nan()
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return _nan()
    dd = float(np.sqrt((downside**2).mean()))
    if dd == 0:
        return _nan()
    return float(daily_returns.mean()) / dd * math.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return (max drawdown as negative fraction, longest underwater stretch in bars)."""
    if len(equity) < 2:
        return _nan(), 0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    longest, current = 0, 0
    for under in dd < 0:
        current = current + 1 if under else 0
        longest = max(longest, current)
    return float(dd.min()), longest


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return _nan()
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return _nan()
    total = float(equity.iloc[-1] / equity.iloc[0])
    if total <= 0:
        return -1.0
    return total ** (1.0 / years) - 1.0


def longest_losing_streak(trades: list[Trade]) -> int:
    streak = longest = 0
    for t in sorted(trades, key=lambda x: (x.exit_ts, x.entry_ts)):
        if t.pnl < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def trade_stats(trades: list[Trade], initial_capital: float) -> dict[str, float]:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": _nan(), "profit_factor": _nan(),
            "expectancy_usd": _nan(), "expectancy_pct": _nan(),
            "avg_win_usd": _nan(), "avg_loss_usd": _nan(),
            "longest_losing_streak": 0, "total_fees_usd": 0.0, "total_slippage_usd": 0.0,
            "gross_profit_usd": 0.0, "gross_loss_usd": 0.0, "cost_share_of_gross": _nan(),
            "avg_holding_bars": _nan(),
        }
    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    fees = float(sum(t.fees for t in trades))
    slip = float(sum(t.slippage for t in trades))
    gross_pnl_before_costs = float(pnls.sum()) + fees + slip
    return {
        "n_trades": float(n),
        "win_rate": float(len(wins)) / n,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else _nan(),
        "expectancy_usd": float(pnls.mean()),
        "expectancy_pct": float(pnls.mean()) / initial_capital,
        "avg_win_usd": float(wins.mean()) if len(wins) else _nan(),
        "avg_loss_usd": float(losses.mean()) if len(losses) else _nan(),
        "longest_losing_streak": float(longest_losing_streak(trades)),
        "total_fees_usd": fees,
        "total_slippage_usd": slip,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "cost_share_of_gross": (fees + slip) / abs(gross_pnl_before_costs)
        if gross_pnl_before_costs != 0 else _nan(),
        "avg_holding_bars": float(np.mean([t.holding_bars for t in trades])),
    }


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    eq = result.equity["equity"]
    rets = result.daily_returns
    mdd, underwater = max_drawdown(eq)
    out: dict[str, float] = {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) >= 2 else _nan(),
        "cagr": cagr(eq),
        "ann_vol": float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else _nan(),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": mdd,
        "longest_underwater_bars": float(underwater),
        "avg_exposure": float(result.equity["exposure"].mean()) if len(result.equity) else _nan(),
        "n_bars": float(len(eq)),
    }
    out.update(trade_stats(result.trades, result.initial_capital))
    return out


def benchmark_comparison(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> dict[str, float]:
    """Alpha/beta (daily OLS, alpha annualized), correlation, information ratio,
    and total/CAGR comparison on the overlapping window."""
    joined = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")], axis=1
    ).dropna()
    if len(joined) < 20:
        return {k: _nan() for k in
                ("alpha_ann", "beta", "correlation", "information_ratio", "active_return_ann")}
    s, b = joined["s"].to_numpy(), joined["b"].to_numpy()
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1]) / var_b if var_b > 0 else _nan()
    alpha_daily = float(s.mean()) - beta * float(b.mean()) if np.isfinite(beta) else _nan()
    active = s - b
    act_sd = float(np.std(active, ddof=1))
    return {
        "alpha_ann": alpha_daily * TRADING_DAYS if np.isfinite(alpha_daily) else _nan(),
        "beta": beta,
        "correlation": float(np.corrcoef(s, b)[0, 1]),
        "information_ratio": float(active.mean()) / act_sd * math.sqrt(TRADING_DAYS)
        if act_sd > 0 else _nan(),
        "active_return_ann": float(active.mean()) * TRADING_DAYS,
    }
