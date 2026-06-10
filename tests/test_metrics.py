"""Metrics engine: hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import Trade
from quantlab.metrics.core import (
    benchmark_comparison,
    cagr,
    longest_losing_streak,
    max_drawdown,
    sharpe,
    trade_stats,
)


def _trade(pnl: float, exit_day: int) -> Trade:
    ts = pd.Timestamp("2020-01-01") + pd.Timedelta(days=exit_day)
    return Trade(symbol="X", entry_ts=ts - pd.Timedelta(days=2), entry_px=100.0, exit_ts=ts,
                 exit_px=100.0 + pnl, qty=1, fees=0.0, slippage=0.0, pnl=pnl, mae=0.0, mfe=0.0,
                 holding_bars=2, exit_reason="signal")


def test_max_drawdown_hand_computed():
    # peak 110 -> trough 99: dd = 99/110 - 1 = -10%
    eq = pd.Series([100.0, 110.0, 99.0, 121.0])
    dd, underwater = max_drawdown(eq)
    assert dd == pytest.approx(-0.1)
    assert underwater == 1  # only the 99 bar is below the running peak


def test_cagr_one_year_doubling():
    eq = pd.Series(np.linspace(100, 200, 252))
    assert cagr(eq) == pytest.approx(1.0, rel=0.02)


def test_sharpe_known_value():
    # alternating +1%/-0.5% has mean .25%, this just pins the formula
    rets = pd.Series([0.01, -0.005] * 50)
    expected = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    assert sharpe(rets) == pytest.approx(float(expected))


def test_sharpe_degenerate_returns_nan():
    assert np.isnan(sharpe(pd.Series([0.01] * 50)))  # zero variance
    assert np.isnan(sharpe(pd.Series([0.01])))       # too short


def test_trade_stats_hand_computed():
    trades = [_trade(10.0, 1), _trade(20.0, 2), _trade(-5.0, 3)]
    stats = trade_stats(trades, initial_capital=10_000.0)
    assert stats["n_trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["profit_factor"] == pytest.approx(30.0 / 5.0)
    assert stats["expectancy_usd"] == pytest.approx(25.0 / 3)
    assert stats["longest_losing_streak"] == 1


def test_longest_losing_streak_ordering():
    # exit order: win, loss, loss, loss, win -> streak 3
    trades = [_trade(5, 1), _trade(-1, 2), _trade(-1, 3), _trade(-1, 4), _trade(5, 5)]
    assert longest_losing_streak(trades) == 3


def test_empty_trades_dont_crash():
    stats = trade_stats([], initial_capital=10_000.0)
    assert stats["n_trades"] == 0
    assert np.isnan(stats["win_rate"])


def test_benchmark_comparison_self_is_beta_one():
    rng = np.random.default_rng(9)
    rets = pd.Series(rng.normal(0.0005, 0.01, 300),
                     index=pd.bdate_range("2020-01-01", periods=300))
    cmp_ = benchmark_comparison(rets, rets)
    assert cmp_["beta"] == pytest.approx(1.0)
    assert cmp_["correlation"] == pytest.approx(1.0)
    assert cmp_["alpha_ann"] == pytest.approx(0.0, abs=1e-12)
    assert cmp_["active_return_ann"] == pytest.approx(0.0, abs=1e-12)
