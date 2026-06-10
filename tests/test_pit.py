"""Point-in-time API: structural no-lookahead, copies, and strategy-level checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.pit import MarketData
from quantlab.data.synth import generate_daily
from quantlab.strategies.catalog.sma_cross import SmaCross


def _market(n: int = 300) -> MarketData:
    return MarketData({"AAA": generate_daily("AAA", n, seed=42), "BBB": generate_daily("BBB", n, seed=43)})


def test_view_never_returns_future_bars():
    market = _market()
    frame = market.frame("AAA")
    for as_of in frame.index[::25]:
        bars = market.view(as_of).bars("AAA")
        assert (bars.index <= as_of).all()
        assert bars.index[-1] == as_of


def test_view_respects_n():
    market = _market()
    as_of = market.frame("AAA").index[100]
    assert len(market.view(as_of).bars("AAA", n=30)) == 30
    assert len(market.view(market.frame("AAA").index[5]).bars("AAA", n=30)) == 6


def test_view_returns_copies():
    market = _market()
    as_of = market.frame("AAA").index[50]
    bars = market.view(as_of).bars("AAA")
    bars.loc[:, "close"] = -1.0
    again = market.view(as_of).bars("AAA")
    assert (again["close"] > 0).all()


def test_strategy_signals_unaffected_by_future_data():
    """End-to-end no-lookahead at the strategy level: change everything after
    day k, signals at day k-1 must be identical."""
    base = generate_daily("AAA", 300, seed=42)
    k = 250
    rng = np.random.default_rng(0)
    mutated = base.copy()
    factor = rng.uniform(0.5, 1.5, len(base) - k)
    for col in ("open", "high", "low", "close"):
        mutated.iloc[k:, mutated.columns.get_loc(col)] = (
            mutated.iloc[k:, mutated.columns.get_loc(col)].to_numpy() * factor
        )

    m1, m2 = MarketData({"AAA": base}), MarketData({"AAA": mutated})
    strat = SmaCross({"fast": 20, "slow": 100})
    as_of = base.index[k - 1]
    s1 = strat.signals(m1.view(as_of), ["AAA"])
    s2 = strat.signals(m2.view(as_of), ["AAA"])
    assert [(s.symbol, s.desired_state, s.context) for s in s1] == \
           [(s.symbol, s.desired_state, s.context) for s in s2]


def test_calendar_is_union_of_symbol_dates():
    a = generate_daily("AAA", 100, seed=1)
    b = generate_daily("BBB", 100, seed=2).iloc[10:]  # shorter history
    market = MarketData({"AAA": a, "BBB": b})
    assert set(market.calendar) == set(a.index) | set(b.index)
