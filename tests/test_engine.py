"""Backtest engine: hand-computed fixtures verifying fill semantics, stops,
sizing, cost application, cash-account invariants, and determinism.

The numbers in these tests are computed by hand in the comments - if the
engine ever disagrees with them, the engine changed behavior, not the test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.backtest.engine import run_backtest
from quantlab.data.pit import MarketData, PITView
from quantlab.foundation.config import BacktestConfig, CostConfig, SizingConfig
from quantlab.strategies.contract import Signal, Strategy

# Base bar: open/close 100, range 99.5..101 -> TR = 1.5, ATR(2) = 1.5
BASE = (100.0, 101.0, 99.5, 100.0)
# After the entry signal the price level steps to 105
HELD = (105.0, 106.0, 104.5, 105.0)


def make_market(specs: list[tuple[float, float, float, float]]) -> MarketData:
    index = pd.bdate_range("2020-01-06", periods=len(specs))
    frame = pd.DataFrame(
        specs, columns=["open", "high", "low", "close"], index=pd.Index(index, name="date")
    )
    frame["volume"] = 1_000_000
    return MarketData({"SYM": frame})


def zero_cost_config(max_position_pct: float = 10.0) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=100_000.0,
        fill_policy="next_bar_open",
        cash_account=True,
        allow_short=False,
        costs=CostConfig(commission_per_share=0.0, min_commission=0.0, half_spread_bps=0.0,
                         impact_coeff=0.0, pessimism_multiplier=1.0),
        sizing=SizingConfig(risk_per_trade_pct=0.5, atr_period=2, atr_stop_mult=3.0,
                            max_position_pct=max_position_pct),
    )


class ScriptStrategy(Strategy):
    """Emits pre-scripted signals on specific dates; used to probe the engine."""

    name = "script"
    version = 1

    def __init__(self, script: dict[pd.Timestamp, list[Signal]]):
        super().__init__({})
        self._script = script

    @classmethod
    def param_space(cls) -> dict:
        return {}

    def warmup_bars(self) -> int:
        return 0

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        return list(self._script.get(view.as_of, []))


def long_flat_script(dates: pd.DatetimeIndex, long_at: int, flat_at: int | None) -> ScriptStrategy:
    script = {dates[long_at]: [Signal("SYM", "long")]}
    if flat_at is not None:
        script[dates[flat_at]] = [Signal("SYM", "flat")]
    return ScriptStrategy(script)


def test_next_bar_open_fill_and_risk_sizing():
    # Signal after close of day 5 (close=100); day 6 opens at 105.
    # ATR(2) at day 5 = 1.5 -> stop_dist = 4.5
    # risk qty = 0.5% * 100k / 4.5 = 111; cap = 10% * 100k / 105 = 95 -> qty 95
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, 10), zero_cost_config(), ["SYM"])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_ts == dates[6]          # next bar, not the signal bar
    assert trade.entry_px == pytest.approx(105.0)  # day-6 OPEN, not day-5 close
    assert trade.qty == 95                     # position cap binds before risk budget
    assert trade.exit_ts == dates[11]          # flat signal day 10 -> exit day 11 open
    assert trade.exit_px == pytest.approx(105.0)
    assert trade.pnl == pytest.approx(0.0)
    assert trade.exit_reason == "signal"
    assert trade.holding_bars == 5             # held through days 6..10
    assert result.equity["equity"].iloc[-1] == pytest.approx(100_000.0)


def test_risk_budget_binds_when_cap_is_loose():
    # cap 25% -> 25000/105 = 238; risk qty = 500/4.5 = 111 -> qty 111
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(
        market, long_flat_script(dates, 5, 10), zero_cost_config(max_position_pct=25.0), ["SYM"]
    )
    assert result.trades[0].qty == 111


def test_stop_loss_fills_at_stop_price():
    # entry day 6 at 105, stop = 105 - 4.5 = 100.5
    # day 8 trades down to 100 (open 105 > stop) -> stop fill at 100.5
    # pnl = (100.5 - 105) * 95 = -427.50
    specs = [BASE] * 6 + [HELD, HELD, (105.0, 106.0, 100.0, 102.0)] + [HELD] * 6
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"])

    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_ts == dates[8]
    assert trade.exit_px == pytest.approx(100.5)
    assert trade.pnl == pytest.approx(-427.5)
    assert result.equity["equity"].iloc[-1] == pytest.approx(100_000.0 - 427.5)


def test_gap_below_stop_fills_at_open_not_stop():
    # day 8 gaps open at 99 (below stop 100.5) -> fill at the open, the honest price
    # pnl = (99 - 105) * 95 = -570
    specs = [BASE] * 6 + [HELD, HELD, (99.0, 105.0, 98.0, 100.0)] + [HELD] * 6
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"])

    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_px == pytest.approx(99.0)
    assert trade.pnl == pytest.approx(-570.0)


def test_end_of_data_liquidation():
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"])

    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_px == pytest.approx(105.0)  # final close, costs are zero here
    assert trade.pnl == pytest.approx(0.0)
    # no phantom open equity: final equity row equals cash
    assert result.equity["equity"].iloc[-1] == pytest.approx(result.equity["cash"].iloc[-1])


def test_costs_are_charged_and_recorded():
    config = BacktestConfig(
        initial_capital=100_000.0, fill_policy="next_bar_open", cash_account=True,
        allow_short=False,
        costs=CostConfig(commission_per_share=0.01, min_commission=1.0, half_spread_bps=10.0,
                         impact_coeff=0.0, pessimism_multiplier=2.0),
        sizing=SizingConfig(risk_per_trade_pct=0.5, atr_period=2, atr_stop_mult=3.0,
                            max_position_pct=10.0),
    )
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, 10), config, ["SYM"])

    trade = result.trades[0]
    assert trade.fees > 0
    assert trade.slippage > 0
    # slippage is embedded in fill prices; pnl must reconcile exactly with them
    assert trade.pnl == pytest.approx((trade.exit_px - trade.entry_px) * trade.qty - trade.fees)
    # buys above reference, sells below it
    assert trade.entry_px > 105.0
    assert trade.exit_px < 105.0


def test_cash_account_invariants():
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(market, long_flat_script(dates, 5, 10), zero_cost_config(), ["SYM"])
    assert (result.equity["cash"] >= 0).all()          # cash never negative
    assert all(t.qty > 0 for t in result.trades)        # long-only by construction


def test_determinism_bit_identical():
    specs = [BASE] * 6 + [HELD, HELD, (105.0, 106.0, 100.0, 102.0)] + [HELD] * 6
    market = make_market(specs)
    dates = market.frame("SYM").index

    r1 = run_backtest(market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"])
    r2 = run_backtest(market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"])
    pd.testing.assert_frame_equal(r1.equity, r2.equity)
    assert [(t.entry_px, t.exit_px, t.qty, t.pnl) for t in r1.trades] == \
           [(t.entry_px, t.exit_px, t.qty, t.pnl) for t in r2.trades]


def test_entry_delay_shifts_fill():
    specs = [BASE] * 6 + [HELD] * 9
    market = make_market(specs)
    dates = market.frame("SYM").index
    result = run_backtest(
        market, long_flat_script(dates, 5, None), zero_cost_config(), ["SYM"], entry_delay_bars=2
    )
    assert result.trades[0].entry_ts == dates[8]  # signal day 5 + 1 + 2 delay
