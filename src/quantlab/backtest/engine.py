"""Event-driven daily backtest engine.

Execution semantics (these ARE the realism guarantees - change them only with
an engine version bump):

1. Signals are computed strictly AFTER the close of bar t, from a PITView that
   cannot see past t.
2. Orders fill at the OPEN of bar t+1 (+ optional extra delay bars for delay
   stress). There are no same-bar fills, ever.
3. Every entry carries a protective stop (entry - atr_stop_mult * ATR at
   signal time). Stops are checked intraday: if bar low touches the stop, the
   exit fills at the stop price - or at the open if the bar gapped below it
   (gaps do not fill at fantasy prices).
4. Position sizing is risk-based: risk_per_trade_pct of current equity divided
   by the stop distance, capped by max_position_pct and by available cash.
   Strategies never size their own positions.
5. Long-only cash account, enforced structurally: a sell can only close an
   existing position; cash can never go negative; shorting cannot be
   represented.
6. All cash arithmetic is integer micro-dollars (1e-6 USD): no float drift in
   the ledger, and results are bit-reproducible.
7. At end of data, remaining positions are liquidated at the final close
   (exit_reason='end_of_data') so metrics never carry phantom open equity.

The engine is fully deterministic: same data + config + strategy params give
byte-identical results. There is no randomness anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.data.pit import MarketData
from quantlab.foundation.config import BacktestConfig
from quantlab.indicators import atr
from quantlab.strategies.contract import Strategy

MICRO = 1_000_000  # micro-dollars per dollar


def _to_micro(dollars: float) -> int:
    return int(round(dollars * MICRO))


@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: pd.Timestamp
    exit_px: float
    qty: int
    fees: float
    slippage: float
    pnl: float
    mae: float  # worst adverse excursion, fraction of entry price (<= 0)
    mfe: float  # best favorable excursion, fraction of entry price (>= 0)
    holding_bars: int
    exit_reason: str  # 'signal' | 'stop' | 'end_of_data'
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Position:
    qty: int
    entry_px: float
    entry_ts: pd.Timestamp
    stop_px: float
    entry_fee: float
    entry_slip: float
    lowest_low: float
    highest_high: float
    holding_bars: int
    context: dict[str, Any]


@dataclass
class _PendingOrder:
    side: str  # 'buy' | 'sell'
    created_idx: int  # position in engine calendar when signal was generated
    delay: int  # extra bars beyond next-open
    stop_dist: float  # for buys: protective stop distance in price units
    adv: float  # 20-day ADV at signal time, for impact cost
    context: dict[str, Any]


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # index date; columns: equity, cash, exposure
    trades: list[Trade]
    initial_capital: float
    n_signals: int

    @property
    def daily_returns(self) -> pd.Series:
        return self.equity["equity"].pct_change().dropna()

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=["symbol", "entry_ts", "entry_px", "exit_ts", "exit_px", "qty",
                         "fees", "slippage", "pnl", "mae", "mfe", "holding_bars", "exit_reason"]
            )
        rows = []
        for t in self.trades:
            rows.append({
                "symbol": t.symbol, "entry_ts": t.entry_ts, "entry_px": t.entry_px,
                "exit_ts": t.exit_ts, "exit_px": t.exit_px, "qty": t.qty, "fees": t.fees,
                "slippage": t.slippage, "pnl": t.pnl, "mae": t.mae, "mfe": t.mfe,
                "holding_bars": t.holding_bars, "exit_reason": t.exit_reason,
                "context_json": json.dumps(t.context, sort_keys=True, default=str),
            })
        return pd.DataFrame(rows)


def run_backtest(
    market: MarketData,
    strategy: Strategy,
    config: BacktestConfig,
    universe: list[str],
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
    cost_multiplier: float = 1.0,
    entry_delay_bars: int = 0,
) -> BacktestResult:
    """Run one deterministic backtest.

    `trade_start`/`trade_end` bound when signals may be GENERATED (walk-forward
    uses this); history before trade_start is still visible point-in-time, so
    warm-up never leaks the test window into the train window.
    """
    universe = [s.upper() for s in universe]
    for sym in universe:
        market.frame(sym)  # raises early on unknown symbols
    costs = CostModel(config.costs, stress_multiplier=cost_multiplier)
    sizing = config.sizing

    calendar = market.calendar
    start = pd.Timestamp(trade_start) if trade_start is not None else calendar[0]
    end = pd.Timestamp(trade_end) if trade_end is not None else calendar[-1]

    cash_micro = _to_micro(config.initial_capital)
    positions: dict[str, _Position] = {}
    pending: dict[str, _PendingOrder] = {}
    trades: list[Trade] = []
    equity_rows: list[tuple[pd.Timestamp, float, float, float]] = []
    last_equity_micro = cash_micro
    last_close: dict[str, float] = {}
    n_signals = 0

    def close_position(
        sym: str, pos: _Position, exit_ts: pd.Timestamp, ref_price: float, reason: str
    ) -> None:
        nonlocal cash_micro
        exit_px = costs.sell_price(ref_price, pos.qty, _adv(sym, exit_ts, include_asof=False))
        fee = costs.commission(pos.qty)
        proceeds = _to_micro(exit_px) * pos.qty - _to_micro(fee)
        cash_micro += proceeds
        exit_slip = (ref_price - exit_px) * pos.qty
        entry_bar_count = pos.holding_bars
        trades.append(
            Trade(
                symbol=sym,
                entry_ts=pos.entry_ts,
                entry_px=pos.entry_px,
                exit_ts=exit_ts,
                exit_px=exit_px,
                qty=pos.qty,
                fees=round(pos.entry_fee + fee, 6),
                slippage=round(pos.entry_slip + exit_slip, 6),
                pnl=round((exit_px - pos.entry_px) * pos.qty - pos.entry_fee - fee, 6),
                mae=round(min(0.0, pos.lowest_low / pos.entry_px - 1.0), 6),
                mfe=round(max(0.0, pos.highest_high / pos.entry_px - 1.0), 6),
                holding_bars=entry_bar_count,
                exit_reason=reason,
                context=pos.context,
            )
        )
        del positions[sym]

    def _adv(sym: str, as_of: pd.Timestamp, include_asof: bool = True) -> float:
        # include_asof=False for fills happening DURING the as_of bar (the bar's
        # own volume is not knowable at its open).
        frame = market.frame(sym)
        side = "right" if include_asof else "left"
        idx = int(np.searchsorted(frame.index.values, as_of.to_datetime64(), side=side))
        window = frame["volume"].iloc[max(0, idx - 20) : idx]
        return float(window.mean()) if len(window) else 0.0

    for cal_idx, today in enumerate(calendar):
        if today > end:
            break

        # ---- phase 1: execute pending orders at today's open ----
        for sym in list(pending):
            order = pending[sym]
            if cal_idx < order.created_idx + 1 + order.delay:
                continue
            frame = market.frame(sym)
            if today not in frame.index:
                continue  # symbol has no bar today; order waits for its next bar
            open_px = float(frame.at[today, "open"])

            if order.side == "sell":
                if sym in positions:
                    close_position(sym, positions[sym], today, open_px, "signal")
                del pending[sym]
                continue

            # buy: size by risk budget at the stop distance
            equity_now = last_equity_micro / MICRO
            if order.stop_dist <= 0 or not np.isfinite(order.stop_dist):
                del pending[sym]
                continue
            risk_qty = int((equity_now * sizing.risk_per_trade_pct / 100.0) / order.stop_dist)
            cap_qty = int((equity_now * sizing.max_position_pct / 100.0) / open_px)
            qty = max(0, min(risk_qty, cap_qty))
            if qty > 0:
                fill_px = costs.buy_price(open_px, qty, order.adv)
                fee = costs.commission(qty)
                cost_micro = _to_micro(fill_px) * qty + _to_micro(fee)
                if cost_micro > cash_micro:
                    # cash account: shrink to what cash affords, never go negative
                    affordable = int((cash_micro - _to_micro(fee)) / max(_to_micro(fill_px), 1))
                    qty = max(0, min(qty, affordable))
                    fill_px = costs.buy_price(open_px, qty, order.adv) if qty else fill_px
                    fee = costs.commission(qty)
                    cost_micro = _to_micro(fill_px) * qty + _to_micro(fee)
            if qty > 0 and cost_micro <= cash_micro:
                cash_micro -= cost_micro
                positions[sym] = _Position(
                    qty=qty,
                    entry_px=fill_px,
                    entry_ts=today,
                    stop_px=fill_px - order.stop_dist,
                    entry_fee=fee,
                    entry_slip=(fill_px - open_px) * qty,
                    lowest_low=fill_px,
                    highest_high=fill_px,
                    holding_bars=0,
                    context=order.context,
                )
            del pending[sym]

        # ---- phase 2: protective stops, checked against today's range ----
        for sym in list(positions):
            frame = market.frame(sym)
            if today not in frame.index:
                continue
            pos = positions[sym]
            low = float(frame.at[today, "low"])
            if low <= pos.stop_px:
                open_px = float(frame.at[today, "open"])
                # gapped below the stop -> fill at open; otherwise at the stop
                ref = open_px if open_px <= pos.stop_px else pos.stop_px
                close_position(sym, pos, today, ref, "stop")

        # ---- phase 3: excursion tracking and mark-to-market at the close ----
        invested_micro = 0
        for sym, pos in positions.items():
            frame = market.frame(sym)
            if today in frame.index:
                pos.lowest_low = min(pos.lowest_low, float(frame.at[today, "low"]))
                pos.highest_high = max(pos.highest_high, float(frame.at[today, "high"]))
                pos.holding_bars += 1
                last_close[sym] = float(frame.at[today, "close"])
            mark = last_close.get(sym, pos.entry_px)
            invested_micro += _to_micro(mark) * pos.qty
        for sym in universe:
            frame = market.frame(sym)
            if today in frame.index:
                last_close[sym] = float(frame.at[today, "close"])

        equity_micro = cash_micro + invested_micro
        last_equity_micro = equity_micro
        if start <= today <= end:
            exposure = invested_micro / equity_micro if equity_micro > 0 else 0.0
            equity_rows.append(
                (today, equity_micro / MICRO, cash_micro / MICRO, round(exposure, 6))
            )

        # ---- phase 4: signals after the close (none on the final bar) ----
        if today < start or today >= end or cal_idx == len(calendar) - 1:
            continue
        view = market.view(today)
        for sig in strategy.signals(view, universe):
            sym = sig.symbol.upper()
            if sym not in universe or sym in pending:
                continue
            n_signals += 1
            if sig.desired_state == "long" and sym not in positions:
                bars = view.bars(sym, n=sizing.atr_period + 1)
                if len(bars) < sizing.atr_period + 1:
                    continue
                atr_val = float(atr(bars["high"], bars["low"], bars["close"], sizing.atr_period).iloc[-1])
                if not np.isfinite(atr_val) or atr_val <= 0:
                    continue
                pending[sym] = _PendingOrder(
                    side="buy",
                    created_idx=cal_idx,
                    delay=entry_delay_bars,
                    stop_dist=sizing.atr_stop_mult * atr_val,
                    adv=_adv(sym, today),
                    context=dict(sig.context),
                )
            elif sig.desired_state == "flat" and sym in positions:
                pending[sym] = _PendingOrder(
                    side="sell", created_idx=cal_idx, delay=0, stop_dist=0.0, adv=0.0,
                    context=dict(sig.context),
                )

    # ---- end of data: liquidate at the final available close ----
    final_ts = equity_rows[-1][0] if equity_rows else calendar[-1]
    for sym in list(positions):
        pos = positions[sym]
        ref = last_close.get(sym, pos.entry_px)
        close_position(sym, pos, final_ts, ref, "end_of_data")
    if equity_rows:
        # restate the final mark with liquidation costs realized
        invested_micro = 0
        equity_rows[-1] = (
            final_ts, cash_micro / MICRO, cash_micro / MICRO, 0.0,
        )

    equity = pd.DataFrame(
        equity_rows, columns=["date", "equity", "cash", "exposure"]
    ).set_index("date")
    return BacktestResult(
        equity=equity,
        trades=trades,
        initial_capital=config.initial_capital,
        n_signals=n_signals,
    )
