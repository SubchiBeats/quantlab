"""Benchmark construction: buy-and-hold on the same data snapshot.

The benchmark pays the same cost model on its initial purchase, marks to
market daily, and never trades again. Comparing against it answers the
question every strategy must answer: did the work beat doing nothing?
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.data.pit import MarketData
from quantlab.foundation.config import CostConfig


def buy_hold_equity(
    market: MarketData,
    symbol: str,
    initial_capital: float,
    cost_cfg: CostConfig,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    frame = market.frame(symbol)
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    if len(frame) < 2:
        raise ValueError(f"benchmark {symbol}: not enough bars in window")

    costs = CostModel(cost_cfg)
    adv = float(frame["volume"].iloc[:20].mean())
    first_open = float(frame["open"].iloc[0])
    # size first (approx), then price the fill with impact for that size
    qty = int(initial_capital / first_open)
    fill_px = costs.buy_price(first_open, qty, adv)
    qty = int((initial_capital - costs.commission(qty)) / fill_px)
    if qty <= 0:
        raise ValueError(f"benchmark {symbol}: capital too small to buy one share")
    cash = initial_capital - qty * fill_px - costs.commission(qty)
    equity = frame["close"] * qty + cash
    equity.name = "equity"
    return equity
