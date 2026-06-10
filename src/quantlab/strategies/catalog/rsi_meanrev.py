"""Reference strategy: short-term RSI mean reversion with trend filter.

Hypothesis shape: liquid equities in uptrends overshoot on short-term selling
pressure; buy brief oversold readings, exit when pressure normalizes.

Enter long when RSI(rsi_n) < buy_below while price is above SMA(trend_sma);
exit when RSI > exit_above or the trend filter fails. Long-only.
"""

from __future__ import annotations

from typing import Any

from quantlab.data.pit import PITView
from quantlab.indicators import rsi, sma
from quantlab.strategies.contract import Signal, Strategy
from quantlab.strategies.factory import register


@register
class RsiMeanRev(Strategy):
    name = "rsi_meanrev"
    version = 1

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {
            "rsi_n": [2, 3, 4],
            "buy_below": [10, 20, 30],
            "exit_above": [50, 60, 70],
            "trend_sma": [200],
        }

    @classmethod
    def params_valid(cls, params: dict[str, Any]) -> bool:
        return params["buy_below"] < params["exit_above"]

    def warmup_bars(self) -> int:
        return int(self.params["trend_sma"]) + 1

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        out: list[Signal] = []
        need = self.warmup_bars()
        for sym in symbols:
            bars = view.bars(sym, n=need)
            if len(bars) < need:
                continue
            close = bars["close"]
            rsi_val = float(rsi(close, self.params["rsi_n"]).iloc[-1])
            trend_val = float(sma(close, self.params["trend_sma"]).iloc[-1])
            price = float(close.iloc[-1])
            in_uptrend = price > trend_val

            # State logic, expressed from indicators alone (stateless):
            # below entry threshold in an uptrend -> long; above exit threshold
            # or broken trend -> flat; in between -> hold whatever the engine
            # holds (signal 'long' only while the entry condition region lasts).
            if in_uptrend and rsi_val < self.params["buy_below"]:
                state = "long"
            elif rsi_val > self.params["exit_above"] or not in_uptrend:
                state = "flat"
            else:
                # neutral zone: keep current exposure; engine treats a missing
                # signal as "no change requested"
                continue
            out.append(
                Signal(
                    sym, state,
                    {"rsi": round(rsi_val, 2), "trend_sma": round(trend_val, 4), "close": price},
                )
            )
        return out
