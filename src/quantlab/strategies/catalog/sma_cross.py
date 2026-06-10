"""Reference strategy: SMA crossover trend-following.

Hypothesis shape: persistent trends exist because information diffuses slowly;
hold a symbol while its fast moving average is above its slow moving average.

Long when SMA(fast) > SMA(slow), flat otherwise. Long-only by platform rule.
"""

from __future__ import annotations

from typing import Any

from quantlab.data.pit import PITView
from quantlab.strategies.contract import Signal, Strategy
from quantlab.strategies.factory import register


@register
class SmaCross(Strategy):
    name = "sma_cross"
    version = 1

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {"fast": [10, 20, 30], "slow": [50, 100, 200]}

    @classmethod
    def params_valid(cls, params: dict[str, Any]) -> bool:
        return params["fast"] < params["slow"]

    def warmup_bars(self) -> int:
        return int(self.params["slow"]) + 1

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        out: list[Signal] = []
        need = self.warmup_bars()
        for sym in symbols:
            bars = view.bars(sym, n=need)
            if len(bars) < need:
                continue  # not enough history yet: no opinion
            close = bars["close"]
            # tail means == SMA(n).iloc[-1] for a simple moving average, but
            # avoid recomputing the full rolling series every bar
            fast_val = float(close.iloc[-self.params["fast"]:].mean())
            slow_val = float(close.iloc[-self.params["slow"]:].mean())
            state = "long" if fast_val > slow_val else "flat"
            out.append(
                Signal(sym, state, {"sma_fast": round(fast_val, 4), "sma_slow": round(slow_val, 4)})
            )
        return out
