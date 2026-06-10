"""Reference strategy: time-series momentum (TSMOM).

Hypothesis shape: assets with positive trailing returns tend to keep them over
the next weeks - one of the most extensively documented anomalies in the
academic literature (Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum").

Long a symbol when its trailing `lookback`-bar return (optionally skipping the
most recent `skip` bars to sidestep short-term reversal) is positive; flat
otherwise. Long-only by platform rule.

This is deliberately the simplest honest formulation. It exists so research
can start from a hypothesis with a real economic literature behind it instead
of pure pattern-mining - the validation gates still apply in full.
"""

from __future__ import annotations

from typing import Any

from quantlab.data.pit import PITView
from quantlab.strategies.contract import Signal, Strategy
from quantlab.strategies.factory import register


@register
class TimeSeriesMomentum(Strategy):
    name = "tsmom"
    version = 1

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {"lookback": [63, 126, 252], "skip": [0, 21]}

    def warmup_bars(self) -> int:
        return int(self.params["lookback"]) + int(self.params["skip"]) + 2

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        lookback, skip = int(self.params["lookback"]), int(self.params["skip"])
        need = self.warmup_bars()
        out: list[Signal] = []
        for sym in symbols:
            bars = view.bars(sym, n=need)
            if len(bars) < need:
                continue
            close = bars["close"]
            recent = float(close.iloc[-1 - skip])
            past = float(close.iloc[-1 - skip - lookback])
            momentum = recent / past - 1.0
            out.append(
                Signal(
                    sym,
                    "long" if momentum > 0 else "flat",
                    {"momentum": round(momentum, 6), "lookback": lookback, "skip": skip},
                )
            )
        return out
