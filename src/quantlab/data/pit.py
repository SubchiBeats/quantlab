"""Point-in-time (PIT) market data access.

This is the ONLY way strategies see data, and it is the platform's structural
defense against lookahead bias: a PITView is bound to an as-of date and can
physically only return bars at or before that date. There is no method that
returns the full history, so strategy code cannot accidentally peek ahead.

MarketData precomputes positional indices per symbol so the per-bar slicing in
the backtest hot loop is O(1) lookup + cheap tail slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketData:
    """Immutable container for a snapshot's bars across symbols."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not frames:
            raise ValueError("MarketData needs at least one symbol")
        self._frames: dict[str, pd.DataFrame] = {}
        for sym, df in frames.items():
            f = df.sort_index()
            f.index = pd.DatetimeIndex(f.index)
            self._frames[sym.upper()] = f
        all_dates = sorted(set().union(*(set(f.index) for f in self._frames.values())))
        self.calendar: pd.DatetimeIndex = pd.DatetimeIndex(all_dates)

    @property
    def symbols(self) -> list[str]:
        return sorted(self._frames)

    def frame(self, symbol: str) -> pd.DataFrame:
        """Full frame access - for the ENGINE and validation layers only.

        Strategies receive PITView, never MarketData. (Phase 2 adds an
        import-linter contract; in Phase 1 the engine simply never passes
        MarketData into strategy code.)
        """
        return self._frames[symbol.upper()]

    def view(self, as_of: pd.Timestamp) -> "PITView":
        return PITView(self, pd.Timestamp(as_of))


class PITView:
    """A read-only window onto MarketData ending at `as_of` (inclusive)."""

    def __init__(self, market: MarketData, as_of: pd.Timestamp):
        self._market = market
        self.as_of = as_of

    def bars(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Bars for `symbol` up to and including as_of; last `n` if given.

        Returns a copy so strategy code cannot mutate stored data.
        """
        frame = self._market.frame(symbol)
        # searchsorted on the right edge: strictly no rows after as_of
        end = int(np.searchsorted(frame.index.values, self.as_of.to_datetime64(), side="right"))
        start = 0 if n is None else max(0, end - n)
        return frame.iloc[start:end].copy()

    def last_close(self, symbol: str) -> float | None:
        bars = self.bars(symbol, n=1)
        return float(bars["close"].iloc[-1]) if len(bars) else None
