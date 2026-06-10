"""DEVELOPMENT-ONLY synthetic market data.

Generates seeded geometric-Brownian-motion daily OHLCV with regime shifts in
drift/volatility. Purpose: developing and testing the platform without network
access, and negative-control tests (a validation suite must reject strategies
on data with no exploitable structure).

Synthetic symbols are registered with source='synthetic' and are clearly
labeled in every report. NEVER base research conclusions on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.foundation.rng import make_rng


def generate_daily(
    symbol: str,
    n_bars: int,
    seed: int,
    start: str = "2015-01-02",
    s0: float = 100.0,
    trend: float = 0.05,
) -> pd.DataFrame:
    """Seeded GBM with three alternating drift/vol regimes; business-day index.

    `trend` is the average annualized drift; regimes swing around it so the
    series has bull/bear/sideways stretches like real data.
    """
    rng = make_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars)

    # Regime blocks of ~6 months with drift/vol drawn around the base trend.
    drifts = np.empty(n_bars)
    vols = np.empty(n_bars)
    i = 0
    while i < n_bars:
        block = int(rng.integers(90, 160))
        drifts[i : i + block] = trend + rng.normal(0.0, 0.15)
        vols[i : i + block] = float(np.clip(rng.normal(0.18, 0.06), 0.08, 0.45))
        i += block

    dt = 1.0 / 252.0
    rets = (drifts - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * rng.standard_normal(n_bars)
    close = s0 * np.exp(np.cumsum(rets))

    open_ = np.empty(n_bars)
    open_[0] = s0
    # Open gaps a fraction of daily vol away from the prior close.
    open_[1:] = close[:-1] * np.exp(vols[1:] * np.sqrt(dt) * 0.3 * rng.standard_normal(n_bars - 1))
    intraday = np.abs(vols * np.sqrt(dt) * rng.standard_normal(n_bars)) * close
    high = np.maximum(open_, close) + intraday * 0.5
    low = np.minimum(open_, close) - intraday * 0.5
    low = np.maximum(low, 0.01)
    volume = rng.integers(500_000, 5_000_000, n_bars)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.Index(dates, name="date"),
    )
    return df.round({"open": 4, "high": 4, "low": 4, "close": 4})
