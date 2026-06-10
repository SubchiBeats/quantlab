"""Indicator framework.

Contract every indicator obeys (property-tested in tests/test_indicators.py):
- pure function: no mutation of inputs, no state, no I/O
- output index identical to input index
- the first (period-dependent) values are NaN - warm-up is explicit, never
  silently filled
- value at time t depends only on inputs at times <= t (no lookahead);
  verified by the perturbed-future property test
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    if n < 1:
        raise ValueError("sma period must be >= 1")
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (adjust=False recursion), NaN during warm-up."""
    if n < 1:
        raise ValueError("ema period must be >= 1")
    out = series.ewm(span=n, adjust=False, min_periods=n).mean()
    return out


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(close: pd.Series, n: int = 20, annualize: bool = True) -> pd.Series:
    vol = log_returns(close).rolling(n, min_periods=n).std()
    return vol * np.sqrt(252.0) if annualize else vol


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]; NaN for the first n values."""
    if n < 1:
        raise ValueError("rsi period must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing = EMA with alpha 1/n
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # all-gain windows: avg_loss == 0 with gains present -> RSI 100;
    # perfectly flat windows (no gains, no losses) -> neutral 50
    out = out.where(~(avg_loss.eq(0.0) & avg_gain.gt(0.0)), 100.0)
    out = out.where(~(avg_loss.eq(0.0) & avg_gain.eq(0.0)), 50.0)
    out[avg_gain.isna()] = np.nan
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = np.nan  # no previous close on the first bar
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Average True Range; NaN until n true ranges are available."""
    if n < 1:
        raise ValueError("atr period must be >= 1")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
