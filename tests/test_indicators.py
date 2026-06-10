"""Indicator contract tests: known values, warm-up NaNs, bounds, and the
no-lookahead property (perturbing the future must not change the past)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.indicators import atr, ema, rsi, sma
from quantlab.indicators.core import true_range


def _series(seed: int, n: int = 120) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def test_sma_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_constant_series():
    s = pd.Series([7.0] * 50)
    out = sma(s, 10)
    assert out.iloc[9:].eq(7.0).all()
    assert out.iloc[:9].isna().all()


def test_ema_warmup_and_convergence():
    s = pd.Series([5.0] * 60)
    out = ema(s, 12)
    assert out.iloc[:11].isna().all()
    assert out.iloc[-1] == pytest.approx(5.0)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.linspace(200, 100, 60))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    flat = pd.Series([100.0] * 60)
    assert rsi(flat, 14).iloc[-1] == pytest.approx(50.0)
    noisy = _series(3)
    vals = rsi(noisy, 14).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_atr_positive_after_warmup():
    close = _series(5)
    high, low = close * 1.01, close * 0.99
    out = atr(high, low, close, 14)
    assert out.iloc[:13].isna().all()
    assert (out.iloc[14:] > 0).all()


def test_true_range_first_bar_nan():
    close = _series(7)
    tr = true_range(close * 1.01, close * 0.99, close)
    assert np.isnan(tr.iloc[0])


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), cut=st.integers(30, 90))
def test_no_lookahead_property(seed: int, cut: int):
    """The platform's core indicator guarantee: values up to t are identical
    when everything after t is replaced with noise."""
    close = _series(seed)
    rng = np.random.default_rng(seed + 1)
    perturbed = close.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:].to_numpy() * rng.uniform(0.5, 1.5, len(close) - cut)

    for fn in (lambda s: sma(s, 10), lambda s: ema(s, 10), lambda s: rsi(s, 5)):
        a, b = fn(close), fn(perturbed)
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])

    high, low = close * 1.01, close * 0.99
    ph, pl = perturbed * 1.01, perturbed * 0.99
    pd.testing.assert_series_equal(
        atr(high, low, close, 5).iloc[:cut], atr(ph, pl, perturbed, 5).iloc[:cut]
    )


def test_indicators_do_not_mutate_input():
    s = _series(11)
    copy = s.copy()
    sma(s, 10), ema(s, 10), rsi(s, 5)
    pd.testing.assert_series_equal(s, copy)
