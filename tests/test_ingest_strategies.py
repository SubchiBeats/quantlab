"""Tiingo payload parsing (offline, no network) and the TSMOM strategy contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.ingest import normalize_frame, parse_tiingo_json
from quantlab.data.pit import MarketData
from quantlab.data.synth import generate_daily
from quantlab.strategies.catalog.tsmom import TimeSeriesMomentum


def _tiingo_rows(n: int = 5) -> list[dict]:
    dates = pd.bdate_range("2024-01-02", periods=n)
    return [
        {
            "date": d.strftime("%Y-%m-%dT00:00:00.000Z"),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,  # unadjusted (ignored)
            "adjOpen": 100.0 + i, "adjHigh": 101.0 + i, "adjLow": 99.0 + i,
            "adjClose": 100.5 + i, "adjVolume": 1_000_000 + i,
            "divCash": 0.0, "splitFactor": 1.0,
        }
        for i, d in enumerate(dates)
    ]


def test_tiingo_parse_uses_adjusted_columns():
    df = parse_tiingo_json(_tiingo_rows(), "TEST")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == pytest.approx(100.5)   # adjClose, not close
    assert df["volume"].dtype == np.int64
    assert df.index[0] == pd.Timestamp("2024-01-02")     # tz stripped, normalized
    assert df.index.is_monotonic_increasing


def test_tiingo_empty_payload_rejected():
    with pytest.raises(ValueError, match="no rows"):
        parse_tiingo_json([], "TEST")


def test_tiingo_missing_adjusted_columns_rejected():
    rows = _tiingo_rows()
    for row in rows:
        del row["adjClose"]
    with pytest.raises(ValueError, match="adjClose"):
        parse_tiingo_json(rows, "TEST")


def test_normalize_rejects_nans():
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=3),
        "open": [1.0, None, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1],
    })
    with pytest.raises(ValueError, match="NaN"):
        normalize_frame(df, "TEST")


# ------------------------------- TSMOM -----------------------------------

def _flat_then_trend(n: int, up: bool) -> pd.DataFrame:
    """300 flat bars then a steady +/-0.2%/bar drift; OHLC kept consistent."""
    drift = 0.002 if up else -0.002
    closes = np.concatenate([np.full(300, 100.0), 100.0 * (1 + drift) ** np.arange(1, n - 299)])
    index = pd.bdate_range("2015-01-02", periods=n)
    df = pd.DataFrame({
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": 1_000_000,
    }, index=pd.Index(index, name="date"))
    return df


def test_tsmom_long_in_uptrend_flat_in_downtrend():
    strat = TimeSeriesMomentum({"lookback": 252, "skip": 21})
    for up, expected in ((True, "long"), (False, "flat")):
        market = MarketData({"SYM": _flat_then_trend(700, up)})
        view = market.view(market.calendar[-1])
        signals = strat.signals(view, ["SYM"])
        assert len(signals) == 1
        assert signals[0].desired_state == expected
        assert "momentum" in signals[0].context


def test_tsmom_silent_until_warmup():
    strat = TimeSeriesMomentum({"lookback": 252, "skip": 21})
    market = MarketData({"SYM": _flat_then_trend(700, True)})
    early = market.view(market.calendar[100])  # < warmup of 275
    assert strat.signals(early, ["SYM"]) == []


def test_tsmom_params_must_come_from_declared_space():
    with pytest.raises(ValueError, match="outside the declared space"):
        TimeSeriesMomentum({"lookback": 200, "skip": 21})


def test_tsmom_no_lookahead():
    base = generate_daily("AAA", 400, seed=77)
    k = 350
    rng = np.random.default_rng(1)
    mutated = base.copy()
    factor = rng.uniform(0.5, 1.5, len(base) - k)
    for col in ("open", "high", "low", "close"):
        mutated.iloc[k:, mutated.columns.get_loc(col)] = (
            mutated.iloc[k:, mutated.columns.get_loc(col)].to_numpy() * factor
        )
    strat = TimeSeriesMomentum({"lookback": 252, "skip": 21})
    as_of = base.index[k - 1]
    s1 = strat.signals(MarketData({"AAA": base}).view(as_of), ["AAA"])
    s2 = strat.signals(MarketData({"AAA": mutated}).view(as_of), ["AAA"])
    assert [(s.desired_state, s.context) for s in s1] == [(s.desired_state, s.context) for s in s2]
