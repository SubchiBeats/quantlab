"""Positive control: prove the validation gates CAN issue a PASS.

The negative control (test_experiment_e2e) proves the platform rejects a random
strategy. This is its mirror image: a strategy with a GENUINE, regime-robust
edge on data engineered to contain one must be ACCEPTED. Together the two
controls bracket the system - it says 'no' to noise and 'yes' to real signal -
which is the only way to trust a 'fail' verdict on real data.

The edge here is honest mean reversion: each instrument is an Ornstein-Uhlenbeck
process that genuinely reverts toward a slowly rising level, so buying dips
below a moving average has real positive expectancy. The benchmark is an
INDEPENDENT random walk, so the strategy's profits (driven by each instrument's
own oscillation) spread across all benchmark regimes - which is what lets a
real edge clear the regime-concentration gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quantlab.strategies  # noqa: F401 - registers the catalog
from quantlab.data.pit import MarketData, PITView
from quantlab.data.store import create_snapshot, write_curated
from quantlab.experiments.runner import run_experiment
from quantlab.foundation.clock import utc_now_iso
from quantlab.strategies.contract import Signal, Strategy
from quantlab.strategies.factory import register

DIPBUYER_YAML = """\
name: dip_buyer
version: 1
universe: [OU1, OU2, OU3, OU4]
params:
  sma_n: 20
  entry_k: 4
"""


@register
class DipBuyer(Strategy):
    """Long when price is entry_k% below its sma_n average (oversold), flat when
    it recovers above the average. Captures mean reversion across regimes."""

    name = "dip_buyer"
    version = 1

    @classmethod
    def param_space(cls) -> dict:
        return {"sma_n": [15, 20, 25], "entry_k": [3, 4, 5]}

    def warmup_bars(self) -> int:
        return int(self.params["sma_n"]) + 1

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        out = []
        need = self.warmup_bars()
        for sym in symbols:
            bars = view.bars(sym, n=need)
            if len(bars) < need:
                continue
            close = bars["close"]
            sma = float(close.iloc[-int(self.params["sma_n"]):].mean())
            price = float(close.iloc[-1])
            below = (price / sma - 1.0) * 100.0
            if below < -float(self.params["entry_k"]):
                out.append(Signal(sym, "long", {"pct_below_sma": round(below, 2)}))
            elif price > sma:
                out.append(Signal(sym, "flat", {"pct_below_sma": round(below, 2)}))
        return out


def _ou_series(symbol: str, n: int, seed: int, theta: float = 0.07,
               sigma: float = 0.02, drift: float = 0.0002) -> pd.DataFrame:
    """Ornstein-Uhlenbeck log-price reverting to a slowly rising level."""
    rng = np.random.default_rng(seed)
    mu = np.cumsum(np.full(n, drift)) + 4.6  # rising mean of log-price (~100 start)
    x = np.empty(n)
    x[0] = mu[0]
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu[t] - x[t - 1]) + sigma * rng.standard_normal()
    close = np.exp(x)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + sigma * 0.3 * rng.standard_normal(n - 1))
    span = np.abs(sigma * rng.standard_normal(n)) * close
    high = np.maximum(open_, close) + span * 0.5
    low = np.minimum(open_, close) - span * 0.5
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1_000_000, 3_000_000, n)},
        index=pd.Index(idx, name="date"),
    ).round(4)


def _independent_benchmark(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.003, "low": close * 0.997, "close": close,
         "volume": rng.integers(1_000_000, 3_000_000, n)},
        index=pd.Index(idx, name="date"),
    ).round(4)


@pytest.fixture
def ou_workspace(workspace):
    # strategy config + data config pointing at the OU universe
    (workspace.strategy_configs / "dip_buyer@1.yaml").write_text(DIPBUYER_YAML, encoding="utf-8")
    data_yaml = (workspace.configs / "data.yaml").read_text()
    data_yaml = data_yaml.replace("benchmark_symbol: BENCH", "benchmark_symbol: BENCHI")
    data_yaml = data_yaml.replace("universe: [SYN1, SYN2, SYN3]",
                                  "universe: [OU1, OU2, OU3, OU4]")
    (workspace.configs / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return workspace


def test_positive_control_genuine_edge_passes(ou_workspace, conn):
    frames = {f"OU{i}": _ou_series(f"OU{i}", 2600, seed=100 + i) for i in range(1, 5)}
    frames["BENCHI"] = _independent_benchmark(2600, seed=200)
    market = MarketData(frames)

    for sym in market.symbols:
        df = market.frame(sym)
        write_curated(ou_workspace, sym, df)
        conn.execute(
            "INSERT INTO instruments(symbol, asset_class, source, first_bar, last_bar,"
            " is_benchmark, created_at) VALUES (?,?,?,?,?,?,?)",
            (sym, "equity", "synthetic", df.index[0].strftime("%Y-%m-%d"),
             df.index[-1].strftime("%Y-%m-%d"), int(sym == "BENCHI"), utc_now_iso()),
        )
    conn.commit()
    snapshot_id = create_snapshot(ou_workspace, conn, market.symbols, "OU positive control")

    from quantlab.vault.vault import register_hypothesis
    h = register_hypothesis(
        conn, "positive-control", "Buying oversold dips on mean-reverting series is profitable",
        "engineered OU mean reversion (system validation, not a market claim)",
        "positive OOS expectancy and gates pass", "any statistical gate fails",
    )
    summary = run_experiment(ou_workspace, conn, h, "dip_buyer", 1, snapshot_id)

    # The whole point: a genuine, regime-robust edge must be ACCEPTED.
    assert summary["verdict"] == "pass", (
        f"gates rejected a genuine edge: {summary['reasons']}"
    )
