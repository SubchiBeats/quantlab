"""Shared fixtures: temp workspace with configs, database, synthetic market data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from quantlab.data.pit import MarketData
from quantlab.data.synth import generate_daily
from quantlab.foundation.db import open_db
from quantlab.foundation.paths import Paths
from quantlab.foundation.rng import derive_seed

ROOT_SEED = 777

GLOBAL_YAML = f"""\
timezone: UTC
log_level: WARNING
root_seed: {ROOT_SEED}
"""

DATA_YAML = """\
frequency: daily
benchmark_symbol: BENCH
universe: [SYN1, SYN2, SYN3]
validation:
  max_consecutive_gap_days: 5
  spike_zscore: 8.0
  min_avg_volume: 10000
  min_bars: 300
"""

BACKTEST_YAML = """\
initial_capital: 100000.0
fill_policy: next_bar_open
cash_account: true
allow_short: false
costs:
  commission_per_share: 0.005
  min_commission: 1.0
  half_spread_bps: 2.5
  impact_coeff: 0.1
  pessimism_multiplier: 1.5
sizing:
  risk_per_trade_pct: 0.5
  atr_period: 14
  atr_stop_mult: 3.0
  max_position_pct: 10.0
"""

VALIDATION_YAML = """\
walkforward:
  scheme: rolling
  train_bars: 350
  test_bars: 150
  embargo_bars: auto
montecarlo:
  n_paths: 200
  block_len: auto
  ruin_drawdown_pct: 15.0
  cost_stress_multipliers: [1.0, 1.5, 2.0]
  delay_stress_bars: [1]
  perturbation_samples: 4
gates:
  min_oos_trades: 30
  min_wf_efficiency: 0.5
  dsr_min: 0.0
  cost_breakeven_min: 1.5
  require_delay_survival: true
  min_positive_regimes: 2
  max_single_regime_pnl_share: 0.8
  min_neighbor_sharpe_ratio: 0.25
"""

SMA_YAML = """\
name: sma_cross
version: 1
universe: [SYN1, SYN2, SYN3]
params:
  fast: 20
  slow: 100
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Paths:
    paths = Paths(tmp_path).ensure()
    (paths.configs / "global.yaml").write_text(GLOBAL_YAML, encoding="utf-8")
    (paths.configs / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    (paths.configs / "backtest.yaml").write_text(BACKTEST_YAML, encoding="utf-8")
    (paths.configs / "validation.yaml").write_text(VALIDATION_YAML, encoding="utf-8")
    (paths.strategy_configs / "sma_cross@1.yaml").write_text(SMA_YAML, encoding="utf-8")
    return paths


@pytest.fixture
def conn(workspace: Paths) -> sqlite3.Connection:
    connection = open_db(workspace.db_file)
    yield connection
    connection.close()


def make_market(trend: float, n_bars: int = 1200, symbols: tuple[str, ...] = ("SYN1", "SYN2", "SYN3")) -> MarketData:
    frames = {
        sym: generate_daily(sym, n_bars, derive_seed(ROOT_SEED, "synth", sym), trend=trend)
        for sym in symbols
    }
    frames["BENCH"] = generate_daily("BENCH", n_bars, derive_seed(ROOT_SEED, "synth", "BENCH"),
                                     trend=max(trend, 0.04))
    return MarketData(frames)


@pytest.fixture
def trending_market() -> MarketData:
    return make_market(trend=0.10)


@pytest.fixture
def driftless_market() -> MarketData:
    return make_market(trend=0.0)
