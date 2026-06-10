"""Monte Carlo: determinism under seeds, percentile ordering, stress monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.validation.montecarlo import block_bootstrap, trade_bootstrap


@pytest.fixture
def pnls() -> list[float]:
    rng = np.random.default_rng(123)
    return list(rng.normal(20.0, 100.0, 80))


def test_trade_bootstrap_deterministic(pnls):
    a = trade_bootstrap(pnls, 100_000.0, 500, seed=42, ruin_drawdown_pct=15.0)
    b = trade_bootstrap(pnls, 100_000.0, 500, seed=42, ruin_drawdown_pct=15.0)
    assert a.stats == b.stats
    c = trade_bootstrap(pnls, 100_000.0, 500, seed=43, ruin_drawdown_pct=15.0)
    assert a.stats != c.stats


def test_trade_bootstrap_percentile_ordering(pnls):
    s = trade_bootstrap(pnls, 100_000.0, 500, seed=42, ruin_drawdown_pct=15.0).stats
    # deeper percentiles are more negative
    assert s["dd_p99"] <= s["dd_p95"] <= s["dd_p50"] <= 0
    assert 0.0 <= s["risk_of_ruin"] <= 1.0


def test_trade_bootstrap_no_trades():
    s = trade_bootstrap([], 100_000.0, 100, seed=1, ruin_drawdown_pct=15.0)
    assert "error" in s.stats


def test_block_bootstrap_deterministic_and_block_len():
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0004, 0.012, 400))
    a = block_bootstrap(rets, 300, seed=5, ruin_drawdown_pct=15.0)
    b = block_bootstrap(rets, 300, seed=5, ruin_drawdown_pct=15.0)
    assert a.stats == b.stats
    assert a.stats["block_len"] == int(np.ceil(np.sqrt(400)))
    assert a.stats["dd_p99"] <= a.stats["dd_p50"] <= 0


def test_block_bootstrap_too_short():
    s = block_bootstrap(pd.Series([0.01] * 10), 100, seed=1, ruin_drawdown_pct=15.0)
    assert "error" in s.stats


def test_cost_stress_monotone_on_real_engine(workspace, trending_market):
    """Higher costs can never IMPROVE results when signals are cost-independent."""
    import quantlab.strategies  # noqa: F401

    from quantlab.backtest.engine import run_backtest
    from quantlab.foundation.config import load_app_config
    from quantlab.strategies.factory import create
    from quantlab.validation.montecarlo import cost_stress

    cfg = load_app_config(workspace.configs, "sma_cross", 1)
    strat_params = cfg.strategy.params

    def rerun(mult: float):
        return run_backtest(trending_market, create("sma_cross", 1, strat_params),
                            cfg.backtest, cfg.strategy.universe, cost_multiplier=mult)

    summary = cost_stress(rerun, [1.0, 2.0])
    table = {row["multiplier"]: row["total_return"] for row in summary.stats["table"]}
    assert table[2.0] <= table[1.0] + 1e-9
