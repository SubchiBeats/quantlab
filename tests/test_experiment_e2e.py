"""End-to-end: the full experiment pipeline on synthetic data, including THE
critical platform test - the negative control. A validation suite that cannot
reject a random strategy is decoration; this test fails the build if the
pipeline ever loses the ability to say no.
"""

from __future__ import annotations

import hashlib
import json

import pytest

import quantlab.strategies  # noqa: F401 - registers the catalog
from quantlab.data.pit import MarketData, PITView
from quantlab.data.store import create_snapshot, write_curated
from quantlab.experiments.runner import run_experiment
from quantlab.foundation.clock import utc_now_iso
from quantlab.strategies.contract import Signal, Strategy
from quantlab.strategies.factory import register
from quantlab.vault.vault import register_hypothesis

RANDOM_YAML = """\
name: random_walk
version: 1
universe: [SYN1, SYN2, SYN3]
params:
  p: 1
"""


@register
class RandomWalkStrategy(Strategy):
    """Deterministic pseudo-random long/flat signals - pure noise by design."""

    name = "random_walk"
    version = 1

    @classmethod
    def param_space(cls) -> dict:
        return {"p": [1, 2, 3]}

    def warmup_bars(self) -> int:
        return 20

    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        out = []
        for sym in symbols:
            digest = hashlib.sha256(
                f"{self.params['p']}|{sym}|{view.as_of.date()}".encode()
            ).digest()
            out.append(Signal(sym, "long" if digest[0] % 2 == 0 else "flat"))
        return out


def _curate_and_snapshot(workspace, conn, market: MarketData) -> str:
    for sym in market.symbols:
        df = market.frame(sym)
        write_curated(workspace, sym, df)
        conn.execute(
            "INSERT INTO instruments(symbol, asset_class, source, first_bar, last_bar,"
            " is_benchmark, created_at) VALUES (?,?,?,?,?,?,?)",
            (sym, "equity", "synthetic", df.index[0].strftime("%Y-%m-%d"),
             df.index[-1].strftime("%Y-%m-%d"), int(sym == "BENCH"), utc_now_iso()),
        )
    conn.commit()
    return create_snapshot(workspace, conn, market.symbols, "e2e test data")


def test_full_pipeline_records_everything_and_reports(workspace, conn, trending_market):
    snapshot_id = _curate_and_snapshot(workspace, conn, trending_market)
    h = register_hypothesis(
        conn, "trend-following", "SMA crossovers capture persistent trends",
        "slow information diffusion", "positive OOS expectancy", "negative OOS expectancy",
    )
    summary = run_experiment(workspace, conn, h, "sma_cross", 1, snapshot_id)

    assert summary["verdict"] in ("pass", "fail")
    e = conn.execute("SELECT * FROM experiments WHERE experiment_id=?",
                     (summary["experiment_id"],)).fetchone()
    assert e["status"] == "complete"
    assert e["verdict"] == summary["verdict"]

    run_types = {r["run_type"] for r in conn.execute(
        "SELECT run_type FROM runs WHERE experiment_id=?", (summary["experiment_id"],))}
    assert run_types == {"backtest", "benchmark", "walkforward", "montecarlo"}

    mc_methods = {r["method"] for r in conn.execute(
        "SELECT method FROM mc_simulations m JOIN runs r ON r.run_id=m.run_id"
        " WHERE r.experiment_id=?", (summary["experiment_id"],))}
    assert mc_methods == {"trade_bootstrap", "block_bootstrap", "cost_stress",
                          "delay_stress", "param_perturbation"}

    artifact_dir = workspace.artifacts / f"E{summary['experiment_id']}"
    gates = json.loads((artifact_dir / "gates.json").read_text(encoding="utf-8"))
    gate_names = {g["name"] for g in gates["gates"]}
    assert {"min_oos_trades", "deflated_sharpe", "cost_breakeven", "delay_survival",
            "positive_regimes", "param_neighborhood"} <= gate_names

    # report generation from vault data only
    from quantlab.reports.tearsheet import build_report

    out = build_report(workspace, conn, summary["experiment_id"])
    html = out.read_text(encoding="utf-8")
    assert "QuantLab Tear Sheet" in html
    assert "SMA crossovers capture persistent trends" in html
    assert "SYNTHETIC" in html  # dev-data banner must be present
    assert summary["config_hash"][:16] in html


def test_negative_control_random_strategy_is_rejected(workspace, conn, driftless_market):
    """THE platform test: pure noise must fail the gates."""
    (workspace.strategy_configs / "random_walk@1.yaml").write_text(RANDOM_YAML, encoding="utf-8")
    snapshot_id = _curate_and_snapshot(workspace, conn, driftless_market)
    h = register_hypothesis(
        conn, "negative-control", "random coin-flip entries make money",
        "none - this is the control", "n/a", "any gate failure",
    )
    summary = run_experiment(workspace, conn, h, "random_walk", 1, snapshot_id)

    assert summary["verdict"] == "fail", "validation suite accepted a RANDOM strategy"
    artifact_dir = workspace.artifacts / f"E{summary['experiment_id']}"
    gates = json.loads((artifact_dir / "gates.json").read_text(encoding="utf-8"))
    failed = {g["name"] for g in gates["gates"] if not g["passed"]}
    statistical = {"oos_expectancy_positive", "oos_sharpe_positive", "deflated_sharpe",
                   "wf_efficiency", "cost_breakeven", "delay_survival"}
    assert failed & statistical, f"expected a statistical gate to fail, got: {failed}"


def test_full_sample_backtest_is_deterministic(workspace, trending_market):
    import pandas as pd

    from quantlab.backtest.engine import run_backtest
    from quantlab.foundation.config import load_app_config
    from quantlab.strategies.factory import create

    cfg = load_app_config(workspace.configs, "sma_cross", 1)
    r1 = run_backtest(trending_market, create("sma_cross", 1, cfg.strategy.params),
                      cfg.backtest, cfg.strategy.universe)
    r2 = run_backtest(trending_market, create("sma_cross", 1, cfg.strategy.params),
                      cfg.backtest, cfg.strategy.universe)
    pd.testing.assert_frame_equal(r1.equity, r2.equity)
    assert [(t.symbol, t.entry_ts, t.entry_px, t.qty, t.pnl) for t in r1.trades] == \
           [(t.symbol, t.entry_ts, t.entry_px, t.qty, t.pnl) for t in r2.trades]


def test_experiment_without_hypothesis_refused(workspace, conn, trending_market):
    snapshot_id = _curate_and_snapshot(workspace, conn, trending_market)
    with pytest.raises(KeyError, match="not found"):
        run_experiment(workspace, conn, 999, "sma_cross", 1, snapshot_id)
