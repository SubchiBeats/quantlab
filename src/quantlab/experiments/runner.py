"""Experiment engine: hypothesis -> full validation battery -> recorded verdict.

One experiment executes, in order:
  1. full-sample backtest (reference + the base for stress re-runs)
  2. benchmark buy-and-hold on the same snapshot and window
  3. walk-forward validation (the evidentiary out-of-sample result)
  4. Monte Carlo battery: trade bootstrap + block bootstrap on OOS,
     cost/delay stress and parameter perturbation as engine re-runs
  5. regime analysis against the benchmark
  6. deflated Sharpe with trials = this experiment's grid x folds PLUS all
     prior trials in the hypothesis family (from the vault)
  7. gate evaluation -> verdict (pass/fail) written immutably to the vault

Every run records config hash, git SHA (+dirty flag), seed, and engine version;
artifacts (equity curves, trades, gate report) land in artifacts/E<id>/.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab import ENGINE_VERSION
from quantlab.backtest.engine import BacktestResult, run_backtest
from quantlab.data.pit import MarketData
from quantlab.data.store import load_snapshot
from quantlab.foundation.audit import append_event
from quantlab.foundation.clock import utc_now_iso
from quantlab.foundation.config import AppConfig, config_hash, load_app_config
from quantlab.foundation.logs import get_logger
from quantlab.foundation.paths import Paths
from quantlab.foundation.rng import derive_seed
from quantlab.metrics.benchmark import buy_hold_equity
from quantlab.metrics.core import benchmark_comparison, compute_metrics, sharpe, trade_stats
from quantlab.strategies.factory import get_class
from quantlab.validation.gates import deflated_sharpe, evaluate_gates, label_regimes, regime_pnl
from quantlab.validation.montecarlo import (
    block_bootstrap,
    cost_stress,
    delay_stress,
    param_perturbation,
    trade_bootstrap,
)
from quantlab.validation.walkforward import run_walkforward
from quantlab.vault import vault

log = get_logger("experiments")


def _git_info(root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nogit", True
    if sha.returncode != 0:
        return "nogit", True
    return sha.stdout.strip(), bool(status.stdout.strip())


def _run_id(experiment_id: int, run_type: str, cfg_hash: str, started: str) -> str:
    digest = hashlib.sha256(f"{experiment_id}|{run_type}|{cfg_hash}|{started}".encode()).hexdigest()
    return f"R{experiment_id}-{run_type}-{digest[:8]}"


def run_experiment(
    paths: Paths,
    conn: sqlite3.Connection,
    hypothesis_id: int,
    strategy_name: str,
    strategy_version: int,
    snapshot_id: str,
    ack_journal_id: int | None = None,
) -> dict[str, Any]:
    cfg: AppConfig = load_app_config(paths.configs, strategy_name, strategy_version)
    cfg_hash = config_hash(cfg)
    git_sha, dirty = _git_info(paths.root)
    benchmark_symbol = cfg.data.benchmark_symbol.upper()

    # -- register (includes the similar-failures check; may raise) --
    experiment_id, similar = vault.register_experiment(
        conn, hypothesis_id, strategy_name, strategy_version, snapshot_id,
        benchmark_symbol, ack_journal_id,
    )
    vault.mark_running(conn, experiment_id)
    log.info("experiment %s registered (%d similar prior failures acknowledged)",
             experiment_id, len(similar))

    artifact_dir = paths.artifacts / f"E{experiment_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    art_rel = str(artifact_dir.relative_to(paths.root))

    # -- load data, sanity-check coverage --
    frames = load_snapshot(paths, conn, snapshot_id)
    missing = [s for s in [*cfg.strategy.universe, benchmark_symbol]
               if s.upper() not in frames]
    if missing:
        raise KeyError(f"snapshot {snapshot_id} lacks symbols required by config: {missing}")
    market = MarketData(frames)
    strategy_cls = get_class(strategy_name, strategy_version)
    universe = cfg.strategy.universe
    root_seed = cfg.global_.root_seed

    def record(run_type: str, started: str, seed: int) -> str:
        rid = _run_id(experiment_id, run_type, cfg_hash, started)
        vault.record_run(conn, rid, experiment_id, run_type, cfg_hash, git_sha, dirty,
                         seed, ENGINE_VERSION, started, art_rel)
        return rid

    # ---- 1. full-sample backtest ----
    started = utc_now_iso()
    strategy = strategy_cls(cfg.strategy.params)
    full = run_backtest(market, strategy, cfg.backtest, universe)
    full_metrics = compute_metrics(full)
    rid_full = record("backtest", started, 0)
    vault.record_metrics(conn, rid_full, full_metrics, "full")
    vault.record_trades(conn, rid_full, full.trades)
    full.equity.to_parquet(artifact_dir / "equity_full.parquet")
    full.trades_frame().to_csv(artifact_dir / "trades_full.csv", index=False)

    # ---- 2. benchmark buy-and-hold over the same window ----
    started = utc_now_iso()
    bench_eq = buy_hold_equity(
        market, benchmark_symbol, cfg.backtest.initial_capital, cfg.backtest.costs,
        start=full.equity.index[0], end=full.equity.index[-1],
    )
    bench_rets = bench_eq.pct_change().dropna()
    from quantlab.metrics.core import cagr as cagr_fn, max_drawdown as mdd_fn
    bench_mdd, _ = mdd_fn(bench_eq)
    bench_metrics = {
        "total_return": float(bench_eq.iloc[-1] / bench_eq.iloc[0] - 1.0),
        "cagr": cagr_fn(bench_eq),
        "sharpe": sharpe(bench_rets),
        "max_drawdown": bench_mdd,
    }
    rid_bench = record("benchmark", started, 0)
    vault.record_metrics(conn, rid_bench, bench_metrics, "full")
    bench_eq.to_frame().to_parquet(artifact_dir / "equity_benchmark.parquet")
    comparison = benchmark_comparison(full.daily_returns, bench_rets)
    vault.record_metrics(conn, rid_full, comparison, "full")

    # ---- 3. walk-forward ----
    started = utc_now_iso()
    wf = run_walkforward(market, cfg)
    oos_trade_metrics = trade_stats(wf.oos_trades, cfg.backtest.initial_capital)
    oos_equity = cfg.backtest.initial_capital * (1.0 + wf.oos_returns).cumprod()
    from quantlab.metrics.core import sortino as sortino_fn
    oos_mdd, oos_underwater = mdd_fn(oos_equity) if len(oos_equity) > 1 else (float("nan"), 0)
    oos_metrics: dict[str, float] = {
        "sharpe": wf.oos_sharpe,
        "sortino": sortino_fn(wf.oos_returns),
        "total_return": float((1.0 + wf.oos_returns).prod() - 1.0) if len(wf.oos_returns) else float("nan"),
        "max_drawdown": oos_mdd,
        "longest_underwater_bars": float(oos_underwater),
        "wf_efficiency": wf.wf_efficiency,
        "is_sharpe_mean": wf.is_sharpe_mean,
        "param_stability": wf.param_stability,
        "n_trials": float(wf.n_trials),
        "n_configs": float(wf.n_configs),
        "n_folds": float(len(wf.folds)),
    }
    oos_metrics.update(oos_trade_metrics)
    rid_wf = record("walkforward", started, 0)
    vault.record_metrics(conn, rid_wf, oos_metrics, "oos")
    vault.record_folds(conn, rid_wf, wf.folds)
    vault.record_trades(conn, rid_wf, wf.oos_trades)
    if len(wf.oos_returns):
        wf.oos_returns.rename("oos_return").to_frame().to_parquet(artifact_dir / "oos_returns.parquet")

    # ---- 4. Monte Carlo battery ----
    started = utc_now_iso()
    mc_cfg = cfg.validation.montecarlo
    rid_mc = record("montecarlo", started, root_seed)

    mc_trade = trade_bootstrap(
        [t.pnl for t in wf.oos_trades], cfg.backtest.initial_capital, mc_cfg.n_paths,
        derive_seed(root_seed, "mc.trade", str(experiment_id)), mc_cfg.ruin_drawdown_pct,
    )
    vault.record_mc(conn, rid_mc, mc_trade)

    mc_block = block_bootstrap(
        wf.oos_returns, mc_cfg.n_paths,
        derive_seed(root_seed, "mc.block", str(experiment_id)), mc_cfg.ruin_drawdown_pct,
        None if mc_cfg.block_len == "auto" else int(mc_cfg.block_len),
    )
    vault.record_mc(conn, rid_mc, mc_block)

    def rerun_cost(mult: float) -> BacktestResult:
        return run_backtest(market, strategy_cls(cfg.strategy.params), cfg.backtest,
                            universe, cost_multiplier=mult)

    mc_cost = cost_stress(rerun_cost, mc_cfg.cost_stress_multipliers)
    vault.record_mc(conn, rid_mc, mc_cost)

    def rerun_delay(extra: int) -> BacktestResult:
        return run_backtest(market, strategy_cls(cfg.strategy.params), cfg.backtest,
                            universe, entry_delay_bars=extra)

    mc_delay = delay_stress(rerun_delay, [0, *mc_cfg.delay_stress_bars])
    vault.record_mc(conn, rid_mc, mc_delay)

    def rerun_params(combo: dict[str, Any]) -> BacktestResult:
        return run_backtest(market, strategy_cls(combo), cfg.backtest, universe)

    mc_perturb = param_perturbation(
        rerun_params, dict(cfg.strategy.params), strategy_cls.grid(), full_metrics["sharpe"],
        mc_cfg.perturbation_samples, derive_seed(root_seed, "mc.perturb", str(experiment_id)),
    )
    vault.record_mc(conn, rid_mc, mc_perturb)

    # ---- 5. regimes (benchmark-derived, OOS trades) ----
    regimes = label_regimes(market.frame(benchmark_symbol)["close"])
    regime_table = regime_pnl(wf.oos_trades, regimes)

    # ---- 6. deflated Sharpe with family accounting ----
    # N for the deflated Sharpe = number of DISTINCT strategy configurations
    # selected among, accumulated across this hypothesis family. Walk-forward
    # re-evaluates the same configs on many rolling windows; those folds are a
    # validation method, NOT additional hypotheses, so they must not multiply N.
    hyp = conn.execute(
        "SELECT family FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
    ).fetchone()
    prior_configs = vault.family_prior_configs(conn, hyp["family"], exclude_experiment=experiment_id)
    dsr = deflated_sharpe(wf.oos_returns, wf.n_configs + prior_configs, wf.trial_sharpes)

    # ---- 7. gates -> verdict ----
    report = evaluate_gates(
        cfg.validation.gates,
        oos_metrics,
        wf.wf_efficiency,
        dsr,
        float(mc_cost.stats.get("max_profitable_multiplier", 0.0)),
        list(mc_delay.stats.get("table", [])),
        regime_table,
        float(mc_perturb.stats.get("neighbor_sharpe_ratio", 0.0)),
    )
    reason = "; ".join(report.reasons) if report.reasons else "all gates passed"
    vault.finalize_experiment(conn, experiment_id, report.verdict, reason)

    gates_payload = {
        "experiment_id": experiment_id,
        "pessimism_multiplier": cfg.backtest.costs.pessimism_multiplier,
        "verdict": report.verdict,
        "reasons": report.reasons,
        "weaknesses": report.weaknesses,
        "dsr": dsr,
        "dsr_n_configs": wf.n_configs + prior_configs,
        "prior_family_trials": prior_configs,
        "regimes": regime_table,
        "gates": [
            {"name": g.name, "passed": g.passed, "measured": g.measured,
             "threshold": g.threshold, "detail": g.detail}
            for g in report.gates
        ],
    }
    (artifact_dir / "gates.json").write_text(
        json.dumps(gates_payload, indent=2, default=str), encoding="utf-8"
    )
    append_event(conn, "system", "experiment.complete",
                 {"experiment_id": experiment_id, "verdict": report.verdict})

    return {
        "experiment_id": experiment_id,
        "verdict": report.verdict,
        "reasons": report.reasons,
        "weaknesses": report.weaknesses,
        "config_hash": cfg_hash,
        "git_sha": git_sha,
        "dirty_tree": dirty,
        "full_metrics": full_metrics,
        "oos_metrics": oos_metrics,
        "benchmark_metrics": bench_metrics,
        "artifact_dir": str(artifact_dir),
    }
