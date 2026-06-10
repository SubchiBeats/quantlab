"""Rejection gates, regime analysis, and the deflated Sharpe ratio.

A strategy's verdict is PASS only if every gate passes. Gates map the five
mandatory rejection causes to measurable tests:

  fails out-of-sample        -> min_oos_trades, OOS expectancy > 0, OOS Sharpe > 0
  relies on overfitting      -> walk-forward efficiency, deflated Sharpe,
                                parameter-neighbor cliff test
  unrealistic execution      -> +1-bar delay stress must stay profitable
  fails after fees/slippage  -> cost-stress breakeven multiplier
  single-regime only         -> >= 2 of 3 benchmark regimes profitable and
                                no regime contributing > max share of P&L

Deflated Sharpe (Bailey & Lopez de Prado 2014): corrects the observed OOS
Sharpe for (a) the number of trials behind it and (b) non-normal returns.
N here counts grid x fold evaluations PLUS prior experiments in the same
hypothesis family from the vault - AI- or human-generated, every attempt
counts. We use stdlib NormalDist; no scipy needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.engine import Trade
from quantlab.foundation.config import GatesConfig

_PHI = NormalDist()


# ---------------------------- regime analysis ----------------------------

def label_regimes(benchmark_close: pd.Series, window: int = 63) -> pd.Series:
    """Label each day down/flat/up by terciles of the benchmark's rolling return."""
    roll = benchmark_close.pct_change(window)
    valid = roll.dropna()
    if len(valid) < 30:
        return pd.Series("flat", index=benchmark_close.index)
    lo, hi = valid.quantile(1 / 3), valid.quantile(2 / 3)
    labels = pd.Series("flat", index=benchmark_close.index)
    labels[roll <= lo] = "down"
    labels[roll >= hi] = "up"
    labels[roll.isna()] = "flat"
    return labels


def regime_pnl(trades: list[Trade], regimes: pd.Series) -> dict[str, dict[str, float]]:
    """Net P&L and trade count per regime; each trade is attributed to the
    regime label on its exit date (when the P&L is realized)."""
    out = {r: {"pnl": 0.0, "n_trades": 0.0} for r in ("down", "flat", "up")}
    for t in trades:
        # exit date may not be in the benchmark index (different calendars):
        # use the most recent label at or before exit
        idx = regimes.index.searchsorted(t.exit_ts, side="right") - 1
        label = str(regimes.iloc[idx]) if idx >= 0 else "flat"
        out[label]["pnl"] += t.pnl
        out[label]["n_trades"] += 1
    return out


# ---------------------------- deflated Sharpe ----------------------------

def deflated_sharpe(
    daily_returns: pd.Series, n_trials: int, trial_sharpes_ann: list[float]
) -> float:
    """Probability that the true Sharpe exceeds the best-of-N-trials noise level.

    Operates on per-period (daily) Sharpe internally; trial Sharpes arrive
    annualized and are de-annualized here. Returns a probability in [0, 1];
    we gate on > 0.5 by convention dsr_min=0.0 means 'any evidence at all',
    while e.g. dsr_min=0.95 demands strong evidence.
    """
    t = len(daily_returns)
    if t < 30:
        return float("nan")
    rets = daily_returns.to_numpy(dtype=float)
    sd = float(np.std(rets, ddof=1))
    if sd == 0:
        return float("nan")
    sr = float(np.mean(rets)) / sd  # per-period observed Sharpe

    # variance of trial Sharpes (per-period); degenerate spread -> 1/T fallback
    trials_pp = [s / math.sqrt(252.0) for s in trial_sharpes_ann if s == s]
    n = max(int(n_trials), 1)
    var_trials = float(np.var(trials_pp, ddof=1)) if len(trials_pp) >= 2 else 1.0 / t
    if var_trials <= 0:
        var_trials = 1.0 / t

    # expected max Sharpe of n pure-noise trials (Euler-Mascheroni gamma)
    gamma = 0.5772156649015329
    if n > 1:
        sr0 = math.sqrt(var_trials) * (
            (1 - gamma) * _PHI.inv_cdf(1 - 1.0 / n) + gamma * _PHI.inv_cdf(1 - 1.0 / (n * math.e))
        )
    else:
        sr0 = 0.0

    # moments of the return series (Pearson kurtosis = excess + 3)
    series = pd.Series(rets)
    skew = float(series.skew())
    kurt = float(series.kurtosis()) + 3.0
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0:
        denom = 1e-9
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)
    return float(_PHI.cdf(z))


# ------------------------------- gates -----------------------------------

@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float
    threshold: float
    detail: str


@dataclass(frozen=True)
class GateReport:
    gates: list[GateResult]
    verdict: str           # 'pass' | 'fail'
    reasons: list[str]
    weaknesses: list[str]  # auto-generated narrative for the report

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def evaluate_gates(
    cfg: GatesConfig,
    oos_metrics: dict[str, float],
    wf_efficiency: float,
    dsr: float,
    cost_breakeven: float,
    delay_table: list[dict[str, float]],
    regime_table: dict[str, dict[str, float]],
    neighbor_ratio: float,
) -> GateReport:
    gates: list[GateResult] = []

    def add(name: str, passed: bool, measured: float, threshold: float, detail: str) -> None:
        gates.append(GateResult(name, bool(passed), float(measured), float(threshold), detail))

    n_oos = oos_metrics.get("n_trades", 0.0)
    add("min_oos_trades", n_oos >= cfg.min_oos_trades, n_oos, cfg.min_oos_trades,
        "out-of-sample trade count")

    expectancy = oos_metrics.get("expectancy_usd", float("nan"))
    add("oos_expectancy_positive", expectancy == expectancy and expectancy > 0, expectancy, 0.0,
        "mean net P&L per OOS trade (USD)")

    oos_sharpe = oos_metrics.get("sharpe", float("nan"))
    add("oos_sharpe_positive", oos_sharpe == oos_sharpe and oos_sharpe > 0, oos_sharpe, 0.0,
        "stitched OOS Sharpe")

    add("wf_efficiency", wf_efficiency >= cfg.min_wf_efficiency, wf_efficiency,
        cfg.min_wf_efficiency, "OOS Sharpe / in-sample Sharpe")

    dsr_ok = dsr == dsr and dsr > 0.5 + cfg.dsr_min / 2  # dsr_min=0 -> >0.5
    add("deflated_sharpe", dsr_ok, dsr if dsr == dsr else -1.0, 0.5 + cfg.dsr_min / 2,
        "P(true Sharpe > best-of-N noise), trials-adjusted")

    add("cost_breakeven", cost_breakeven >= cfg.cost_breakeven_min, cost_breakeven,
        cfg.cost_breakeven_min, "highest cost multiplier still profitable")

    if cfg.require_delay_survival:
        delayed = [row for row in delay_table if row["extra_delay_bars"] > 0]
        survives = bool(delayed) and all(
            row["total_return"] == row["total_return"] and row["total_return"] > 0
            for row in delayed
        )
        worst = min((row["total_return"] for row in delayed), default=float("nan"))
        add("delay_survival", survives, worst, 0.0,
            "total return with +1 bar execution delay")

    positive = sum(1 for r in regime_table.values() if r["pnl"] > 0)
    add("positive_regimes", positive >= cfg.min_positive_regimes, positive,
        cfg.min_positive_regimes, "benchmark regimes (down/flat/up) with positive P&L")

    total_pnl = sum(r["pnl"] for r in regime_table.values())
    if total_pnl > 0:
        max_share = max(max(r["pnl"], 0.0) for r in regime_table.values()) / total_pnl
    else:
        max_share = 1.0
    add("regime_concentration", max_share <= cfg.max_single_regime_pnl_share, max_share,
        cfg.max_single_regime_pnl_share, "largest single-regime share of total P&L")

    add("param_neighborhood", neighbor_ratio >= cfg.min_neighbor_sharpe_ratio, neighbor_ratio,
        cfg.min_neighbor_sharpe_ratio, "median neighbor-params Sharpe / chosen Sharpe")

    failed = [g for g in gates if not g.passed]
    weaknesses = _weakness_narrative(gates, oos_metrics)
    return GateReport(
        gates=gates,
        verdict="fail" if failed else "pass",
        reasons=[f"{g.name}: measured {g.measured:.4g} vs threshold {g.threshold:.4g}" for g in failed],
        weaknesses=weaknesses,
    )


def _weakness_narrative(gates: list[GateResult], oos_metrics: dict[str, float]) -> list[str]:
    """Honest weaknesses/failure scenarios, derived from measurements - not prose-polished."""
    notes: list[str] = []
    by_name = {g.name: g for g in gates}

    def near(name: str, margin: float = 0.2) -> bool:
        g = by_name.get(name)
        if g is None or not g.passed or g.threshold == 0:
            return False
        return abs(g.measured - g.threshold) <= abs(g.threshold) * margin

    for g in gates:
        if not g.passed:
            notes.append(f"FAILED {g.name}: {g.detail} = {g.measured:.4g} (needs {g.threshold:.4g})")
    for name in ("cost_breakeven", "wf_efficiency", "deflated_sharpe", "regime_concentration"):
        if near(name):
            g = by_name[name]
            notes.append(f"MARGINAL {name}: {g.measured:.4g} barely clears {g.threshold:.4g}")
    mdd = oos_metrics.get("max_drawdown", float("nan"))
    if mdd == mdd and mdd < -0.15:
        notes.append(f"OOS max drawdown {mdd:.1%} already exceeds the Phase 2 hard limit (-15%)")
    streak = oos_metrics.get("longest_losing_streak", 0)
    if streak >= 8:
        notes.append(f"Longest OOS losing streak is {streak:.0f} trades - expect discipline stress")
    if not notes:
        notes.append("No gate failures or marginal passes detected; remaining risk is regime "
                     "non-stationarity that validation cannot rule out.")
    return notes
