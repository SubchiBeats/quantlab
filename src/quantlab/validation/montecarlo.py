"""Monte Carlo simulation suite.

Four methods, deliberately reported as DIFFERENT evidence classes because they
answer different questions:

- trade_bootstrap: resample the trade P&L sequence with replacement. Answers
  "how bad could sequencing luck be IF the edge is real and trades were
  independent". Weakest evidence - it destroys autocorrelation by design.
- block_bootstrap: resample contiguous blocks of daily returns (block length
  ~ sqrt(T) by default), preserving short-range clustering. More realistic
  drawdown distributions.
- cost_stress / delay_stress: not resampling - actual re-runs of the backtest
  with multiplied costs / added execution delay. Answers "does the edge
  survive worse frictions than modeled".
- param_perturbation: re-runs at neighboring parameter sets (one step in the
  declared grid). A real edge degrades gracefully; an overfit artifact falls
  off a cliff.

All resampling is seeded; identical seeds give identical distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestResult
from quantlab.foundation.rng import make_rng
from quantlab.metrics.core import compute_metrics, sharpe


@dataclass(frozen=True)
class McSummary:
    method: str
    n_paths: int
    seed: int
    stats: dict[str, Any]


def _max_dd_of_path(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def trade_bootstrap(
    trade_pnls: list[float], initial_capital: float, n_paths: int, seed: int,
    ruin_drawdown_pct: float,
) -> McSummary:
    rng = make_rng(seed)
    pnls = np.asarray(trade_pnls, dtype=float)
    if len(pnls) == 0:
        return McSummary("trade_bootstrap", n_paths, seed, {"error": "no trades"})
    n = len(pnls)
    dds = np.empty(n_paths)
    finals = np.empty(n_paths)
    for i in range(n_paths):
        sample = pnls[rng.integers(0, n, n)]
        equity = initial_capital + np.cumsum(sample)
        equity = np.maximum(equity, 1e-9)
        dds[i] = _max_dd_of_path(np.concatenate(([initial_capital], equity)))
        finals[i] = equity[-1]
    ruin = float(np.mean(dds <= -ruin_drawdown_pct / 100.0))
    return McSummary(
        "trade_bootstrap", n_paths, seed,
        {
            "dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 5)),   # 95th worst = 5th percentile of negatives
            "dd_p99": float(np.percentile(dds, 1)),
            "risk_of_ruin": ruin,
            "ruin_drawdown_pct": ruin_drawdown_pct,
            "final_equity_p5": float(np.percentile(finals, 5)),
            "final_equity_p50": float(np.percentile(finals, 50)),
            "prob_negative": float(np.mean(finals < initial_capital)),
        },
    )


def block_bootstrap(
    daily_returns: pd.Series, n_paths: int, seed: int, ruin_drawdown_pct: float,
    block_len: int | None = None,
) -> McSummary:
    rng = make_rng(seed)
    rets = daily_returns.to_numpy(dtype=float)
    t = len(rets)
    if t < 40:
        return McSummary("block_bootstrap", n_paths, seed, {"error": "too few return observations"})
    blk = block_len or int(np.ceil(np.sqrt(t)))
    n_blocks = int(np.ceil(t / blk))
    dds = np.empty(n_paths)
    ann_rets = np.empty(n_paths)
    for i in range(n_paths):
        starts = rng.integers(0, t - blk + 1, n_blocks)
        path = np.concatenate([rets[s : s + blk] for s in starts])[:t]
        equity = np.cumprod(1.0 + path)
        dds[i] = _max_dd_of_path(np.concatenate(([1.0], equity)))
        ann_rets[i] = equity[-1] ** (252.0 / t) - 1.0
    return McSummary(
        "block_bootstrap", n_paths, seed,
        {
            "block_len": blk,
            "dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 5)),
            "dd_p99": float(np.percentile(dds, 1)),
            "risk_of_ruin": float(np.mean(dds <= -ruin_drawdown_pct / 100.0)),
            "ruin_drawdown_pct": ruin_drawdown_pct,
            "cagr_p5": float(np.percentile(ann_rets, 5)),
            "cagr_p50": float(np.percentile(ann_rets, 50)),
            "prob_negative_cagr": float(np.mean(ann_rets < 0)),
        },
    )


def cost_stress(
    rerun: Callable[[float], BacktestResult], multipliers: list[float]
) -> McSummary:
    """`rerun(multiplier)` must re-execute the full backtest at that cost level."""
    table: list[dict[str, float]] = []
    breakeven = 0.0
    for mult in sorted(multipliers):
        m = compute_metrics(rerun(mult))
        table.append({"multiplier": mult, "cagr": m["cagr"], "sharpe": m["sharpe"],
                      "total_return": m["total_return"]})
        if m["total_return"] == m["total_return"] and m["total_return"] > 0:
            breakeven = mult
    return McSummary("cost_stress", len(multipliers), 0,
                     {"table": table, "max_profitable_multiplier": breakeven})


def delay_stress(
    rerun: Callable[[int], BacktestResult], delays: list[int]
) -> McSummary:
    table: list[dict[str, float]] = []
    for d in sorted(delays):
        m = compute_metrics(rerun(d))
        table.append({"extra_delay_bars": d, "cagr": m["cagr"], "sharpe": m["sharpe"],
                      "total_return": m["total_return"]})
    return McSummary("delay_stress", len(delays), 0, {"table": table})


def param_perturbation(
    rerun: Callable[[dict[str, Any]], BacktestResult],
    chosen: dict[str, Any],
    grid: list[dict[str, Any]],
    chosen_sharpe: float,
    n_samples: int,
    seed: int,
) -> McSummary:
    """Evaluate up to n_samples grid neighbors (one param one step away)."""
    rng = make_rng(seed)

    def is_neighbor(combo: dict[str, Any]) -> bool:
        diffs = [k for k in chosen if combo.get(k) != chosen[k]]
        return len(diffs) == 1

    neighbors = [c for c in grid if is_neighbor(c)]
    if not neighbors:
        return McSummary("param_perturbation", 0, seed, {"error": "no grid neighbors"})
    if len(neighbors) > n_samples:
        idx = rng.choice(len(neighbors), size=n_samples, replace=False)
        neighbors = [neighbors[i] for i in sorted(idx)]

    sharpes = []
    for combo in neighbors:
        res = rerun(combo)
        sr = sharpe(res.daily_returns)
        sharpes.append(sr if sr == sr else 0.0)
    med = float(np.median(sharpes))
    ratio = med / chosen_sharpe if chosen_sharpe and chosen_sharpe > 0 else 0.0
    return McSummary(
        "param_perturbation", len(neighbors), seed,
        {
            "neighbor_sharpes": [round(s, 4) for s in sharpes],
            "neighbor_median_sharpe": med,
            "chosen_sharpe": chosen_sharpe,
            "neighbor_sharpe_ratio": float(ratio),
        },
    )
