"""Walk-forward validation with purging via an embargo gap.

Scheme (rolling):
  [ train_bars ][ embargo ][ test_bars ] -> roll forward by test_bars -> ...

- Parameters are chosen on each train window by grid search over the
  strategy's DECLARED parameter space only, then frozen for the test window.
- An embargo of at least the strategy's warm-up separates train and test, so
  indicator state spanning the boundary cannot leak fit information.
- Only out-of-sample (test) segments are stitched into the headline result.
  In-sample numbers are reported solely to compute walk-forward efficiency.
- The grid's Sharpe distribution and total trial count are returned for the
  deflated-Sharpe computation downstream.

Selection objective on train: Sharpe, requiring a minimum trade count; ties
break by lower max drawdown, then by parameter order (deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.backtest.engine import Trade, run_backtest
from quantlab.data.pit import MarketData
from quantlab.foundation.config import AppConfig
from quantlab.metrics.core import compute_metrics
from quantlab.strategies.factory import get_class

MIN_TRAIN_TRADES = 5  # below this, a train-window Sharpe is noise; combo is skipped


@dataclass
class FoldResult:
    fold_n: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_bars: int
    chosen_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_trades: int


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oos_returns: pd.Series           # stitched test-segment daily returns
    oos_trades: list[Trade]
    is_sharpe_mean: float            # mean train Sharpe of chosen params
    oos_sharpe: float
    wf_efficiency: float             # oos_sharpe / is_sharpe_mean (0 if IS <= 0)
    n_trials: int                    # total (combo x fold) evaluations, for DSR
    trial_sharpes: list[float]       # train Sharpes of ALL evaluated combos
    chosen_params_per_fold: list[dict[str, Any]] = field(default_factory=list)

    @property
    def param_stability(self) -> float:
        """Fraction of folds choosing the modal parameter set (1.0 = perfectly stable)."""
        if not self.chosen_params_per_fold:
            return 0.0
        keys = [tuple(sorted(p.items())) for p in self.chosen_params_per_fold]
        return keys.count(max(set(keys), key=keys.count)) / len(keys)


def build_folds(
    calendar: pd.DatetimeIndex, train_bars: int, test_bars: int, embargo_bars: int
) -> list[tuple[int, int, int, int]]:
    """Index spans (train_start, train_end, test_start, test_end), end-exclusive."""
    folds = []
    pos = 0
    while pos + train_bars + embargo_bars + test_bars <= len(calendar):
        folds.append(
            (pos, pos + train_bars,
             pos + train_bars + embargo_bars,
             pos + train_bars + embargo_bars + test_bars)
        )
        pos += test_bars
    return folds


def run_walkforward(market: MarketData, cfg: AppConfig) -> WalkForwardResult:
    wf_cfg = cfg.validation.walkforward
    strategy_cls = get_class(cfg.strategy.name, cfg.strategy.version)
    universe = cfg.strategy.universe

    probe = strategy_cls(cfg.strategy.params)
    embargo = probe.warmup_bars() if wf_cfg.embargo_bars == "auto" else int(wf_cfg.embargo_bars)

    calendar = market.calendar
    spans = build_folds(calendar, wf_cfg.train_bars, wf_cfg.test_bars, embargo)
    if not spans:
        raise ValueError(
            f"not enough history for walk-forward: need at least "
            f"{wf_cfg.train_bars + embargo + wf_cfg.test_bars} bars, have {len(calendar)}"
        )

    grid = strategy_cls.grid()
    folds: list[FoldResult] = []
    oos_return_chunks: list[pd.Series] = []
    oos_trades: list[Trade] = []
    trial_sharpes: list[float] = []
    n_trials = 0

    for fold_n, (a, b, c, d) in enumerate(spans):
        train_start, train_end = calendar[a], calendar[b - 1]
        test_start, test_end = calendar[c], calendar[d - 1]

        best: tuple[float, float, dict[str, Any]] | None = None  # (sharpe, -mdd, params)
        for combo in grid:
            n_trials += 1
            strat = strategy_cls(combo)
            res = run_backtest(market, strat, cfg.backtest, universe, train_start, train_end)
            m = compute_metrics(res)
            sr = m["sharpe"]
            trial_sharpes.append(sr if sr == sr else 0.0)  # NaN -> 0 for the trial record
            if m["n_trades"] < MIN_TRAIN_TRADES or sr != sr:
                continue
            key = (sr, m["max_drawdown"])  # higher sharpe, then shallower (less negative) dd
            if best is None or key > (best[0], best[1]):
                best = (sr, m["max_drawdown"], combo)

        if best is None:
            # nothing tradeable on this train window; an honest zero-return test segment
            chosen = dict(probe.params)
            train_sr = float("nan")
            test_rets = pd.Series(dtype=float)
            test_sr, test_ret, test_n = float("nan"), 0.0, 0
        else:
            train_sr, _, chosen = best
            strat = strategy_cls(chosen)
            test_res = run_backtest(market, strat, cfg.backtest, universe, test_start, test_end)
            test_m = compute_metrics(test_res)
            test_rets = test_res.daily_returns
            test_sr = test_m["sharpe"]
            test_ret = test_m["total_return"]
            test_n = int(test_m["n_trades"])
            oos_trades.extend(test_res.trades)

        folds.append(
            FoldResult(fold_n, train_start, train_end, test_start, test_end, embargo,
                       chosen, float(train_sr), float(test_sr), float(test_ret), test_n)
        )
        if len(test_rets):
            oos_return_chunks.append(test_rets)

    oos_returns = pd.concat(oos_return_chunks).sort_index() if oos_return_chunks else pd.Series(dtype=float)
    from quantlab.metrics.core import sharpe as sharpe_fn  # local import to avoid cycle at module load

    oos_sharpe = sharpe_fn(oos_returns)
    train_srs = [f.train_sharpe for f in folds if f.train_sharpe == f.train_sharpe]
    is_mean = sum(train_srs) / len(train_srs) if train_srs else float("nan")
    if is_mean == is_mean and is_mean > 0 and oos_sharpe == oos_sharpe:
        efficiency = oos_sharpe / is_mean
    else:
        efficiency = 0.0

    return WalkForwardResult(
        folds=folds,
        oos_returns=oos_returns,
        oos_trades=oos_trades,
        is_sharpe_mean=float(is_mean),
        oos_sharpe=float(oos_sharpe),
        wf_efficiency=float(efficiency),
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        chosen_params_per_fold=[f.chosen_params for f in folds],
    )
