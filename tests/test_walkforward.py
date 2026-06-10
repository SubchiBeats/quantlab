"""Walk-forward: fold geometry (no overlap, embargo respected) and OOS-only stitching."""

from __future__ import annotations

import pandas as pd

from quantlab.foundation.config import load_app_config
from quantlab.validation.walkforward import build_folds, run_walkforward


def test_fold_geometry():
    calendar = pd.bdate_range("2015-01-01", periods=1000)
    folds = build_folds(calendar, train_bars=300, test_bars=100, embargo_bars=50)
    assert folds, "expected at least one fold"
    for a, b, c, d in folds:
        assert b - a == 300
        assert c - b == 50          # embargo gap between train end and test start
        assert d - c == 100
        assert d <= len(calendar)
    # consecutive test windows tile without overlap
    for (_, _, c1, d1), (_, _, c2, _) in zip(folds, folds[1:]):
        assert c2 == c1 + 100
    # train windows never touch their own test windows
    for a, b, c, d in folds:
        assert b <= c


def test_insufficient_history_raises(workspace, trending_market):
    cfg = load_app_config(workspace.configs, "sma_cross", 1)
    import pytest

    from quantlab.data.pit import MarketData
    small = MarketData({s: trending_market.frame(s).iloc[:200] for s in trending_market.symbols})
    with pytest.raises(ValueError, match="not enough history"):
        run_walkforward(small, cfg)


def test_walkforward_oos_only_and_trial_accounting(workspace, trending_market):
    import quantlab.strategies  # noqa: F401  (register catalog)

    cfg = load_app_config(workspace.configs, "sma_cross", 1)
    wf = run_walkforward(trending_market, cfg)

    assert len(wf.folds) >= 2
    # OOS returns live strictly inside test windows
    test_windows = [(f.test_start, f.test_end) for f in wf.folds]
    for ts in wf.oos_returns.index:
        assert any(start <= ts <= end for start, end in test_windows), f"{ts} outside all test windows"
    # every (combo x fold) evaluation is counted toward multiple-testing
    from quantlab.strategies.factory import get_class
    grid_size = len(get_class("sma_cross", 1).grid())
    assert wf.n_trials == grid_size * len(wf.folds)
    assert len(wf.trial_sharpes) == wf.n_trials
    # chosen params always come from the declared grid
    grid = get_class("sma_cross", 1).grid()
    for f in wf.folds:
        assert f.chosen_params in grid
    assert 0.0 <= wf.param_stability <= 1.0
