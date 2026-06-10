"""Config system: validation, invariants, hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantlab.foundation.config import config_hash, load_app_config


def test_load_and_hash_stable(workspace):
    cfg1 = load_app_config(workspace.configs, "sma_cross", 1)
    cfg2 = load_app_config(workspace.configs, "sma_cross", 1)
    assert config_hash(cfg1) == config_hash(cfg2)
    assert len(config_hash(cfg1)) == 64


def test_different_params_different_hash(workspace):
    cfg1 = load_app_config(workspace.configs, "sma_cross", 1)
    strat_file = workspace.strategy_configs / "sma_cross@1.yaml"
    strat_file.write_text(strat_file.read_text().replace("fast: 20", "fast: 10"), encoding="utf-8")
    cfg2 = load_app_config(workspace.configs, "sma_cross", 1)
    assert config_hash(cfg1) != config_hash(cfg2)


def test_pessimism_below_one_rejected(workspace):
    bt = workspace.configs / "backtest.yaml"
    bt.write_text(bt.read_text().replace("pessimism_multiplier: 1.5", "pessimism_multiplier: 0.8"),
                  encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_config(workspace.configs, "sma_cross", 1)


def test_shorting_cannot_be_enabled(workspace):
    bt = workspace.configs / "backtest.yaml"
    bt.write_text(bt.read_text().replace("allow_short: false", "allow_short: true"),
                  encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_config(workspace.configs, "sma_cross", 1)


def test_unknown_keys_rejected(workspace):
    bt = workspace.configs / "backtest.yaml"
    bt.write_text(bt.read_text() + "\nmystery_setting: 42\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_config(workspace.configs, "sma_cross", 1)


def test_risk_per_trade_capped(workspace):
    bt = workspace.configs / "backtest.yaml"
    bt.write_text(bt.read_text().replace("risk_per_trade_pct: 0.5", "risk_per_trade_pct: 5.0"),
                  encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_config(workspace.configs, "sma_cross", 1)
