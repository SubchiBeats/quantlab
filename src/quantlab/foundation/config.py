"""Configuration system.

Rules this module enforces (they are requirements, not conventions):
- Every config is schema-validated; unknown keys are errors (catches typos that
  would otherwise silently change behavior).
- v1 invariants (daily bars, next-bar-open fills, cash account, no shorting)
  are validated values - editing them to anything else is a load error.
- The pessimism multiplier on costs cannot go below 1.0.
- The SHA-256 of the fully-resolved config is stamped on every run, so any
  result can be traced to the exact settings that produced it.
- CLI overrides do not exist: a different setting means a different file and
  therefore a different hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GlobalConfig(_StrictModel):
    timezone: Literal["UTC"]
    log_level: str = "INFO"
    root_seed: int


class DataValidationConfig(_StrictModel):
    max_consecutive_gap_days: int = Field(ge=1)
    spike_zscore: float = Field(gt=0)
    min_avg_volume: float = Field(ge=0)
    min_bars: int = Field(ge=50)


class DataConfig(_StrictModel):
    frequency: Literal["daily"]
    benchmark_symbol: str
    universe: list[str] = Field(min_length=1)
    validation: DataValidationConfig


class CostConfig(_StrictModel):
    commission_per_share: float = Field(ge=0)
    min_commission: float = Field(ge=0)
    half_spread_bps: float = Field(ge=0)
    impact_coeff: float = Field(ge=0)
    pessimism_multiplier: float = Field(ge=1.0)  # may never understate costs


class SizingConfig(_StrictModel):
    risk_per_trade_pct: float = Field(gt=0, le=2.0)  # >2% per trade is reckless by design
    atr_period: int = Field(ge=2)
    atr_stop_mult: float = Field(gt=0)
    max_position_pct: float = Field(gt=0, le=25.0)


class BacktestConfig(_StrictModel):
    initial_capital: float = Field(gt=0)
    fill_policy: Literal["next_bar_open"]
    cash_account: Literal[True]
    allow_short: Literal[False]
    costs: CostConfig
    sizing: SizingConfig


class WalkForwardConfig(_StrictModel):
    scheme: Literal["rolling"]
    train_bars: int = Field(ge=100)
    test_bars: int = Field(ge=20)
    embargo_bars: int | Literal["auto"] = "auto"


class MonteCarloConfig(_StrictModel):
    n_paths: int = Field(ge=100)
    block_len: int | Literal["auto"] = "auto"
    ruin_drawdown_pct: float = Field(gt=0, lt=100)
    cost_stress_multipliers: list[float] = Field(min_length=1)
    delay_stress_bars: list[int] = Field(min_length=1)
    perturbation_samples: int = Field(ge=1)

    @field_validator("cost_stress_multipliers")
    @classmethod
    def _multipliers_sane(cls, v: list[float]) -> list[float]:
        if any(m < 1.0 for m in v):
            raise ValueError("cost stress multipliers below 1.0 would understate costs")
        return sorted(v)


class GatesConfig(_StrictModel):
    min_oos_trades: int = Field(ge=1)
    min_wf_efficiency: float
    dsr_min: float
    cost_breakeven_min: float = Field(ge=1.0)
    require_delay_survival: bool
    min_positive_regimes: int = Field(ge=0, le=3)
    max_single_regime_pnl_share: float = Field(gt=0, le=1.0)
    min_neighbor_sharpe_ratio: float = Field(ge=0)


class ValidationConfig(_StrictModel):
    walkforward: WalkForwardConfig
    montecarlo: MonteCarloConfig
    gates: GatesConfig


class StrategyConfig(_StrictModel):
    name: str
    version: int = Field(ge=1)
    universe: list[str] = Field(min_length=1)
    params: dict[str, Any]


class AppConfig(_StrictModel):
    """Fully-resolved configuration for one experiment run."""

    global_: GlobalConfig
    data: DataConfig
    backtest: BacktestConfig
    validation: ValidationConfig
    strategy: StrategyConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"config file is not a mapping: {path}")
    return loaded


def load_app_config(configs_dir: Path, strategy_name: str, strategy_version: int) -> AppConfig:
    strat_file = configs_dir / "strategies" / f"{strategy_name}@{strategy_version}.yaml"
    return AppConfig(
        global_=GlobalConfig(**_load_yaml(configs_dir / "global.yaml")),
        data=DataConfig(**_load_yaml(configs_dir / "data.yaml")),
        backtest=BacktestConfig(**_load_yaml(configs_dir / "backtest.yaml")),
        validation=ValidationConfig(**_load_yaml(configs_dir / "validation.yaml")),
        strategy=StrategyConfig(**_load_yaml(strat_file)),
    )


def config_hash(cfg: AppConfig) -> str:
    """SHA-256 over the canonical JSON form of the resolved config."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def assert_env_not_tracked(root: Path) -> None:
    """Refuse to operate if .env is committed to git (secrets must never be versioned)."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=root,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no git available: nothing to check
    if result.returncode == 0:
        raise RuntimeError(
            ".env is tracked by git. Remove it from version control "
            "(git rm --cached .env) before running QuantLab."
        )
