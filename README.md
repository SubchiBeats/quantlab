# QuantLab

![tests](https://github.com/SubchiBeats/quantlab/actions/workflows/ci.yml/badge.svg)

Local-first quantitative research platform. **Research tool only** — Phase 1 contains no
broker connectivity, no order routing, and no live trading of any kind. Its job is to help
you research trading ideas and, more importantly, to make it structurally hard to fool
yourself while doing it.

This is not a money printer and doesn't claim to be one. Most strategies fail validation
here — **that is the point**. The test suite asserts that a random strategy gets rejected,
and the platform's own first demo strategy (out-of-sample Sharpe 1.06) was refused by the
deflated-Sharpe gate as statistically indistinguishable from multiple-testing noise.

Core principles (enforced by code, not convention):

- **Capital preservation first** — conservative cost model (1.5× pessimism multiplier),
  risk-based sizing, protective stops required, long-only cash account by construction.
- **Deterministic & reproducible** — every run records config hash, git SHA, data snapshot
  hash, and seed; same inputs give byte-identical results (integer micro-dollar ledger).
- **No lookahead, structurally** — strategies see data only through a point-in-time view
  that cannot return future bars; property tests perturb the future and assert the past
  is unchanged.
- **The vault never forgets** — hypotheses, experiments, runs, trades, and verdicts are
  append-only (database triggers block UPDATE/DELETE). Failed experiments block similar
  new ones until acknowledged in the journal.
- **Multiple-testing honesty** — every parameter combination evaluated in every fold is
  counted, accumulated per hypothesis family, and fed into the deflated Sharpe gate. The
  more you mine a vein, the harder it is to pass.

## Install

```powershell
cd quantlab
python -m venv .venv                       # Python 3.12+
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest           # 67 tests, all must pass
copy .env.example .env                     # secrets live here only (gitignored)
```

Pinned versions used during development are in `requirements.lock`
(`pip install -r requirements.lock` for an exact environment).

## The research workflow

```powershell
.\.venv\Scripts\quantlab.exe init

# 1. Data: fetch (Stooq, free), import CSVs, or generate dev-only synthetic data
quantlab data fetch -s SPY -s AAPL -s MSFT -s JNJ -s XOM
quantlab data import-csv path\to\export.csv --symbol SPY
quantlab data synth -s SPY -s AAPL          # DEV ONLY - reports carry a warning banner

# 2. Freeze an immutable, content-addressed snapshot (experiments bind to its hash)
quantlab data snapshot -m "why this dataset" -s SPY -s AAPL -s MSFT -s JNJ -s XOM

# 3. Pre-register the hypothesis BEFORE any backtest (mandatory)
quantlab hypothesis new --family trend-following `
  --statement "SMA crossovers capture medium-term trend persistence" `
  --mechanism "slow information diffusion" `
  --success "positive OOS expectancy, all gates pass" `
  --falsification "negative OOS expectancy or failure under 1.5x costs"

# 4. Run the experiment: full backtest + benchmark + walk-forward + Monte Carlo + gates
quantlab experiment run --hypothesis 1 --strategy sma_cross --version 1 --snapshot <id>

# 5. Generate the tear sheet (every figure traceable to vault identifiers)
quantlab report build 1                     # -> artifacts\E1\report.html

# Inspect / verify
quantlab vault experiments
quantlab vault show 1
quantlab journal add "decision notes..."
quantlab audit verify                       # walk the hash-chained audit log
```

If similar **failed** experiments exist in the vault, `experiment run` refuses to start
until you review them and acknowledge with a journal entry (`--ack-journal <id>`).

## What an experiment runs

1. **Full-sample backtest** — event-driven daily engine: signals strictly after the close,
   fills at next bar's open, ATR-based protective stops (gaps fill at the open, not the
   stop), risk-based sizing, commission + spread + impact costs, all ×1.5 pessimism.
2. **Benchmark** — buy-and-hold (default SPY) on the same snapshot with the same costs.
3. **Walk-forward** — rolling train/test folds with an embargo gap; parameters chosen per
   fold by grid search over the *declared* parameter space only; out-of-sample segments
   stitched into the evidentiary result.
4. **Monte Carlo** — trade bootstrap, block bootstrap, cost stress (re-runs at 1.5–3×
   costs), execution-delay stress (+1 bar), parameter-neighborhood perturbation.
5. **Regime analysis** — P&L split across benchmark down/flat/up regimes.
6. **Gates → verdict** — ten rejection gates including deflated Sharpe (counts every trial
   in the hypothesis family, ever). One failed gate = recorded `fail`, forever.

## Writing a strategy

Subclass `Strategy` in `src/quantlab/strategies/catalog/`, declare the **full** parameter
space upfront, and register with the factory. Strategies return desired position state
(`long`/`flat`) only — sizing and execution belong to the engine, and account state is
deliberately invisible to strategy code. See `sma_cross.py` and `rsi_meanrev.py`.

Add a config in `configs/strategies/<name>@<version>.yaml`. Params outside the declared
space are a load error; widening the space means a new version (a recorded event).

## Layout

```
configs/          YAML configs (validated, hashed; CLI overrides do not exist)
src/quantlab/
  foundation/     config, sqlite+migrations, audit chain, seeds, paths, clock
  data/           Stooq/CSV ingest, validation suite, Parquet store, snapshots, PIT API
  indicators/     pure, warm-up-aware, property-tested (SMA/EMA/RSI/ATR/vol)
  strategies/     contract, factory registry, catalog
  backtest/       event-driven engine, cost model
  metrics/        performance metrics, benchmark comparison
  validation/     walk-forward, Monte Carlo, gates/regimes/deflated Sharpe
  vault/          append-only research vault + journal
  experiments/    experiment engine (orchestration -> verdict)
  reports/        HTML tear-sheet generator
  cli/            typer CLI
db/               quantlab.sqlite (vault, audit, registry)
datastore/        raw/ curated/ snapshots/
artifacts/        per-experiment outputs (equity, trades, gates.json, report.html)
```

`db/`, `datastore/`, and `artifacts/` are not in git — **back them up**; they are your
research history.

## Phase boundaries

Phase 1 (this): research, backtesting, validation, vault, reporting.
Phase 2 (not built): paper broker, risk engine, kill switches, operational controls.
Phase 3 (not built): AI research assistant — read-only, never in any order path.
