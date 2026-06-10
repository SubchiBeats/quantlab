"""QuantLab CLI - the single entry point for the Phase 1 research workflow.

    quantlab init
    quantlab data fetch -s SPY -s AAPL          (Stooq, no key needed)
    quantlab data import-csv file.csv --symbol X
    quantlab data synth -s TEST1                (DEV ONLY synthetic data)
    quantlab data validate -s SPY
    quantlab data snapshot -m "description" -s SPY -s AAPL
    quantlab journal add "text"
    quantlab hypothesis new --family f --statement ... --mechanism ...
    quantlab experiment run --hypothesis 1 --strategy sma_cross --snapshot <id>
    quantlab report build 1
    quantlab vault experiments | vault hypotheses
    quantlab audit verify
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from quantlab.foundation import audit as audit_mod
from quantlab.foundation.clock import utc_now_iso
from quantlab.foundation.config import assert_env_not_tracked
from quantlab.foundation.db import open_db
from quantlab.foundation.paths import Paths, resolve_root

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="QuantLab: local-first quant research platform (Phase 1: research only).")
data_app = typer.Typer(no_args_is_help=True, help="Data ingestion, validation, snapshots.")
journal_app = typer.Typer(no_args_is_help=True, help="Append-only research journal.")
hypothesis_app = typer.Typer(no_args_is_help=True, help="Pre-registered hypotheses.")
experiment_app = typer.Typer(no_args_is_help=True, help="Run experiments (full validation battery).")
report_app = typer.Typer(no_args_is_help=True, help="Generate HTML tear sheets from the vault.")
vault_app = typer.Typer(no_args_is_help=True, help="Inspect the research vault.")
audit_app = typer.Typer(no_args_is_help=True, help="Audit log operations.")
for name, sub in (("data", data_app), ("journal", journal_app), ("hypothesis", hypothesis_app),
                  ("experiment", experiment_app), ("report", report_app), ("vault", vault_app),
                  ("audit", audit_app)):
    app.add_typer(sub, name=name)

_ROOT: Optional[Path] = None


@app.callback()
def _main(root: Optional[Path] = typer.Option(None, "--root", help="Workspace root (default: cwd or QUANTLAB_HOME)")) -> None:
    global _ROOT
    _ROOT = root


def _paths() -> Paths:
    paths = resolve_root(_ROOT)
    load_dotenv(paths.root / ".env")
    assert_env_not_tracked(paths.root)
    return paths


def _db(paths: Paths):
    if not paths.db_file.exists():
        typer.echo("No database found - run `quantlab init` first.", err=True)
        raise typer.Exit(1)
    return open_db(paths.db_file)


def _register_instrument(conn, symbol: str, source: str, df, is_benchmark: bool = False) -> None:
    conn.execute(
        "INSERT INTO instruments(symbol, asset_class, source, first_bar, last_bar, is_benchmark,"
        " created_at) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(symbol) DO UPDATE SET source=excluded.source, first_bar=excluded.first_bar,"
        " last_bar=excluded.last_bar",
        (symbol.upper(), "equity", source, df.index[0].strftime("%Y-%m-%d"),
         df.index[-1].strftime("%Y-%m-%d"), int(is_benchmark), utc_now_iso()),
    )
    conn.commit()


def _validate_and_store(paths: Paths, conn, symbol: str, df, source: str) -> bool:
    """Validate; only write to curated on pass. Returns True if stored."""
    from quantlab.data.store import write_curated
    from quantlab.data.validate import passes, validate_frame
    from quantlab.foundation.config import DataConfig, _load_yaml

    data_cfg = DataConfig(**_load_yaml(paths.configs / "data.yaml"))
    issues = validate_frame(df, data_cfg.validation)
    for issue in issues:
        conn.execute(
            "INSERT INTO data_quality_log(symbol, range_start, range_end, check_name, severity,"
            " detail, created_at) VALUES (?,?,?,?,?,?,?)",
            (symbol.upper(), issue.range_start, issue.range_end, issue.check,
             issue.severity, issue.detail, utc_now_iso()),
        )
        color = {"fail": typer.colors.RED, "warn": typer.colors.YELLOW}.get(issue.severity)
        typer.secho(f"  [{issue.severity}] {issue.check}: {issue.detail}", fg=color)
    conn.commit()
    if not passes(issues):
        typer.secho(f"  {symbol}: FAILED validation - not written to curated store", fg=typer.colors.RED)
        return False
    write_curated(paths, symbol, df)
    _register_instrument(conn, symbol, source, df,
                         is_benchmark=(symbol.upper() == data_cfg.benchmark_symbol.upper()))
    audit_mod.append_event(conn, "system", "data.curated",
                           {"symbol": symbol.upper(), "source": source, "bars": len(df)})
    typer.secho(f"  {symbol}: {len(df)} bars curated", fg=typer.colors.GREEN)
    return True


# --------------------------------- init ----------------------------------

@app.command()
def init() -> None:
    """Create the workspace directories and database."""
    paths = _paths().ensure()
    conn = open_db(paths.db_file)
    audit_mod.append_event(conn, "system", "workspace.init", {"root": str(paths.root)})
    typer.secho(f"Workspace ready at {paths.root}", fg=typer.colors.GREEN)
    typer.echo("Next: quantlab data fetch -s SPY -s AAPL  (or data synth for offline dev)")


# --------------------------------- data ----------------------------------

@data_app.command("fetch")
def data_fetch(
    symbols: list[str] = typer.Option(None, "--symbol", "-s"),
    source: str = typer.Option("tiingo", "--source", help="tiingo (adjusted, needs TIINGO_API_KEY) | stooq (no key)"),
) -> None:
    """Fetch daily history from a vendor, validate, and curate."""
    import os

    from quantlab.data.ingest import fetch_stooq, fetch_tiingo, save_raw
    from quantlab.foundation.config import DataConfig, _load_yaml

    if source not in ("tiingo", "stooq"):
        typer.secho(f"unknown source '{source}' (use tiingo or stooq)", fg=typer.colors.RED)
        raise typer.Exit(1)
    paths = _paths()
    conn = _db(paths)
    if not symbols:
        symbols = DataConfig(**_load_yaml(paths.configs / "data.yaml")).universe
    token = os.environ.get("TIINGO_API_KEY", "")
    if source == "tiingo" and not token:
        typer.secho("TIINGO_API_KEY is not set. Get a free key at https://www.tiingo.com, "
                    "put it in .env, and retry (or use --source stooq).", fg=typer.colors.RED)
        raise typer.Exit(1)
    for sym in symbols:
        typer.echo(f"{sym}: fetching from {source}...")
        try:
            if source == "tiingo":
                df, raw = fetch_tiingo(sym, token)
                save_raw(paths.raw, "tiingo", sym, raw, ext="json")
            else:
                df, raw = fetch_stooq(sym)
                save_raw(paths.raw, "stooq", sym, raw)
        except Exception as exc:  # noqa: BLE001 - report and continue with other symbols
            typer.secho(f"  {sym}: fetch failed: {exc}", fg=typer.colors.RED)
            continue
        _validate_and_store(paths, conn, sym, df, source)


@data_app.command("import-csv")
def data_import_csv(
    file: Path = typer.Argument(..., exists=True),
    symbol: str = typer.Option(..., "--symbol"),
) -> None:
    """Import a local CSV (date,open,high,low,close,volume), validate, and curate."""
    from quantlab.data.ingest import read_csv_file, save_raw

    paths = _paths()
    conn = _db(paths)
    df, raw = read_csv_file(file, symbol)
    save_raw(paths.raw, "csv", symbol, raw)
    _validate_and_store(paths, conn, symbol, df, "csv")


@data_app.command("synth")
def data_synth(
    symbols: list[str] = typer.Option(..., "--symbol", "-s"),
    bars: int = typer.Option(2400, help="number of daily bars"),
    trend: float = typer.Option(0.05, help="average annualized drift"),
) -> None:
    """DEV ONLY: generate seeded synthetic data. Reports will carry a synthetic-data banner."""
    from quantlab.data.synth import generate_daily
    from quantlab.foundation.config import GlobalConfig, _load_yaml
    from quantlab.foundation.rng import derive_seed

    paths = _paths()
    conn = _db(paths)
    root_seed = GlobalConfig(**_load_yaml(paths.configs / "global.yaml")).root_seed
    typer.secho("WARNING: synthetic data is for development only - results are not research.",
                fg=typer.colors.YELLOW)
    for sym in symbols:
        df = generate_daily(sym, bars, derive_seed(root_seed, "synth", sym.upper()), trend=trend)
        _validate_and_store(paths, conn, sym, df, "synthetic")


@data_app.command("validate")
def data_validate(symbols: list[str] = typer.Option(..., "--symbol", "-s")) -> None:
    """Re-run the validation suite against curated data (results go to data_quality_log)."""
    from quantlab.data.store import load_curated
    from quantlab.data.validate import passes, validate_frame
    from quantlab.foundation.config import DataConfig, _load_yaml

    paths = _paths()
    conn = _db(paths)
    data_cfg = DataConfig(**_load_yaml(paths.configs / "data.yaml"))
    failed = False
    for sym in symbols:
        df = load_curated(paths, sym)
        issues = validate_frame(df, data_cfg.validation)
        typer.echo(f"{sym}: {len(df)} bars, {len(issues)} findings")
        for issue in issues:
            conn.execute(
                "INSERT INTO data_quality_log(symbol, range_start, range_end, check_name,"
                " severity, detail, created_at) VALUES (?,?,?,?,?,?,?)",
                (sym.upper(), issue.range_start, issue.range_end, issue.check,
                 issue.severity, issue.detail, utc_now_iso()),
            )
            typer.echo(f"  [{issue.severity}] {issue.check}: {issue.detail}")
        conn.commit()
        failed = failed or not passes(issues)
    raise typer.Exit(1 if failed else 0)


@data_app.command("snapshot")
def data_snapshot(
    symbols: list[str] = typer.Option(..., "--symbol", "-s"),
    message: str = typer.Option(..., "--message", "-m", help="what this snapshot is for"),
) -> None:
    """Freeze curated data into an immutable, content-addressed snapshot."""
    from quantlab.data.store import create_snapshot

    paths = _paths()
    conn = _db(paths)
    snapshot_id = create_snapshot(paths, conn, symbols, message)
    audit_mod.append_event(conn, "system", "snapshot.create",
                           {"snapshot_id": snapshot_id, "symbols": sorted(s.upper() for s in symbols)})
    typer.secho(f"snapshot {snapshot_id} created", fg=typer.colors.GREEN)


@data_app.command("snapshots")
def data_snapshots() -> None:
    """List snapshots."""
    conn = _db(_paths())
    for row in conn.execute("SELECT * FROM snapshots ORDER BY created_at"):
        typer.echo(f"{row['snapshot_id']}  {row['created_at']}  "
                   f"{row['coverage_start']}..{row['coverage_end']}  {row['description']}")


# -------------------------------- journal --------------------------------

@journal_app.command("add")
def journal_add(
    body: str = typer.Argument(...),
    tags: Optional[str] = typer.Option(None),
    author: str = typer.Option("sahib"),
) -> None:
    """Append a journal entry (append-only; used for acknowledgments and decisions)."""
    from quantlab.vault.vault import add_journal

    conn = _db(_paths())
    journal_id = add_journal(conn, author, body, tags)
    typer.secho(f"journal entry {journal_id} recorded", fg=typer.colors.GREEN)


@journal_app.command("list")
def journal_list(last: int = typer.Option(20)) -> None:
    conn = _db(_paths())
    for row in conn.execute(
        "SELECT * FROM journal_entries ORDER BY journal_id DESC LIMIT ?", (last,)
    ):
        typer.echo(f"[{row['journal_id']}] {row['ts']} {row['author']}: {row['body'][:100]}")


# ------------------------------- hypothesis ------------------------------

@hypothesis_app.command("new")
def hypothesis_new(
    family: str = typer.Option(..., help="research family for multiple-testing accounting"),
    statement: str = typer.Option(..., help="the testable claim"),
    mechanism: str = typer.Option(..., help="WHY should this edge exist?"),
    success: str = typer.Option(..., help="pre-registered success criteria"),
    falsification: str = typer.Option(..., help="what result kills this idea"),
) -> None:
    """Pre-register a hypothesis. Experiments cannot run without one."""
    from quantlab.vault.vault import register_hypothesis

    conn = _db(_paths())
    hypothesis_id = register_hypothesis(conn, family, statement, mechanism, success, falsification)
    typer.secho(f"hypothesis {hypothesis_id} registered (family: {family})", fg=typer.colors.GREEN)


# ------------------------------- experiment ------------------------------

@experiment_app.command("run")
def experiment_run(
    hypothesis: int = typer.Option(..., help="pre-registered hypothesis id"),
    strategy: str = typer.Option(..., help="registered strategy name"),
    version: int = typer.Option(1, help="strategy version"),
    snapshot: str = typer.Option(..., help="data snapshot id"),
    ack_journal: Optional[int] = typer.Option(
        None, help="journal id acknowledging similar prior failures (required when they exist)"
    ),
) -> None:
    """Run the full validation battery and record an immutable verdict."""
    import quantlab.strategies  # noqa: F401 - registers the catalog
    from quantlab.experiments.runner import run_experiment
    from quantlab.vault.vault import SimilarFailuresError

    paths = _paths()
    conn = _db(paths)
    try:
        summary = run_experiment(paths, conn, hypothesis, strategy, version, snapshot, ack_journal)
    except SimilarFailuresError as exc:
        typer.secho(str(exc), fg=typer.colors.YELLOW)
        raise typer.Exit(2) from None

    verdict = summary["verdict"]
    color = typer.colors.GREEN if verdict == "pass" else typer.colors.RED
    typer.secho(f"\nExperiment E{summary['experiment_id']}: {verdict.upper()}", fg=color, bold=True)
    if summary["reasons"]:
        for reason in summary["reasons"]:
            typer.echo(f"  - {reason}")
    oos = summary["oos_metrics"]
    typer.echo(
        f"\nOOS: sharpe={oos['sharpe']:.2f} trades={oos['n_trades']:.0f} "
        f"maxDD={oos['max_drawdown']:.1%} wf_eff={oos['wf_efficiency']:.2f}"
    )
    if summary["dirty_tree"]:
        typer.secho("note: git tree was dirty - this run is recorded as non-promotable",
                    fg=typer.colors.YELLOW)
    typer.echo(f"artifacts: {summary['artifact_dir']}")
    typer.echo(f"next: quantlab report build {summary['experiment_id']}")


# --------------------------------- report --------------------------------

@report_app.command("build")
def report_build(experiment_id: int = typer.Argument(...)) -> None:
    """Generate the HTML tear sheet for a completed experiment."""
    from quantlab.reports.tearsheet import build_report

    paths = _paths()
    conn = _db(paths)
    out = build_report(paths, conn, experiment_id)
    audit_mod.append_event(conn, "system", "report.build",
                           {"experiment_id": experiment_id, "path": str(out)})
    typer.secho(f"report written: {out}", fg=typer.colors.GREEN)


# --------------------------------- vault ---------------------------------

@vault_app.command("experiments")
def vault_experiments() -> None:
    conn = _db(_paths())
    for row in conn.execute(
        "SELECT e.*, h.family FROM experiments e JOIN hypotheses h USING(hypothesis_id)"
        " ORDER BY e.experiment_id"
    ):
        typer.echo(
            f"E{row['experiment_id']}  H{row['hypothesis_id']} [{row['family']}] "
            f"{row['strategy_name']}@{row['strategy_version']}  snap={row['snapshot_id'][:8]}  "
            f"{row['status']}  verdict={row['verdict'] or '-'}"
        )


@vault_app.command("hypotheses")
def vault_hypotheses() -> None:
    conn = _db(_paths())
    for row in conn.execute("SELECT * FROM hypotheses ORDER BY hypothesis_id"):
        typer.echo(f"H{row['hypothesis_id']} [{row['family']}] {row['statement'][:90]}")


@vault_app.command("show")
def vault_show(experiment_id: int = typer.Argument(...)) -> None:
    """Full detail of one experiment: runs, metrics, verdict."""
    conn = _db(_paths())
    e = conn.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
    if e is None:
        typer.secho("not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(json.dumps({k: e[k] for k in e.keys()}, indent=2, default=str))
    for r in conn.execute("SELECT * FROM runs WHERE experiment_id=? ORDER BY started_at", (experiment_id,)):
        typer.echo(f"\nrun {r['run_id']} ({r['run_type']}) seed={r['seed']} promotable={r['promotable']}")
        for m in conn.execute(
            "SELECT metric, period, value FROM run_metrics WHERE run_id=? ORDER BY metric", (r["run_id"],)
        ):
            value = "n/a" if m["value"] is None else f"{m['value']:.6g}"
            typer.echo(f"  {m['period']}.{m['metric']} = {value}")


# --------------------------------- audit ---------------------------------

@audit_app.command("verify")
def audit_verify() -> None:
    """Walk the audit hash chain and report integrity."""
    conn = _db(_paths())
    ok, message = audit_mod.verify_chain(conn)
    typer.secho(message, fg=typer.colors.GREEN if ok else typer.colors.RED)
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
