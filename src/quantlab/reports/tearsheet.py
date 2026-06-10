"""Reporting engine.

Reports are GENERATED from vault data, never authored: every number on the
tear sheet is read back from the database/artifacts of a completed experiment,
and the footer carries the identifiers (experiment, hypothesis, snapshot,
config hash, git SHA, engine version) needed to re-derive it. If a metric is
missing it renders as n/a - the generator never fabricates values.
"""

from __future__ import annotations

import base64
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from quantlab import ENGINE_VERSION
from quantlab.foundation.clock import utc_now_iso
from quantlab.foundation.paths import Paths

TEMPLATES = Path(__file__).parent / "templates"


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _metric(metrics: dict[str, float], name: str, fmt: str) -> str:
    value = metrics.get(name)
    if value is None or value != value:
        return "n/a"
    if fmt == "pct":
        return f"{value:.2%}"
    if fmt == "f2":
        return f"{value:.2f}"
    if fmt == "usd":
        return f"${value:,.2f}"
    if fmt == "int":
        return f"{value:.0f}"
    return str(value)


def _load_metrics(conn: sqlite3.Connection, run_id: str, period: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT metric, value FROM run_metrics WHERE run_id = ? AND period = ?", (run_id, period)
    ).fetchall()
    return {r["metric"]: (float("nan") if r["value"] is None else r["value"]) for r in rows}


def _mc_table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{label}</th>" for _, label, _ in columns)
    body = ""
    for row in rows:
        cells = ""
        for key, _, fmt in columns:
            value = row.get(key)
            if value is None or (isinstance(value, float) and value != value):
                cells += "<td>n/a</td>"
            elif fmt == "pct":
                cells += f"<td>{value:.2%}</td>"
            elif fmt == "f2":
                cells += f"<td>{value:.2f}</td>"
            else:
                cells += f"<td>{value}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def build_report(paths: Paths, conn: sqlite3.Connection, experiment_id: int) -> Path:
    e = conn.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    if e is None:
        raise KeyError(f"experiment {experiment_id} not found")
    if e["status"] != "complete":
        raise RuntimeError(f"experiment {experiment_id} is not complete (status={e['status']})")
    h = conn.execute(
        "SELECT * FROM hypotheses WHERE hypothesis_id = ?", (e["hypothesis_id"],)
    ).fetchone()

    runs = {
        r["run_type"]: r
        for r in conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY started_at", (experiment_id,)
        )
    }
    full_m = _load_metrics(conn, runs["backtest"]["run_id"], "full")
    oos_m = _load_metrics(conn, runs["walkforward"]["run_id"], "oos")
    bench_m = _load_metrics(conn, runs["benchmark"]["run_id"], "full")

    folds = conn.execute(
        "SELECT * FROM walkforward_folds WHERE run_id = ? ORDER BY fold_n",
        (runs["walkforward"]["run_id"],),
    ).fetchall()
    mc_rows = conn.execute(
        "SELECT * FROM mc_simulations WHERE run_id = ? ORDER BY id",
        (runs["montecarlo"]["run_id"],),
    ).fetchall()

    artifact_dir = paths.root / runs["backtest"]["artifact_dir"]
    gates_payload = json.loads((artifact_dir / "gates.json").read_text(encoding="utf-8"))

    # data source: synthetic banner if any snapshot symbol came from the synth generator
    synthetic = bool(
        conn.execute(
            "SELECT 1 FROM instruments WHERE source = 'synthetic' LIMIT 1"
        ).fetchone()
    )

    # ---- charts ----
    chart_equity = chart_dd = chart_oos = None
    eq_path = artifact_dir / "equity_full.parquet"
    bench_path = artifact_dir / "equity_benchmark.parquet"
    if eq_path.exists():
        eq = pd.read_parquet(eq_path)["equity"]
        fig, ax = plt.subplots(figsize=(9.5, 4))
        ax.plot(eq.index, eq.values, label="strategy", linewidth=1.2)
        if bench_path.exists():
            be = pd.read_parquet(bench_path)["equity"]
            ax.plot(be.index, be.values, label=f"benchmark ({e['benchmark_symbol']})",
                    linewidth=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
        chart_equity = _fig_to_b64(fig)

        dd = eq / eq.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(9.5, 2.6))
        ax.fill_between(dd.index, dd.values, 0, color="#c62828", alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_ylabel("drawdown")
        chart_dd = _fig_to_b64(fig)

    oos_path = artifact_dir / "oos_returns.parquet"
    if oos_path.exists():
        oos = pd.read_parquet(oos_path)["oos_return"]
        cum = (1 + oos).cumprod() - 1
        fig, ax = plt.subplots(figsize=(9.5, 3))
        ax.plot(cum.index, cum.values, linewidth=1.2, color="#16213e")
        ax.grid(alpha=0.3)
        ax.set_ylabel("cumulative OOS return")
        chart_oos = _fig_to_b64(fig)

    # ---- performance table (all mandated metrics) ----
    rows_spec = [
        ("Total return", "total_return", "pct"),
        ("Annualized return (CAGR)", "cagr", "pct"),
        ("Sharpe ratio (rf=0)", "sharpe", "f2"),
        ("Sortino ratio", "sortino", "f2"),
        ("Max drawdown (worst)", "max_drawdown", "pct"),
        ("Longest underwater (bars)", "longest_underwater_bars", "int"),
        ("Profit factor", "profit_factor", "f2"),
        ("Win rate", "win_rate", "pct"),
        ("Expectancy per trade", "expectancy_usd", "usd"),
        ("Trades", "n_trades", "int"),
        ("Longest losing streak", "longest_losing_streak", "int"),
        ("Avg exposure", "avg_exposure", "pct"),
        ("Cost share of gross P&L", "cost_share_of_gross", "pct"),
        ("Alpha (ann., vs benchmark)", "alpha_ann", "pct"),
        ("Beta (vs benchmark)", "beta", "f2"),
        ("Information ratio", "information_ratio", "f2"),
    ]
    perf_rows = [
        {
            "label": label,
            "full": _metric(full_m, key, fmt),
            "oos": _metric(oos_m, key, fmt),
            "bench": _metric(bench_m, key, fmt),
        }
        for label, key, fmt in rows_spec
    ]

    # ---- Monte Carlo blocks ----
    mc_blocks: list[dict[str, str]] = []
    titles = {
        "trade_bootstrap": "Trade-order bootstrap (OOS trades; assumes independence - weakest evidence)",
        "block_bootstrap": "Block bootstrap on OOS daily returns (preserves clustering)",
        "cost_stress": "Cost stress (full re-runs at multiplied costs)",
        "delay_stress": "Execution delay stress (full re-runs)",
        "param_perturbation": "Parameter neighborhood (full re-runs at adjacent parameter sets)",
    }
    for row in mc_rows:
        stats = json.loads(row["summary_json"])
        method = row["method"]
        if "error" in stats:
            html = f"<p class='meta'>not available: {stats['error']}</p>"
        elif method in ("trade_bootstrap", "block_bootstrap"):
            html = _mc_table_html(
                [stats],
                [("dd_p50", "Median max DD", "pct"), ("dd_p95", "P95 max DD", "pct"),
                 ("dd_p99", "P99 max DD", "pct"), ("risk_of_ruin", "Risk of ruin*", "pct"),
                 ("prob_negative", "P(loss)", "pct"), ("prob_negative_cagr", "P(CAGR<0)", "pct")],
            ) + (f"<p class='meta'>*ruin = drawdown beyond -{stats.get('ruin_drawdown_pct')}%; "
                 f"{row['n_paths']} paths, seed {row['seed']}.</p>")
        elif method in ("cost_stress", "delay_stress"):
            key = "multiplier" if method == "cost_stress" else "extra_delay_bars"
            html = _mc_table_html(
                stats.get("table", []),
                [(key, "Stress level", "raw"), ("total_return", "Total return", "pct"),
                 ("cagr", "CAGR", "pct"), ("sharpe", "Sharpe", "f2")],
            )
        else:  # param_perturbation
            html = _mc_table_html(
                [stats],
                [("chosen_sharpe", "Chosen Sharpe", "f2"),
                 ("neighbor_median_sharpe", "Neighbor median Sharpe", "f2"),
                 ("neighbor_sharpe_ratio", "Ratio", "f2")],
            )
        mc_blocks.append({"title": titles.get(method, method), "html": html})

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    html_out = env.get_template("tearsheet.html.j2").render(
        e=dict(e),
        h=dict(h),
        gates=gates_payload["gates"],
        weaknesses=gates_payload["weaknesses"],
        regimes=gates_payload["regimes"],
        prior_trials=gates_payload.get("prior_family_trials", 0),
        perf_rows=perf_rows,
        folds=[dict(f) for f in folds],
        mc_blocks=mc_blocks,
        chart_equity=chart_equity,
        chart_dd=chart_dd,
        chart_oos=chart_oos,
        config_hash=runs["backtest"]["config_hash"][:16],
        git_sha=runs["backtest"]["git_sha"][:12],
        dirty_tree=bool(runs["backtest"]["dirty_tree"]),
        engine_version=ENGINE_VERSION,
        generated_at=utc_now_iso(),
        pessimism=gates_payload.get("pessimism_multiplier", "configured"),
        synthetic=synthetic,
    )
    out_path = artifact_dir / "report.html"
    out_path.write_text(html_out, encoding="utf-8")
    return out_path
