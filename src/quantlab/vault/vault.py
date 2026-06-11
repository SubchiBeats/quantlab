"""Research vault: the platform's append-only institutional memory.

Everything here writes to tables whose UPDATE/DELETE paths are blocked by
database triggers (see 001_initial.sql). Key behaviors:

- Hypotheses are pre-registered with success/falsification criteria BEFORE any
  run; the experiment engine refuses to run without one.
- At experiment registration, the vault searches prior FAILED experiments in
  the same family and via full-text similarity over hypothesis statements.
  If matches exist, registration requires an acknowledgment journal entry -
  'the platform must never forget failed experiments' made operational.
- Family trial counts accumulate across experiments (human- or AI-authored)
  and feed the deflated Sharpe ratio, so re-mining the same vein gets harder
  with every attempt.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from quantlab.backtest.engine import Trade
from quantlab.foundation.audit import append_event
from quantlab.foundation.clock import utc_now_iso
from quantlab.validation.montecarlo import McSummary
from quantlab.validation.walkforward import FoldResult


class SimilarFailuresError(Exception):
    """Raised when similar failed experiments exist and no acknowledgment was given."""

    def __init__(self, similar: list[dict[str, Any]]):
        self.similar = similar
        lines = "\n".join(
            f"  - experiment {s['experiment_id']} (hypothesis {s['hypothesis_id']}, "
            f"family '{s['family']}'): {s['reason']}"
            for s in similar
        )
        super().__init__(
            "Similar FAILED experiments exist in the vault:\n"
            f"{lines}\n"
            "Review them, then acknowledge with a journal entry "
            "(quantlab journal add \"...\") and pass --ack-journal <id>."
        )


# ------------------------------ journal ----------------------------------

def add_journal(
    conn: sqlite3.Connection, author: str, body: str,
    tags: str | None = None, linked: list[str] | None = None,
) -> int:
    if not body.strip():
        raise ValueError("journal entries cannot be empty")
    cur = conn.execute(
        "INSERT INTO journal_entries(ts, author, body, tags, linked_json) VALUES (?,?,?,?,?)",
        (utc_now_iso(), author, body.strip(), tags, json.dumps(linked or [])),
    )
    conn.commit()
    journal_id = int(cur.lastrowid or 0)
    append_event(conn, "human", "journal.add", {"journal_id": journal_id})
    return journal_id


# ----------------------------- hypotheses --------------------------------

def register_hypothesis(
    conn: sqlite3.Connection, family: str, statement: str, mechanism: str,
    success_criteria: str, falsification_criteria: str, author: str = "human",
) -> int:
    for label, value in (
        ("statement", statement), ("mechanism", mechanism),
        ("success_criteria", success_criteria), ("falsification_criteria", falsification_criteria),
    ):
        if not value.strip():
            raise ValueError(f"hypothesis {label} is required - pre-registration is not optional")
    cur = conn.execute(
        "INSERT INTO hypotheses(family, statement, mechanism, success_criteria,"
        " falsification_criteria, author, created_at) VALUES (?,?,?,?,?,?,?)",
        (family.strip(), statement.strip(), mechanism.strip(), success_criteria.strip(),
         falsification_criteria.strip(), author, utc_now_iso()),
    )
    conn.commit()
    hypothesis_id = int(cur.lastrowid or 0)
    append_event(conn, "human", "hypothesis.register",
                 {"hypothesis_id": hypothesis_id, "family": family})
    return hypothesis_id


def _fts_query(text: str) -> str:
    tokens = sorted({t.lower() for t in re.findall(r"[a-zA-Z]{4,}", text)})
    return " OR ".join(f'"{t}"' for t in tokens[:12])


def find_similar_failures(
    conn: sqlite3.Connection, family: str, statement: str, mechanism: str,
    exclude_hypothesis: int | None = None,
) -> list[dict[str, Any]]:
    """Failed experiments in the same family, plus FTS text matches on
    statement/mechanism of hypotheses behind failed experiments."""
    similar: dict[int, dict[str, Any]] = {}

    rows = conn.execute(
        "SELECT e.experiment_id, e.hypothesis_id, h.family, e.verdict_reason "
        "FROM experiments e JOIN hypotheses h ON h.hypothesis_id = e.hypothesis_id "
        "WHERE e.verdict = 'fail' AND h.family = ?",
        (family.strip(),),
    ).fetchall()
    for r in rows:
        similar[r["experiment_id"]] = {
            "experiment_id": r["experiment_id"], "hypothesis_id": r["hypothesis_id"],
            "family": r["family"], "reason": f"same family; failed: {r['verdict_reason']}",
        }

    query = _fts_query(f"{statement} {mechanism}")
    if query:
        rows = conn.execute(
            "SELECT e.experiment_id, e.hypothesis_id, h.family, e.verdict_reason, "
            "       bm25(hypotheses_fts) AS rank "
            "FROM hypotheses_fts f "
            "JOIN hypotheses h ON h.hypothesis_id = f.rowid "
            "JOIN experiments e ON e.hypothesis_id = h.hypothesis_id "
            "WHERE hypotheses_fts MATCH ? AND e.verdict = 'fail' "
            "ORDER BY rank LIMIT 10",
            (query,),
        ).fetchall()
        for r in rows:
            if exclude_hypothesis is not None and r["hypothesis_id"] == exclude_hypothesis:
                continue
            similar.setdefault(
                r["experiment_id"],
                {
                    "experiment_id": r["experiment_id"], "hypothesis_id": r["hypothesis_id"],
                    "family": r["family"],
                    "reason": f"similar hypothesis text; failed: {r['verdict_reason']}",
                },
            )
    return sorted(similar.values(), key=lambda s: s["experiment_id"])


# ----------------------------- experiments -------------------------------

def register_experiment(
    conn: sqlite3.Connection, hypothesis_id: int, strategy_name: str, strategy_version: int,
    snapshot_id: str, benchmark_symbol: str, ack_journal_id: int | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    hyp = conn.execute(
        "SELECT family, statement, mechanism FROM hypotheses WHERE hypothesis_id = ?",
        (hypothesis_id,),
    ).fetchone()
    if hyp is None:
        raise KeyError(
            f"hypothesis {hypothesis_id} not found - register one first "
            "(experiments cannot run without a pre-registered hypothesis)"
        )
    similar = find_similar_failures(
        conn, hyp["family"], hyp["statement"], hyp["mechanism"], exclude_hypothesis=hypothesis_id
    )
    if similar and ack_journal_id is None:
        raise SimilarFailuresError(similar)
    if ack_journal_id is not None:
        ok = conn.execute(
            "SELECT 1 FROM journal_entries WHERE journal_id = ?", (ack_journal_id,)
        ).fetchone()
        if ok is None:
            raise KeyError(f"acknowledgment journal entry {ack_journal_id} does not exist")

    cur = conn.execute(
        "INSERT INTO experiments(hypothesis_id, strategy_name, strategy_version, snapshot_id,"
        " benchmark_symbol, status, similar_failures_json, ack_journal_id, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (hypothesis_id, strategy_name, strategy_version, snapshot_id, benchmark_symbol,
         "registered", json.dumps(similar), ack_journal_id, utc_now_iso()),
    )
    conn.commit()
    experiment_id = int(cur.lastrowid or 0)
    append_event(conn, "system", "experiment.register",
                 {"experiment_id": experiment_id, "hypothesis_id": hypothesis_id,
                  "strategy": f"{strategy_name}@{strategy_version}", "snapshot": snapshot_id,
                  "similar_failures": [s["experiment_id"] for s in similar]})
    return experiment_id, similar


def mark_running(conn: sqlite3.Connection, experiment_id: int) -> None:
    conn.execute(
        "UPDATE experiments SET status = 'running' WHERE experiment_id = ?", (experiment_id,)
    )
    conn.commit()


def finalize_experiment(
    conn: sqlite3.Connection, experiment_id: int, verdict: str, reason: str
) -> None:
    conn.execute(
        "UPDATE experiments SET status = 'complete', verdict = ?, verdict_reason = ?,"
        " completed_at = ? WHERE experiment_id = ?",
        (verdict, reason, utc_now_iso(), experiment_id),
    )
    conn.commit()
    append_event(conn, "system", "experiment.verdict",
                 {"experiment_id": experiment_id, "verdict": verdict, "reason": reason})


# -------------------------------- runs -----------------------------------

def record_run(
    conn: sqlite3.Connection, run_id: str, experiment_id: int, run_type: str,
    config_hash: str, git_sha: str, dirty_tree: bool, seed: int, engine_version: str,
    started_at: str, artifact_dir: str | None,
) -> None:
    conn.execute(
        "INSERT INTO runs(run_id, experiment_id, run_type, config_hash, git_sha, dirty_tree,"
        " seed, engine_version, started_at, finished_at, artifact_dir, promotable)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, experiment_id, run_type, config_hash, git_sha, int(dirty_tree), seed,
         engine_version, started_at, utc_now_iso(), artifact_dir, int(not dirty_tree)),
    )
    conn.commit()


def record_metrics(
    conn: sqlite3.Connection, run_id: str, metrics: dict[str, float], period: str
) -> None:
    rows = [
        (run_id, name, period, None if value != value else float(value))
        for name, value in metrics.items()
    ]
    conn.executemany(
        "INSERT INTO run_metrics(run_id, metric, period, value) VALUES (?,?,?,?)", rows
    )
    conn.commit()


def record_trades(conn: sqlite3.Connection, run_id: str, trades: list[Trade]) -> None:
    rows = [
        (run_id, t.symbol, t.entry_ts.strftime("%Y-%m-%d"), t.entry_px,
         t.exit_ts.strftime("%Y-%m-%d"), t.exit_px, t.qty, t.fees, t.slippage, t.pnl,
         t.mae, t.mfe, t.holding_bars, t.exit_reason,
         json.dumps(t.context, sort_keys=True, default=str))
        for t in trades
    ]
    conn.executemany(
        "INSERT INTO trades(run_id, symbol, entry_ts, entry_px, exit_ts, exit_px, qty, fees,"
        " slippage, pnl, mae, mfe, holding_bars, exit_reason, context_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def record_folds(conn: sqlite3.Connection, run_id: str, folds: list[FoldResult]) -> None:
    rows = [
        (run_id, f.fold_n, f.train_start.strftime("%Y-%m-%d"), f.train_end.strftime("%Y-%m-%d"),
         f.test_start.strftime("%Y-%m-%d"), f.test_end.strftime("%Y-%m-%d"), f.embargo_bars,
         json.dumps(f.chosen_params, sort_keys=True),
         None if f.train_sharpe != f.train_sharpe else f.train_sharpe,
         None if f.test_sharpe != f.test_sharpe else f.test_sharpe,
         f.test_return, f.test_trades)
        for f in folds
    ]
    conn.executemany(
        "INSERT INTO walkforward_folds(run_id, fold_n, train_start, train_end, test_start,"
        " test_end, embargo_bars, chosen_params_json, train_sharpe, test_sharpe, test_return,"
        " test_trades) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def record_mc(conn: sqlite3.Connection, run_id: str, summary: McSummary) -> None:
    conn.execute(
        "INSERT INTO mc_simulations(run_id, method, n_paths, seed, summary_json, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (run_id, summary.method, summary.n_paths, summary.seed,
         json.dumps(summary.stats, sort_keys=True, default=str), utc_now_iso()),
    )
    conn.commit()


# --------------------------- family accounting ---------------------------

def family_prior_trials(
    conn: sqlite3.Connection, family: str, exclude_experiment: int | None = None
) -> int:
    """Sum of recorded walk-forward trial counts across all prior experiments in
    a hypothesis family. Feeds the deflated Sharpe N - every past attempt at
    this vein of research makes the current one harder to pass."""
    sql = (
        "SELECT COALESCE(SUM(m.value), 0) AS trials FROM run_metrics m "
        "JOIN runs r ON r.run_id = m.run_id "
        "JOIN experiments e ON e.experiment_id = r.experiment_id "
        "JOIN hypotheses h ON h.hypothesis_id = e.hypothesis_id "
        "WHERE m.metric = 'n_trials' AND r.run_type = 'walkforward' AND h.family = ?"
    )
    params: list[Any] = [family.strip()]
    if exclude_experiment is not None:
        sql += " AND e.experiment_id != ?"
        params.append(exclude_experiment)
    row = conn.execute(sql, params).fetchone()
    return int(row["trials"] or 0)


def family_prior_configs(
    conn: sqlite3.Connection, family: str, exclude_experiment: int | None = None
) -> int:
    """Sum of DISTINCT configurations tried across prior experiments in a family.

    This - not the grid x fold evaluation count - is the correct multiple-testing
    N for the deflated Sharpe ratio: it counts how many distinct strategy
    hypotheses have been selected among on this data. Every past attempt at the
    same vein of research still raises the bar for the next one, but walk-forward
    folds (a validation method) do not inflate it."""
    sql = (
        "SELECT COALESCE(SUM(m.value), 0) AS cfgs FROM run_metrics m "
        "JOIN runs r ON r.run_id = m.run_id "
        "JOIN experiments e ON e.experiment_id = r.experiment_id "
        "JOIN hypotheses h ON h.hypothesis_id = e.hypothesis_id "
        "WHERE m.metric = 'n_configs' AND r.run_type = 'walkforward' AND h.family = ?"
    )
    params: list[Any] = [family.strip()]
    if exclude_experiment is not None:
        sql += " AND e.experiment_id != ?"
        params.append(exclude_experiment)
    row = conn.execute(sql, params).fetchone()
    return int(row["cfgs"] or 0)
