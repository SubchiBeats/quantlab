"""Vault: append-only enforcement, pre-registration discipline, the
never-forget-failures gate, and audit chain integrity."""

from __future__ import annotations

import sqlite3

import pytest

from quantlab.foundation.audit import append_event, verify_chain
from quantlab.foundation.clock import utc_now_iso
from quantlab.vault.vault import (
    SimilarFailuresError,
    add_journal,
    family_prior_trials,
    finalize_experiment,
    find_similar_failures,
    register_experiment,
    register_hypothesis,
)


def _add_snapshot(conn, snapshot_id: str = "snap0001") -> str:
    conn.execute(
        "INSERT INTO snapshots(snapshot_id, created_at, description, frequency, manifest_json)"
        " VALUES (?,?,?,?,?)",
        (snapshot_id, utc_now_iso(), "test", "daily", "{}"),
    )
    conn.commit()
    return snapshot_id


def test_journal_is_append_only(conn):
    journal_id = add_journal(conn, "tester", "first entry")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE journal_entries SET body='edited' WHERE journal_id=?", (journal_id,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM journal_entries WHERE journal_id=?", (journal_id,))


def test_hypotheses_require_all_fields(conn):
    with pytest.raises(ValueError, match="mechanism"):
        register_hypothesis(conn, "fam", "statement", "  ", "success", "falsify")


def test_experiment_requires_hypothesis(conn):
    _add_snapshot(conn)
    with pytest.raises(KeyError, match="hypothesis 99 not found"):
        register_experiment(conn, 99, "sma_cross", 1, "snap0001", "SPY")


def test_failed_experiments_are_never_deleted(conn):
    snap = _add_snapshot(conn)
    h = register_hypothesis(conn, "trend", "trends persist", "slow info diffusion", "s", "f")
    e, _ = register_experiment(conn, h, "sma_cross", 1, snap, "SPY")
    finalize_experiment(conn, e, "fail", "min_oos_trades")
    with pytest.raises(sqlite3.DatabaseError, match="never deleted"):
        conn.execute("DELETE FROM experiments WHERE experiment_id=?", (e,))
    # a recorded verdict is final
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        finalize_experiment(conn, e, "pass", "rewriting history")


def test_similar_failures_block_until_acknowledged(conn):
    snap = _add_snapshot(conn)
    h1 = register_hypothesis(conn, "trend", "momentum persists in equities",
                             "slow information diffusion", "s", "f")
    e1, _ = register_experiment(conn, h1, "sma_cross", 1, snap, "SPY")
    finalize_experiment(conn, e1, "fail", "failed OOS")

    # same family -> blocked without acknowledgment
    h2 = register_hypothesis(conn, "trend", "momentum persists with faster windows",
                             "slow information diffusion again", "s", "f")
    with pytest.raises(SimilarFailuresError):
        register_experiment(conn, h2, "sma_cross", 1, snap, "SPY")

    # acknowledged via journal -> allowed, similarity recorded
    ack = add_journal(conn, "tester", f"reviewed failed E{e1}; new variant differs by window")
    e2, similar = register_experiment(conn, h2, "sma_cross", 1, snap, "SPY", ack_journal_id=ack)
    assert any(s["experiment_id"] == e1 for s in similar)

    # different family but similar TEXT also matches via full-text search
    h3 = register_hypothesis(conn, "other-family", "equities momentum strategies persist",
                             "information diffusion is slow", "s", "f")
    found = find_similar_failures(conn, "other-family", "equities momentum strategies persist",
                                  "information diffusion is slow")
    assert any(s["experiment_id"] == e1 for s in found)


def test_ack_journal_must_exist(conn):
    snap = _add_snapshot(conn)
    h = register_hypothesis(conn, "x", "something", "reason", "s", "f")
    with pytest.raises(KeyError, match="journal entry 12345"):
        register_experiment(conn, h, "sma_cross", 1, snap, "SPY", ack_journal_id=12345)


def test_family_prior_trials_accumulate(conn):
    snap = _add_snapshot(conn)
    h = register_hypothesis(conn, "fam-a", "statement", "mech", "s", "f")
    e, _ = register_experiment(conn, h, "sma_cross", 1, snap, "SPY")
    from quantlab.vault.vault import record_metrics, record_run

    record_run(conn, "R1-walkforward-x", e, "walkforward", "cfg", "sha", False, 0, "1.0.0",
               utc_now_iso(), None)
    record_metrics(conn, "R1-walkforward-x", {"n_trials": 36.0}, "oos")
    assert family_prior_trials(conn, "fam-a") == 36
    assert family_prior_trials(conn, "fam-a", exclude_experiment=e) == 0
    assert family_prior_trials(conn, "fam-other") == 0


def test_runs_and_trades_append_only(conn):
    snap = _add_snapshot(conn)
    h = register_hypothesis(conn, "fam", "st", "mech", "s", "f")
    e, _ = register_experiment(conn, h, "sma_cross", 1, snap, "SPY")
    from quantlab.vault.vault import record_run

    record_run(conn, "R-test", e, "backtest", "cfg", "sha", False, 0, "1.0.0", utc_now_iso(), None)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE runs SET git_sha='forged' WHERE run_id='R-test'")


def test_audit_chain_verifies(conn):
    append_event(conn, "system", "test.one", {"a": 1})
    append_event(conn, "human", "test.two", {"b": 2})
    append_event(conn, "system", "test.three", {})
    ok, message = verify_chain(conn)
    assert ok, message
    # entries are immutable at the database level
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE audit_log SET payload_json='{}' WHERE seq=1")
