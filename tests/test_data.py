"""Data plane: the validation suite must catch every seeded defect, the curated
store must round-trip exactly, and snapshots must detect tampering."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.store import create_snapshot, load_curated, load_snapshot, write_curated
from quantlab.data.synth import generate_daily
from quantlab.data.validate import passes, validate_frame
from quantlab.foundation.config import DataValidationConfig

CFG = DataValidationConfig(
    max_consecutive_gap_days=5, spike_zscore=8.0, min_avg_volume=10_000, min_bars=300
)


def _clean(n: int = 600) -> pd.DataFrame:
    return generate_daily("CLEAN", n, seed=101)


def test_clean_data_passes():
    issues = validate_frame(_clean(), CFG)
    assert passes(issues), [f"{i.check}: {i.detail}" for i in issues if i.severity == "fail"]


def test_long_gap_detected():
    df = _clean()
    df = df.drop(df.index[200:215])  # 15 missing business days
    issues = validate_frame(df, CFG)
    assert any(i.check == "gaps" and i.severity == "fail" for i in issues)


def test_price_spike_detected():
    df = _clean()
    df.iloc[300, df.columns.get_indexer(["open", "high", "low", "close"])] *= 4.0
    issues = validate_frame(df, CFG)
    assert any(i.check == "spikes" and i.severity == "fail" for i in issues)


def test_unadjusted_split_detected():
    df = _clean()
    half = df.index[350]
    df.loc[df.index >= half, ["open", "high", "low", "close"]] /= 2.0
    df.loc[half, "volume"] = int(df["volume"].median() * 10)  # split-day volume surge
    issues = validate_frame(df, CFG)
    assert any(i.check == "split_echo" and i.severity == "fail" for i in issues)


def test_short_history_fails():
    issues = validate_frame(_clean(100), CFG)
    assert any(i.check == "length" and i.severity == "fail" for i in issues)


def test_negative_price_fails():
    df = _clean()
    df.iloc[100, df.columns.get_loc("low")] = -1.0
    issues = validate_frame(df, CFG)
    assert any(i.severity == "fail" for i in issues)


def test_curated_roundtrip(workspace):
    df = _clean()
    write_curated(workspace, "CLEAN", df)
    back = load_curated(workspace, "CLEAN")
    # check_freq=False: Parquet round-trips drop the cosmetic `freq` attribute
    # of the DatetimeIndex; all values, dtypes, and order must still be exact.
    pd.testing.assert_frame_equal(df, back, check_freq=False)


def test_snapshot_content_addressed_and_tamper_evident(workspace, conn):
    write_curated(workspace, "AAA", _clean())
    snap1 = create_snapshot(workspace, conn, ["AAA"], "first")
    # identical content -> identical snapshot id (content-addressed)
    assert create_snapshot(workspace, conn, ["AAA"], "again") == snap1

    data = load_snapshot(workspace, conn, snap1)
    assert "AAA" in data and len(data["AAA"]) == 600

    # tampering with a snapshot file is detected on load
    target = workspace.snapshots / snap1 / "AAA.parquet"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_snapshot(workspace, conn, snap1)


def test_snapshot_rows_immutable(workspace, conn):
    import sqlite3

    write_curated(workspace, "AAA", _clean())
    snap = create_snapshot(workspace, conn, ["AAA"], "first")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        conn.execute("DELETE FROM snapshots WHERE snapshot_id=?", (snap,))
