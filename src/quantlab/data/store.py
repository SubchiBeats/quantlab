"""Curated Parquet store and content-addressed snapshots.

- Curated: one Parquet file per symbol under datastore/curated/daily/. Only
  validated data is written here (the CLI enforces validation before write).
- Snapshots: an immutable copy of chosen curated files under
  datastore/snapshots/<id>/, where <id> is the SHA-256 of the manifest
  (symbols + per-file hashes). Every research run binds to a snapshot id, so
  results stay reproducible even after the curated store is refreshed with
  newer data. Snapshot directories are never modified after creation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from quantlab.foundation.clock import utc_now_iso
from quantlab.foundation.paths import Paths


def write_curated(paths: Paths, symbol: str, df: pd.DataFrame) -> Path:
    target = paths.curated / f"{symbol.upper()}.parquet"
    df.to_parquet(target, engine="pyarrow", index=True)
    return target


def load_curated(paths: Paths, symbol: str) -> pd.DataFrame:
    target = paths.curated / f"{symbol.upper()}.parquet"
    if not target.exists():
        raise FileNotFoundError(f"no curated data for {symbol}; run `quantlab data fetch` first")
    return pd.read_parquet(target, engine="pyarrow")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_snapshot(
    paths: Paths, conn: sqlite3.Connection, symbols: list[str], description: str
) -> str:
    """Freeze the current curated files for `symbols` into an immutable snapshot."""
    files: list[dict[str, str]] = []
    coverage_start, coverage_end = None, None
    for sym in sorted({s.upper() for s in symbols}):
        src = paths.curated / f"{sym}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"cannot snapshot {sym}: no curated data")
        df = pd.read_parquet(src, engine="pyarrow")
        start, end = df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")
        coverage_start = min(coverage_start or start, start)
        coverage_end = max(coverage_end or end, end)
        files.append({"symbol": sym, "file": src.name, "sha256": _file_sha256(src)})

    manifest = {"frequency": "daily", "files": files}
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    snapshot_id = hashlib.sha256(manifest_json.encode()).hexdigest()[:16]

    snap_dir = paths.snapshots / snapshot_id
    if snap_dir.exists():
        return snapshot_id  # identical content already snapshotted
    snap_dir.mkdir(parents=True)
    for entry in files:
        shutil.copy2(paths.curated / entry["file"], snap_dir / entry["file"])

    conn.execute(
        "INSERT INTO snapshots(snapshot_id, created_at, description, frequency, manifest_json,"
        " coverage_start, coverage_end) VALUES (?,?,?,?,?,?,?)",
        (snapshot_id, utc_now_iso(), description, "daily", manifest_json, coverage_start, coverage_end),
    )
    conn.commit()
    return snapshot_id


def load_snapshot(paths: Paths, conn: sqlite3.Connection, snapshot_id: str) -> dict[str, pd.DataFrame]:
    """Load all symbols of a snapshot, verifying file hashes against the manifest."""
    row = conn.execute(
        "SELECT manifest_json FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown snapshot: {snapshot_id}")
    manifest = json.loads(row["manifest_json"])
    snap_dir = paths.snapshots / snapshot_id
    data: dict[str, pd.DataFrame] = {}
    for entry in manifest["files"]:
        path = snap_dir / entry["file"]
        if _file_sha256(path) != entry["sha256"]:
            raise RuntimeError(
                f"snapshot {snapshot_id} corrupted: {entry['file']} hash mismatch - "
                "restore from backup; snapshots are immutable"
            )
        data[entry["symbol"]] = pd.read_parquet(path, engine="pyarrow")
    return data
