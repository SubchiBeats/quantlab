"""Hash-chained audit log.

Each entry's hash covers (prev_hash, ts, actor, event_type, canonical payload),
so any tampering with history breaks the chain. `verify_chain` walks the whole
log and reports the first broken link, if any.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from quantlab.foundation.clock import utc_now_iso

GENESIS = "0" * 64


def _entry_hash(prev_hash: str, ts: str, actor: str, event_type: str, payload_json: str) -> str:
    h = hashlib.sha256()
    for part in (prev_hash, ts, actor, event_type, payload_json):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def append_event(
    conn: sqlite3.Connection, actor: str, event_type: str, payload: dict[str, Any]
) -> int:
    row = conn.execute("SELECT this_hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = row["this_hash"] if row else GENESIS
    ts = utc_now_iso()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    this_hash = _entry_hash(prev_hash, ts, actor, event_type, payload_json)
    cur = conn.execute(
        "INSERT INTO audit_log(ts, actor, event_type, payload_json, prev_hash, this_hash) "
        "VALUES (?,?,?,?,?,?)",
        (ts, actor, event_type, payload_json, prev_hash, this_hash),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def verify_chain(conn: sqlite3.Connection) -> tuple[bool, str]:
    """Return (ok, message). Recomputes every hash in sequence order."""
    prev = GENESIS
    for row in conn.execute(
        "SELECT seq, ts, actor, event_type, payload_json, prev_hash, this_hash "
        "FROM audit_log ORDER BY seq"
    ):
        if row["prev_hash"] != prev:
            return False, f"chain broken at seq {row['seq']}: prev_hash mismatch"
        expected = _entry_hash(prev, row["ts"], row["actor"], row["event_type"], row["payload_json"])
        if row["this_hash"] != expected:
            return False, f"chain broken at seq {row['seq']}: entry hash mismatch"
        prev = row["this_hash"]
    return True, "audit chain verified"
