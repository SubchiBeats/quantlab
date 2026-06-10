"""SQLite access and migrations.

Design notes:
- WAL mode, foreign keys on, single-writer discipline (Phase 1 is a CLI, so
  there is one writer per process by construction).
- Migrations are plain numbered .sql files applied in order inside a
  transaction and recorded with a checksum; a changed historical migration is
  detected and refused. Plain SQL keeps the schema a human-auditable artifact.
- Append-only tables are protected by BEFORE UPDATE / BEFORE DELETE triggers
  that RAISE(ABORT, ...) - 'never forget failed experiments' is enforced by
  the database itself, not by convention.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from quantlab.foundation.clock import utc_now_iso

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(db_file: Path) -> sqlite3.Connection:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _assert_fts5(conn)
    return conn


def _assert_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts5_probe")
    except sqlite3.OperationalError as exc:  # pragma: no cover - depends on sqlite build
        raise RuntimeError(
            "This SQLite build lacks FTS5, which QuantLab needs for "
            "failure-similarity search. Use the standard python.org build."
        ) from exc


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations; return the versions applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    applied = {
        row["version"]: row["checksum"]
        for row in conn.execute("SELECT version, checksum FROM schema_migrations")
    }
    done: list[int] = []
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = int(sql_file.stem.split("_", 1)[0])
        sql = sql_file.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode()).hexdigest()
        if version in applied:
            if applied[version] != checksum:
                raise RuntimeError(
                    f"migration {version} changed after being applied "
                    f"({sql_file.name}); migrations are immutable - add a new one"
                )
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at, checksum) VALUES (?,?,?)",
            (version, utc_now_iso(), checksum),
        )
        conn.commit()
        done.append(version)
    return done


def open_db(db_file: Path) -> sqlite3.Connection:
    conn = connect(db_file)
    migrate(conn)
    return conn
