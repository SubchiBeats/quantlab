"""Time utilities. All platform timestamps are UTC, ISO-8601, seconds precision."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
