"""Data ingestion adapters.

Sources:
- Stooq (https://stooq.com): free daily OHLCV CSV download, no API key.
- Local CSV import: any file with date/open/high/low/close/volume columns.

Both produce the same normalized frame: DatetimeIndex (naive dates, ascending,
unique) with float columns open/high/low/close and int64 volume. The vendor
payload is also saved verbatim under datastore/raw/ - raw data is never edited.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

from quantlab.foundation.logs import get_logger

log = get_logger("data.ingest")

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
STOOQ_URL = "https://stooq.com/q/d/l/"


def normalize_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Lower-case columns, parse dates, sort, de-duplicate, type-check."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{symbol}: no 'date' column in input")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    out = df[REQUIRED_COLS].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": "int64"}
    )
    if out.isna().any().any():
        raise ValueError(f"{symbol}: NaNs present after normalization")
    return out


def fetch_stooq(symbol: str, timeout: float = 30.0) -> tuple[pd.DataFrame, str]:
    """Download full daily history for a US symbol from Stooq.

    Returns (normalized frame, raw CSV text). Stooq uses suffixed tickers
    (AAPL -> aapl.us); we accept the plain US symbol and add the suffix.
    """
    stooq_symbol = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    resp = httpx.get(STOOQ_URL, params={"s": stooq_symbol, "i": "d"}, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text.lower().startswith("date,"):
        # Stooq returns an HTML page or 'No data' text for unknown symbols.
        raise ValueError(f"Stooq returned no data for '{symbol}' (tried '{stooq_symbol}')")
    df = pd.read_csv(io.StringIO(text))
    return normalize_frame(df, symbol), text


def read_csv_file(path: Path, symbol: str) -> tuple[pd.DataFrame, str]:
    text = Path(path).read_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(text))
    return normalize_frame(df, symbol), text


def save_raw(raw_dir: Path, source: str, symbol: str, raw_text: str) -> Path:
    """Persist the vendor payload exactly as received (audit trail for data)."""
    target_dir = raw_dir / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{symbol.upper()}.csv"
    target.write_text(raw_text, encoding="utf-8")
    return target
