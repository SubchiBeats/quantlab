"""Data ingestion adapters.

Sources:
- Tiingo (https://www.tiingo.com): free API key, split- AND dividend-adjusted
  daily OHLCV. Primary source for research: unadjusted prices make every
  backtest wrong at the first split.
- Stooq (https://stooq.com): free daily OHLCV CSV download, no API key.
- Local CSV import: any file with date/open/high/low/close/volume columns.

All produce the same normalized frame: DatetimeIndex (naive dates, ascending,
unique) with float columns open/high/low/close and int64 volume. The vendor
payload is also saved verbatim under datastore/raw/ - raw data is never edited.

Honesty note on adjusted prices: back-adjustment embeds knowledge of future
corporate actions into past prices. For total-return research on liquid
large-cap equities at daily frequency this is the standard, acceptable
trade-off; strictly point-in-time corporate-action handling is a Phase 2+
upgrade and is tracked as a known limitation.
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
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"


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


def parse_tiingo_json(rows: list[dict], symbol: str) -> pd.DataFrame:
    """Normalize a Tiingo daily-prices JSON payload using the adjusted columns."""
    if not rows:
        raise ValueError(f"Tiingo returned no rows for '{symbol}'")
    df = pd.DataFrame(rows)
    needed = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Tiingo payload for '{symbol}' missing columns {missing}")
    df = df[needed].rename(columns={
        "adjOpen": "open", "adjHigh": "high", "adjLow": "low",
        "adjClose": "close", "adjVolume": "volume",
    })
    # Tiingo dates are tz-aware ISO timestamps; bars are daily, keep the date only
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return normalize_frame(df, symbol)


def fetch_tiingo(symbol: str, token: str, timeout: float = 30.0) -> tuple[pd.DataFrame, str]:
    """Download full adjusted daily history for a US symbol from Tiingo.

    Requires TIINGO_API_KEY in .env (free tier at tiingo.com). The token goes
    in the Authorization header, never in the URL, so it cannot leak into the
    raw-payload archive or logs.
    """
    if not token:
        raise ValueError("Tiingo requires an API key: set TIINGO_API_KEY in .env")
    resp = httpx.get(
        TIINGO_URL.format(symbol=symbol.lower()),
        params={"startDate": "1990-01-01", "format": "json"},
        headers={"Authorization": f"Token {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_tiingo_json(resp.json(), symbol), resp.text


def read_csv_file(path: Path, symbol: str) -> tuple[pd.DataFrame, str]:
    text = Path(path).read_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(text))
    return normalize_frame(df, symbol), text


def save_raw(raw_dir: Path, source: str, symbol: str, raw_text: str, ext: str = "csv") -> Path:
    """Persist the vendor payload exactly as received (audit trail for data)."""
    target_dir = raw_dir / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{symbol.upper()}.{ext}"
    target.write_text(raw_text, encoding="utf-8")
    return target
