"""Data validation suite.

Every check returns QualityIssue records with severity info/warn/fail. A symbol
with any 'fail' issue cannot be written to the curated store - bad data stops
at the door, because every downstream guarantee assumes clean bars.

Checks:
- schema/positivity: OHLC > 0, low <= open/close <= high
- calendar: monotonic unique dates, weekend bars flagged
- gaps: business-day gaps longer than the configured maximum
- spikes: robust z-score (median/MAD) of daily log returns
- split echo: ~1/2, 1/3, 2x, 3x close ratios with a volume spike (likely
  unadjusted corporate action in vendor data)
- volume: 20-day average below floor
- length: history shorter than the configured minimum
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.foundation.config import DataValidationConfig


@dataclass(frozen=True)
class QualityIssue:
    check: str
    severity: str  # 'info' | 'warn' | 'fail'
    detail: str
    range_start: str | None = None
    range_end: str | None = None


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def validate_frame(df: pd.DataFrame, cfg: DataValidationConfig) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    if len(df) < cfg.min_bars:
        issues.append(
            QualityIssue("length", "fail", f"only {len(df)} bars (< {cfg.min_bars} minimum)")
        )
        return issues  # too short to run the statistical checks meaningfully

    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        issues.append(QualityIssue("calendar", "fail", "dates not strictly increasing/unique"))
        return issues

    # OHLC sanity
    bad_pos = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if len(bad_pos):
        issues.append(
            QualityIssue(
                "positivity", "fail", f"{len(bad_pos)} bars with non-positive prices",
                _iso(bad_pos.index[0]), _iso(bad_pos.index[-1]),
            )
        )
    bad_ohlc = df[
        (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
        | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
    ]
    if len(bad_ohlc):
        issues.append(
            QualityIssue(
                "ohlc_bounds", "fail", f"{len(bad_ohlc)} bars violate low<=open/close<=high",
                _iso(bad_ohlc.index[0]), _iso(bad_ohlc.index[-1]),
            )
        )

    weekend = df[df.index.dayofweek >= 5]
    if len(weekend):
        issues.append(QualityIssue("weekend_bars", "warn", f"{len(weekend)} weekend bars present"))

    # Gap check on business days
    bdays = pd.bdate_range(df.index[0], df.index[-1])
    missing = bdays.difference(df.index)
    if len(missing):
        # group consecutive missing business days into runs
        runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        start = prev = missing[0]
        for d in missing[1:]:
            if (d - prev).days > 3:  # allow weekend in between
                runs.append((start, prev))
                start = d
            prev = d
        runs.append((start, prev))
        long_runs = [
            (a, b) for a, b in runs if len(pd.bdate_range(a, b)) > cfg.max_consecutive_gap_days
        ]
        if long_runs:
            a, b = long_runs[0]
            issues.append(
                QualityIssue(
                    "gaps", "fail",
                    f"{len(long_runs)} gaps longer than {cfg.max_consecutive_gap_days} business days",
                    _iso(a), _iso(b),
                )
            )
        else:
            issues.append(
                QualityIssue("gaps", "info", f"{len(runs)} short gaps (holidays/halts), within limit")
            )

    # Spike check: robust z-score of daily log returns. We must distinguish two
    # very different things a large move can be:
    #   - BAD PRINT: an isolated price-feed error. It REVERSES the next day
    #     (round-trips), trades on ORDINARY volume, and is extreme even against
    #     its own LOCAL neighborhood (it sits in an otherwise calm window). Only
    #     this combination is treated as a data error -> 'fail'.
    #   - REAL EXTREME MOVE: e.g. the 2008 and 2020 crisis days, AAPL's 2000
    #     profit warning (-52%), MSFT's +13%/-16% COVID whipsaw. Real
    #     dislocations CLUSTER - they are surrounded by other large moves, so
    #     they are NOT locally extreme - and they come with volume surges.
    #     Genuine market history -> 'warn', never 'fail'.
    # (Unadjusted splits are caught separately by the split-echo check below.)
    logret = np.log(df["close"]).diff()
    clean = logret.dropna()
    if len(clean) > 20:
        med = float(clean.median())
        mad = float((clean - med).abs().median())
        scale = mad * 1.4826 if mad > 0 else float(clean.std()) or 1e-9
        z = (logret - med).abs() / scale
        candidates = z[z > cfg.spike_zscore].dropna()
        vol_median = df["volume"].rolling(20, min_periods=5).median()
        bad_prints: list[pd.Timestamp] = []
        real_moves: list[pd.Timestamp] = []
        values = logret.to_numpy()
        for ts in candidates.index:
            i = int(logret.index.get_loc(ts))
            this = values[i]
            nxt = values[i + 1] if i + 1 < len(values) else 0.0
            round_trip = this * nxt < 0 and abs(this + nxt) < 0.4 * abs(this)

            med_vol = vol_median.iloc[i]
            vol_surge = bool(med_vol > 0 and df["volume"].iloc[i] > 1.5 * med_vol)

            # local robustness: extreme even vs the +/-10-bar neighborhood?
            # real dislocations cluster (neighbors also volatile) -> modest
            # local z; an isolated glitch in a calm window -> huge local z.
            lo, hi = max(0, i - 10), min(len(values), i + 11)
            local = np.concatenate([values[lo:i], values[i + 1 : hi]])
            local = local[~np.isnan(local)]
            if len(local) >= 5:
                lmed = float(np.median(local))
                lmad = float(np.median(np.abs(local - lmed))) * 1.4826
                local_z = abs(this - lmed) / lmad if lmad > 0 else float("inf")
            else:
                local_z = float("inf")

            # a data error reverses, on ordinary volume, AND is locally isolated
            if round_trip and not vol_surge and local_z > cfg.spike_zscore:
                bad_prints.append(ts)
            else:
                real_moves.append(ts)
        if bad_prints:
            issues.append(
                QualityIssue(
                    "spikes", "fail",
                    f"{len(bad_prints)} single-day spike(s) that reverse the next day "
                    f"(bad-print signature) - inspect/repair before use",
                    _iso(bad_prints[0]), _iso(bad_prints[-1]),
                )
            )
        if real_moves:
            issues.append(
                QualityIssue(
                    "spikes", "warn",
                    f"{len(real_moves)} large but persistent move(s) beyond "
                    f"{cfg.spike_zscore:.0f} robust z (kept as genuine market events)",
                    _iso(real_moves[0]), _iso(real_moves[-1]),
                )
            )

    # Split echo: close ratio near 1/2, 1/3, 2, 3 with volume spike
    ratio = (df["close"] / df["close"].shift(1)).dropna()
    vol_ratio = (df["volume"] / df["volume"].rolling(20).median()).reindex(ratio.index)
    suspects = ratio[
        (np.isclose(ratio, 0.5, rtol=0.02) | np.isclose(ratio, 2.0, rtol=0.02)
         | np.isclose(ratio, 1 / 3, rtol=0.02) | np.isclose(ratio, 3.0, rtol=0.02))
        & (vol_ratio > 3)
    ]
    if len(suspects):
        issues.append(
            QualityIssue(
                "split_echo", "fail",
                f"{len(suspects)} bars look like unadjusted splits (price halves/doubles with volume spike)",
                _iso(suspects.index[0]), _iso(suspects.index[-1]),
            )
        )

    avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
    if avg_vol < cfg.min_avg_volume:
        issues.append(
            QualityIssue("volume", "warn", f"20-day avg volume {avg_vol:,.0f} below floor {cfg.min_avg_volume:,.0f}")
        )

    return issues


def passes(issues: list[QualityIssue]) -> bool:
    return not any(i.severity == "fail" for i in issues)
