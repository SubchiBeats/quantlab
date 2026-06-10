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

    # Spike check: robust z-score of daily log returns
    logret = np.log(df["close"]).diff().dropna()
    if len(logret) > 20:
        med = float(logret.median())
        mad = float((logret - med).abs().median())
        scale = mad * 1.4826 if mad > 0 else float(logret.std()) or 1e-9
        z = (logret - med).abs() / scale
        hard = z[z > 2 * cfg.spike_zscore]
        soft = z[(z > cfg.spike_zscore) & (z <= 2 * cfg.spike_zscore)]
        if len(hard):
            issues.append(
                QualityIssue(
                    "spikes", "fail",
                    f"{len(hard)} returns beyond {2 * cfg.spike_zscore:.0f} robust z "
                    f"(max z={float(z.max()):.1f}); likely bad prints or unadjusted actions",
                    _iso(hard.index[0]), _iso(hard.index[-1]),
                )
            )
        elif len(soft):
            issues.append(
                QualityIssue("spikes", "warn", f"{len(soft)} returns beyond {cfg.spike_zscore:.0f} robust z")
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
