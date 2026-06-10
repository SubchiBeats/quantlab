"""Structured-enough logging: one stderr handler, consistent format, idempotent setup."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)sZ %(levelname)s %(name)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
        )
        root = logging.getLogger("quantlab")
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(f"quantlab.{name}")
