"""Deterministic randomness.

Every stochastic component derives its own seed from the root seed plus string
labels, so a single root seed reproduces the entire platform's behavior and two
components can never accidentally share a stream.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(root_seed: int, *labels: str) -> int:
    h = hashlib.sha256()
    h.update(str(root_seed).encode())
    for label in labels:
        h.update(b"|")
        h.update(label.encode())
    # 8 bytes -> int in numpy's accepted seed range
    return int.from_bytes(h.digest()[:8], "big")


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
