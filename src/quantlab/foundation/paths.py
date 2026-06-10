"""Filesystem layout for a QuantLab workspace.

A workspace root contains configs/, db/, datastore/{raw,curated,snapshots}/ and
artifacts/. Everything derives from one root so tests can run in a temp dir and
the real workspace can live anywhere (set QUANTLAB_HOME or pass --root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def strategy_configs(self) -> Path:
        return self.configs / "strategies"

    @property
    def db_file(self) -> Path:
        return self.root / "db" / "quantlab.sqlite"

    @property
    def raw(self) -> Path:
        return self.root / "datastore" / "raw"

    @property
    def curated(self) -> Path:
        return self.root / "datastore" / "curated" / "daily"

    @property
    def snapshots(self) -> Path:
        return self.root / "datastore" / "snapshots"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def ensure(self) -> "Paths":
        for p in (
            self.configs,
            self.strategy_configs,
            self.db_file.parent,
            self.raw,
            self.curated,
            self.snapshots,
            self.artifacts,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def resolve_root(explicit: str | os.PathLike[str] | None = None) -> Paths:
    """Workspace root resolution order: explicit arg > QUANTLAB_HOME env > cwd."""
    if explicit is not None:
        return Paths(Path(explicit).resolve())
    env = os.environ.get("QUANTLAB_HOME")
    if env:
        return Paths(Path(env).resolve())
    return Paths(Path.cwd().resolve())
