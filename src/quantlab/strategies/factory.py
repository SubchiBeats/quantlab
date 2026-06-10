"""Strategy factory.

Strategies are instantiated ONLY from the registry by (name, version) plus a
parameter mapping - never from arbitrary code paths or dynamic imports. A run
is therefore fully reproducible from its config alone, and (in Phase 3) AI
proposals can only ever reference registered, version-pinned strategies.
"""

from __future__ import annotations

from typing import Any, Mapping

from quantlab.strategies.contract import Strategy

_REGISTRY: dict[tuple[str, int], type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    key = (cls.name, cls.version)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        raise ValueError(f"strategy {cls.name}@{cls.version} already registered by {existing}")
    _REGISTRY[key] = cls
    return cls


def get_class(name: str, version: int) -> type[Strategy]:
    try:
        return _REGISTRY[(name, version)]
    except KeyError:
        known = ", ".join(f"{n}@{v}" for n, v in sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown strategy {name}@{version}; registered: {known}") from None


def create(name: str, version: int, params: Mapping[str, Any]) -> Strategy:
    return get_class(name, version)(params)


def registered() -> list[tuple[str, int]]:
    return sorted(_REGISTRY)
