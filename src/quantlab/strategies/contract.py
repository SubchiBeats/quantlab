"""Strategy contract.

A strategy is a pure decision function: given a point-in-time view of the
market, return the desired position STATE per symbol ('long' or 'flat').
Everything else is deliberately someone else's job:

- position SIZING belongs to the engine's risk-based sizer, never the strategy
- order EXECUTION belongs to the engine (next-bar-open fills, costs, stops)
- the strategy never sees account state, so it cannot martingale or
  revenge-size; it expresses a market opinion and nothing more

Parameter discipline: each class declares its full parameter space upfront
(param_space). Instances may only use values drawn from that space - this is
enforced at construction and is how the platform counts the size of the search
space for multiple-testing corrections. Wanting a value outside the space
means declaring a new strategy version (a recorded, auditable event).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping

from quantlab.data.pit import PITView


@dataclass(frozen=True)
class Signal:
    symbol: str
    desired_state: Literal["long", "flat"]
    context: dict[str, Any] = field(default_factory=dict)  # indicator values behind the decision


class Strategy(ABC):
    """Base class. Subclasses set name/version and implement the hooks."""

    name: ClassVar[str]
    version: ClassVar[int]

    def __init__(self, params: Mapping[str, Any]):
        space = self.param_space()
        unknown = set(params) - set(space)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        missing = set(space) - set(params)
        if missing:
            raise ValueError(f"{self.name}: missing params {sorted(missing)}")
        for key, value in params.items():
            if value not in space[key]:
                raise ValueError(
                    f"{self.name}: {key}={value!r} is outside the declared space {space[key]} - "
                    "declare a new strategy version to widen the space"
                )
        if not self.params_valid(dict(params)):
            raise ValueError(f"{self.name}: invalid param combination {dict(params)}")
        self.params: dict[str, Any] = dict(params)

    # ---------------- declarations ----------------

    @classmethod
    @abstractmethod
    def param_space(cls) -> dict[str, list[Any]]:
        """Full declared parameter space: param name -> allowed values."""

    @classmethod
    def params_valid(cls, params: dict[str, Any]) -> bool:
        """Optional structural constraint (e.g. fast < slow). Used to filter grids."""
        return True

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Every valid parameter combination in the declared space."""
        keys = sorted(cls.param_space())
        combos: list[dict[str, Any]] = [{}]
        for key in keys:
            combos = [{**c, key: v} for c in combos for v in cls.param_space()[key]]
        return [c for c in combos if cls.params_valid(c)]

    # ---------------- behavior ----------------

    @abstractmethod
    def warmup_bars(self) -> int:
        """Bars of history needed before signals are meaningful."""

    @abstractmethod
    def signals(self, view: PITView, symbols: list[str]) -> list[Signal]:
        """Desired position state per symbol, given data up to view.as_of only."""
