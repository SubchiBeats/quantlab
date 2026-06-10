"""Transaction cost model.

Components:
- commission: per-share with a minimum per order
- half-spread: you buy at the ask, sell at the bid (bps of price)
- impact: extra slippage proportional to participation (order qty / 20-day ADV)

Everything is scaled by pessimism_multiplier (>= 1.0, enforced by config) and
optionally by an extra stress multiplier during cost-stress validation. The
model is deliberately conservative: a strategy must survive overstated costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlab.foundation.config import CostConfig


@dataclass(frozen=True)
class CostModel:
    cfg: CostConfig
    stress_multiplier: float = 1.0

    @property
    def _scale(self) -> float:
        return self.cfg.pessimism_multiplier * self.stress_multiplier

    def commission(self, qty: int) -> float:
        if qty <= 0:
            return 0.0
        return max(self.cfg.min_commission, self.cfg.commission_per_share * qty) * self._scale

    def slip_fraction(self, qty: int, adv: float) -> float:
        """Fractional price concession for an order of `qty` shares."""
        participation = (qty / adv) if adv > 0 else 0.0
        return (self.cfg.half_spread_bps / 1e4 + self.cfg.impact_coeff * participation) * self._scale

    def buy_price(self, ref_price: float, qty: int, adv: float) -> float:
        return ref_price * (1.0 + self.slip_fraction(qty, adv))

    def sell_price(self, ref_price: float, qty: int, adv: float) -> float:
        return ref_price * (1.0 - self.slip_fraction(qty, adv))
