"""Stylized repeated market-making game (the spread analog of Calvano's Bertrand
pricing game) for studying tacit collusion among independent RL market makers.

Each period, N market makers simultaneously post a half-spread from a discrete
grid. Order flow is elastic in the best posted spread (wider quotes, less flow);
the tightest quote captures the flow (Bertrand competition), ties split. A
fraction of flow is informed, imposing adverse selection on whoever is filled.
Per-period profit for a maker is captured half-spread times volume, minus
adverse selection and inventory cost.

The module also computes the two benchmarks that define the collusion index:
the competitive (Nash, zero-economic-profit undercutting) spread and the joint
monopoly (collusive) spread, so Delta = (observed - Nash)/(monopoly - Nash).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def grid_from_tick(tick: float, lo: float = 0.1, hi: float = 2.0) -> tuple[float, ...]:
    """Build an evenly-spaced spread grid at a given tick size (a design lever).

    Spacing is exactly ``tick`` and the grid never exceeds ``hi`` (the number of
    levels is floored), so coarse ticks yield fewer admissible spreads.
    """

    n = int((hi - lo) / tick + 1e-9) + 1
    return tuple(round(lo + i * tick, 6) for i in range(n))


@dataclass(frozen=True)
class MarketMakingGame:
    """A stylized repeated market-making game with elastic demand.

    Demand (total two-sided volume) as a function of the best (tightest) posted
    half-spread ``a_best`` is ``Q(a_best) = q0 * exp(-elasticity * a_best)``,
    decreasing in spread. The maker(s) posting the tightest spread split the
    volume. A fraction ``adverse_frac`` of fills is informed and costs
    ``adverse_cost`` per unit. Inventory cost is a small per-unit penalty.
    """

    n_makers: int = 2
    spread_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    q0: float = 1.0
    elasticity: float = 1.0
    adverse_frac: float = 0.0
    adverse_cost: float = 0.0
    inventory_cost: float = 0.0
    # Market-design levers (the prevention study). maker_rebate adds to per-fill
    # margin (a maker rebate); tie_rule sets how flow is allocated when makers
    # post equal tightest spreads ("split" or "winner_take_all").
    maker_rebate: float = 0.0
    tie_rule: str = "split"

    @property
    def n_actions(self) -> int:
        return len(self.spread_grid)

    def _volume(self, best_spread: float) -> float:
        return float(self.q0 * np.exp(-self.elasticity * best_spread))

    def _unit_margin(self, spread: float) -> float:
        # Expected per-unit profit for a filled maker at this half-spread.
        return float(
            spread - self.adverse_frac * self.adverse_cost + self.maker_rebate - self.inventory_cost
        )

    def step(self, actions: np.ndarray, rng: np.random.RandomState | None = None) -> np.ndarray:
        """Return per-maker profit for one period given action indices.

        With ``tie_rule == "winner_take_all"`` and an rng, equal tightest spreads
        award all flow to one randomly chosen maker (others get zero this period);
        in expectation this equals the split outcome, but the per-period
        winner-take-all makes the collusive equal-spread profile less stable.
        """

        spreads = np.array([self.spread_grid[int(a)] for a in actions], dtype=float)
        best = spreads.min()
        volume = self._volume(best)
        winners = np.flatnonzero(np.isclose(spreads, best))
        profit = np.zeros(self.n_makers, dtype=float)
        if self.tie_rule == "winner_take_all" and len(winners) > 1 and rng is not None:
            w = int(winners[rng.randint(len(winners))])
            profit[w] = self._unit_margin(spreads[w]) * volume
        else:
            share = volume / len(winners)
            for w in winners:
                profit[w] = self._unit_margin(spreads[w]) * share
        return profit

    # ── Benchmarks ────────────────────────────────────────────────────────────

    def symmetric_profit(self, action: int) -> float:
        """Per-maker profit if ALL makers post the same spread (split volume)."""

        spread = self.spread_grid[int(action)]
        volume = self._volume(spread)
        return self._unit_margin(spread) * volume / self.n_makers

    def joint_profit(self, action: int) -> float:
        """Total industry profit if all makers post the same spread."""

        return self.symmetric_profit(action) * self.n_makers

    def monopoly_action(self) -> int:
        """Spread index maximizing joint (industry) profit: the collusive target."""

        return int(np.argmax([self.joint_profit(a) for a in range(self.n_actions)]))

    def nash_action(self) -> int:
        """Competitive benchmark: the tightest spread that still yields non-negative
        per-unit margin. Bertrand undercutting drives makers here; posting tighter
        loses money, posting wider is undercut. Returns that spread's index."""

        profitable = [a for a in range(self.n_actions) if self._unit_margin(self.spread_grid[a]) > 0]
        if not profitable:
            return int(np.argmax([self._unit_margin(s) for s in self.spread_grid]))
        # tightest profitable spread = smallest spread with positive margin
        return int(min(profitable, key=lambda a: self.spread_grid[a]))

    def benchmarks(self) -> dict[str, float]:
        nash_a, mono_a = self.nash_action(), self.monopoly_action()
        nash_pi = self.symmetric_profit(nash_a)
        mono_pi = self.symmetric_profit(mono_a)
        return {
            "nash_action": nash_a,
            "nash_spread": self.spread_grid[nash_a],
            "nash_profit": nash_pi,
            "monopoly_action": mono_a,
            "monopoly_spread": self.spread_grid[mono_a],
            "monopoly_profit": mono_pi,
        }

    def collusion_index(self, observed_profit: float) -> float:
        """Calvano's Delta: 0 = competitive (Nash), 1 = fully collusive (monopoly)."""

        b = self.benchmarks()
        denom = b["monopoly_profit"] - b["nash_profit"]
        if abs(denom) < 1e-12:
            return float("nan")
        return float((observed_profit - b["nash_profit"]) / denom)
