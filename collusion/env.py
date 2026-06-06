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
    # margin (a maker rebate); taker_fee raises the all-in price seen by order
    # flow; tie_rule sets how flow is allocated when makers post equal tightest
    # spreads.
    maker_rebate: float = 0.0
    taker_fee: float = 0.0
    tie_rule: str = "split"
    priority_share: float = 0.7
    allocation_noise: float = 12.0
    maker_rebates: tuple[float, ...] | None = None
    maker_costs: tuple[float, ...] | None = None
    latency_cost: float = 0.0

    def __post_init__(self) -> None:
        allowed = {
            "split",
            "proportional_split",
            "winner_take_all",
            "random_priority",
            "pro_rata_noise",
            "queue_priority",
            "persistent_queue",
        }
        if self.tie_rule not in allowed:
            raise ValueError(f"unknown tie_rule {self.tie_rule!r}; expected one of {sorted(allowed)}")
        if not 0.0 <= self.priority_share <= 1.0:
            raise ValueError("priority_share must be between 0 and 1")
        if self.allocation_noise <= 0:
            raise ValueError("allocation_noise must be positive")
        for name in ("maker_rebates", "maker_costs"):
            values = getattr(self, name)
            if values is None:
                continue
            values = tuple(float(v) for v in values)
            if len(values) != self.n_makers:
                raise ValueError(f"{name} must have length n_makers")
            object.__setattr__(self, name, values)

    @property
    def n_actions(self) -> int:
        return len(self.spread_grid)

    def _volume(self, best_spread: float) -> float:
        all_in_spread = max(0.0, best_spread + self.taker_fee)
        return float(self.q0 * np.exp(-self.elasticity * all_in_spread))

    def _maker_rebate(self, maker: int | None) -> float:
        if self.maker_rebates is None:
            return float(self.maker_rebate)
        if maker is None:
            return float(np.mean(self.maker_rebates))
        return float(self.maker_rebates[int(maker)])

    def _maker_cost(self, maker: int | None) -> float:
        if self.maker_costs is None:
            return 0.0
        if maker is None:
            return float(np.mean(self.maker_costs))
        return float(self.maker_costs[int(maker)])

    def _unit_margin(self, spread: float, maker: int | None = None) -> float:
        # Expected per-unit profit for a filled maker at this half-spread.
        return float(
            spread
            - self.adverse_frac * self.adverse_cost
            + self._maker_rebate(maker)
            - self._maker_cost(maker)
            - self.inventory_cost
        )

    def _tie_shares(
        self,
        winners: np.ndarray,
        rng: np.random.RandomState | None,
        queue_order: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(winners) == 1 or self.tie_rule in {"split", "proportional_split"}:
            return np.full(len(winners), 1.0 / len(winners), dtype=float)
        if self.tie_rule == "persistent_queue":
            if queue_order is None:
                return np.full(len(winners), 1.0 / len(winners), dtype=float)
            winner_pos = {int(w): j for j, w in enumerate(winners)}
            ordered = [int(m) for m in queue_order if int(m) in winner_pos]
            shares = np.full(len(winners), (1.0 - self.priority_share) / max(1, len(winners) - 1))
            shares[winner_pos[ordered[0]]] = self.priority_share
            if len(winners) == 1:
                shares[0] = 1.0
            return shares
        if self.tie_rule == "winner_take_all":
            if rng is None:
                return np.full(len(winners), 1.0 / len(winners), dtype=float)
            shares = np.zeros(len(winners), dtype=float)
            shares[int(rng.randint(len(winners)))] = 1.0
            return shares
        if self.tie_rule == "random_priority":
            if rng is None:
                return np.full(len(winners), 1.0 / len(winners), dtype=float)
            order = rng.permutation(len(winners))
            shares = np.full(len(winners), (1.0 - self.priority_share) / max(1, len(winners) - 1))
            shares[order[0]] = self.priority_share
            if len(winners) == 1:
                shares[0] = 1.0
            return shares
        if self.tie_rule == "pro_rata_noise":
            if rng is None:
                return np.full(len(winners), 1.0 / len(winners), dtype=float)
            return rng.dirichlet(np.full(len(winners), self.allocation_noise, dtype=float))
        if self.tie_rule == "queue_priority":
            shares = np.full(len(winners), (1.0 - self.priority_share) / max(1, len(winners) - 1))
            shares[0] = self.priority_share
            if len(winners) == 1:
                shares[0] = 1.0
            return shares
        raise RuntimeError(f"unhandled tie_rule {self.tie_rule!r}")

    def update_queue_order(
        self,
        actions: np.ndarray,
        prev_actions: np.ndarray | None,
        queue_order: np.ndarray | None,
    ) -> np.ndarray | None:
        """Update persistent queue priority after quote changes.

        Makers that keep their quote retain seniority; makers that change quote
        move behind stable makers while preserving their previous relative order.
        """

        if self.tie_rule != "persistent_queue":
            return queue_order
        if queue_order is None:
            queue_order = np.arange(self.n_makers, dtype=int)
        if prev_actions is None:
            return np.array(queue_order, dtype=int)
        changed = np.asarray(actions, dtype=int) != np.asarray(prev_actions, dtype=int)
        stable = [int(m) for m in queue_order if not changed[int(m)]]
        moved = [int(m) for m in queue_order if changed[int(m)]]
        return np.array(stable + moved, dtype=int)

    def step(
        self,
        actions: np.ndarray,
        rng: np.random.RandomState | None = None,
        prev_actions: np.ndarray | None = None,
        queue_order: np.ndarray | None = None,
    ) -> np.ndarray:
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
        shares = self._tie_shares(winners, rng, queue_order)
        for w, share in zip(winners, shares):
            profit[w] = self._unit_margin(spreads[w], int(w)) * volume * float(share)
        if self.latency_cost and prev_actions is not None:
            prev_spreads = np.array([self.spread_grid[int(a)] for a in prev_actions], dtype=float)
            profit -= self.latency_cost * np.abs(spreads - prev_spreads)
        return profit

    # ── Benchmarks ────────────────────────────────────────────────────────────

    def symmetric_profit(self, action: int) -> float:
        """Mean per-maker profit if ALL makers post the same spread."""

        spread = self.spread_grid[int(action)]
        volume = self._volume(spread)
        winners = np.arange(self.n_makers)
        shares = self._tie_shares(winners, None)
        profit = np.array(
            [self._unit_margin(spread, i) * volume * shares[i] for i in range(self.n_makers)],
            dtype=float,
        )
        return float(profit.mean())

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

        profitable = [a for a in range(self.n_actions) if self.symmetric_profit(a) > 0]
        if not profitable:
            return int(np.argmax([self.symmetric_profit(a) for a in range(self.n_actions)]))
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
