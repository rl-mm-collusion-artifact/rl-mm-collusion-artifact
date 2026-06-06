"""Two-venue routing extension for the market-making collusion game.

Makers choose both a venue and a half-spread. Order flow routes to the venue
with the lowest effective half-spread, ``best_spread_v + taker_fee_v``; ties
across venues split flow. Within a routed venue, flow is allocated among the
best quotes using the same tie-allocation logic as ``MarketMakingGame``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collusion.env import MarketMakingGame


@dataclass(frozen=True)
class TwoVenueGame:
    n_makers: int = 2
    spread_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    q0: float = 1.0
    elasticity: float = 1.0
    adverse_selection_cost: float = 0.1
    inventory_cost: float = 0.0
    maker_rebates: tuple[float, float] = (0.0, 0.0)
    taker_fees: tuple[float, float] = (0.0, 0.0)
    tie_rule: str = "split"
    priority_share: float = 0.7
    allocation_noise: float = 12.0
    _allocator: MarketMakingGame = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.maker_rebates) != 2 or len(self.taker_fees) != 2:
            raise ValueError("maker_rebates and taker_fees must have length two")
        object.__setattr__(self, "maker_rebates", tuple(float(x) for x in self.maker_rebates))
        object.__setattr__(self, "taker_fees", tuple(float(x) for x in self.taker_fees))
        object.__setattr__(
            self,
            "_allocator",
            MarketMakingGame(
                n_makers=self.n_makers,
                tie_rule=self.tie_rule,
                priority_share=self.priority_share,
                allocation_noise=self.allocation_noise,
            ),
        )

    @property
    def n_spread_actions(self) -> int:
        return len(self.spread_grid)

    @property
    def n_actions(self) -> int:
        return 2 * self.n_spread_actions

    def action(self, venue: int, spread_action: int) -> int:
        if venue not in (0, 1):
            raise ValueError("venue must be 0 or 1")
        if not 0 <= int(spread_action) < self.n_spread_actions:
            raise ValueError("spread_action out of range")
        return int(venue) * self.n_spread_actions + int(spread_action)

    def decode_action(self, action: int) -> tuple[int, int]:
        a = int(action)
        return a // self.n_spread_actions, a % self.n_spread_actions

    def decode_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acts = np.asarray(actions, dtype=int)
        return acts // self.n_spread_actions, acts % self.n_spread_actions

    def _volume(self, effective_spread: float) -> float:
        return float(self.q0 * np.exp(-self.elasticity * max(0.0, effective_spread)))

    def _unit_margin(self, venue: int, spread: float) -> float:
        return float(
            spread
            - self.adverse_selection_cost
            + self.maker_rebates[int(venue)]
            - self.inventory_cost
        )

    def step(self, actions: np.ndarray, rng: np.random.RandomState | None = None) -> tuple[np.ndarray, dict]:
        venues, spread_actions = self.decode_actions(actions)
        spreads = np.array([self.spread_grid[int(a)] for a in spread_actions], dtype=float)
        best_spreads = np.full(2, np.inf, dtype=float)
        venue_winners: list[np.ndarray] = [np.array([], dtype=int), np.array([], dtype=int)]

        for v in (0, 1):
            makers = np.flatnonzero(venues == v)
            if len(makers) == 0:
                continue
            local_spreads = spreads[makers]
            best = float(local_spreads.min())
            best_spreads[v] = best
            venue_winners[v] = makers[np.isclose(local_spreads, best)]

        effective = best_spreads + np.array(self.taker_fees, dtype=float)
        best_effective = float(effective.min())
        routed_venues = np.flatnonzero(np.isclose(effective, best_effective))
        volume = self._volume(best_effective)
        venue_share = 1.0 / len(routed_venues)

        profit = np.zeros(self.n_makers, dtype=float)
        realized_spread = 0.0
        venue_a_flow_share = 0.0
        for v in routed_venues:
            v = int(v)
            winners = venue_winners[v]
            shares = self._allocator._tie_shares(winners, rng)
            routed_volume = volume * venue_share
            realized_spread += float(best_spreads[v]) * venue_share
            if v == 0:
                venue_a_flow_share += venue_share
            for w, share in zip(winners, shares):
                profit[int(w)] += self._unit_margin(v, spreads[int(w)]) * routed_volume * float(share)

        return profit, {
            "realized_spread": realized_spread,
            "effective_spread": best_effective,
            "venue_a_flow_share": venue_a_flow_share,
            "volume": volume,
        }

    # Benchmarks over symmetric venue-spread actions.

    def symmetric_profit(self, action: int) -> float:
        venue, spread_action = self.decode_action(action)
        spread = self.spread_grid[spread_action]
        effective = spread + self.taker_fees[venue]
        volume = self._volume(effective)
        winners = np.arange(self.n_makers)
        shares = self._allocator._tie_shares(winners, None)
        profit = self._unit_margin(venue, spread) * volume * shares
        return float(profit.mean())

    def joint_profit(self, action: int) -> float:
        return self.symmetric_profit(action) * self.n_makers

    def nash_action(self) -> int:
        candidates = [
            a for a in range(self.n_actions)
            if self._unit_margin(*self._spread_for_action(a)) > 0
        ]
        if not candidates:
            return int(np.argmax([self.symmetric_profit(a) for a in range(self.n_actions)]))
        min_effective = min(self.effective_spread_for_action(a) for a in candidates)
        tied = [a for a in candidates if np.isclose(self.effective_spread_for_action(a), min_effective)]
        return int(max(tied, key=self.symmetric_profit))

    def monopoly_action(self) -> int:
        return int(np.argmax([self.joint_profit(a) for a in range(self.n_actions)]))

    def _spread_for_action(self, action: int) -> tuple[int, float]:
        venue, spread_action = self.decode_action(action)
        return venue, self.spread_grid[spread_action]

    def effective_spread_for_action(self, action: int) -> float:
        venue, spread = self._spread_for_action(action)
        return float(spread + self.taker_fees[venue])

    def benchmarks(self) -> dict[str, float]:
        nash_a = self.nash_action()
        mono_a = self.monopoly_action()
        nash_v, nash_s = self._spread_for_action(nash_a)
        mono_v, mono_s = self._spread_for_action(mono_a)
        return {
            "nash_action": nash_a,
            "nash_venue": nash_v,
            "nash_spread": nash_s,
            "nash_effective_spread": self.effective_spread_for_action(nash_a),
            "nash_profit": self.symmetric_profit(nash_a),
            "monopoly_action": mono_a,
            "monopoly_venue": mono_v,
            "monopoly_spread": mono_s,
            "monopoly_effective_spread": self.effective_spread_for_action(mono_a),
            "monopoly_profit": self.symmetric_profit(mono_a),
        }

    def collusion_index(self, observed_profit: float, benchmarks: dict | None = None) -> float:
        b = benchmarks or self.benchmarks()
        denom = b["monopoly_profit"] - b["nash_profit"]
        if abs(denom) < 1e-12:
            return float("nan")
        return float((observed_profit - b["nash_profit"]) / denom)
