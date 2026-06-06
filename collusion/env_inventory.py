"""Extended market-making game with a stochastic mid-price and inventory.

This adds the realism the stylized game omits, the reviewer's first-choice
extension, while keeping benchmarks computable. Relative to ``MarketMakingGame``:

- The mid-price follows a random walk (volatility ``sigma_mid``).
- Order flow each period has a random buy/sell imbalance, so the maker that
  captures the flow accumulates inventory.
- Inventory is marked to market against mid moves (risk), penalized quadratically
  (``inventory_penalty``), and hard-capped (``inventory_cap``).

Spread capture is unchanged (the winner earns the half-spread, plus rebate, minus
adverse selection, on its volume), so the Bertrand tension is preserved. Because
the mid random walk is mean-zero and the penalty is symmetric, symmetric play
keeps inventory mean-zero; we compute the competitive and monopoly benchmarks by
simulating symmetric play at each grid spread, so the collusion index remains
well defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryGame:
    n_makers: int = 2
    spread_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
    q0: float = 1.0
    elasticity: float = 1.0
    adverse_frac: float = 0.5
    adverse_cost: float = 0.2
    maker_rebate: float = 0.0
    tie_rule: str = "split"
    # new dynamics
    sigma_mid: float = 0.3          # mid-price random-walk volatility
    imbalance_std: float = 0.5      # std of per-period buy/sell imbalance in [-1,1]
    inventory_penalty: float = 0.3   # kappa: quadratic inventory cost per period
    inventory_decay: float = 0.2     # per-period hedging/offload toward zero
    inventory_cap: float = 20.0
    n_inv_bins: int = 5             # inventory discretization for the RL state

    @property
    def n_actions(self) -> int:
        return len(self.spread_grid)

    def _volume(self, best_spread: float) -> float:
        return float(self.q0 * np.exp(-self.elasticity * best_spread))

    def _unit_margin(self, spread: float) -> float:
        return float(spread - self.adverse_frac * self.adverse_cost + self.maker_rebate)

    def inv_bin(self, inv: float) -> int:
        """Discretize inventory into a signed bin index for the RL state."""
        edges = np.linspace(-self.inventory_cap, self.inventory_cap, self.n_inv_bins + 1)
        return int(np.clip(np.digitize(inv, edges[1:-1]), 0, self.n_inv_bins - 1))

    def step(self, actions: np.ndarray, inventory: np.ndarray, rng: np.random.RandomState):
        """Advance one period. Returns (profit, new_inventory)."""
        spreads = np.array([self.spread_grid[int(a)] for a in actions], dtype=float)
        best = spreads.min()
        volume = self._volume(best)
        winners = np.flatnonzero(np.isclose(spreads, best))
        # buy/sell imbalance drives inventory accumulation for the winner(s).
        imb = float(np.clip(rng.normal(0.0, self.imbalance_std), -1.0, 1.0))
        net_flow = volume * imb  # >0 means net buying pressure -> winner sells, goes short
        dm = self.sigma_mid * rng.normal()  # mean-zero mid move (inventory risk)

        profit = np.zeros(self.n_makers, dtype=float)
        # mark existing inventory to the mid move and penalize it (all makers).
        profit += inventory * dm - self.inventory_penalty * inventory ** 2
        # hedging: inventory mean-reverts toward zero each period.
        new_inv = (1.0 - self.inventory_decay) * inventory.astype(float)

        if self.tie_rule == "winner_take_all" and len(winners) > 1:
            winners = np.array([int(winners[rng.randint(len(winners))])])
        share_vol = volume / len(winners)
        share_flow = net_flow / len(winners)
        for w in winners:
            profit[w] += self._unit_margin(spreads[w]) * share_vol
            new_inv[w] = float(np.clip(new_inv[w] - share_flow, -self.inventory_cap, self.inventory_cap))
        return profit, new_inv

    # ── Benchmarks via symmetric-play simulation ────────────────────────────────

    def symmetric_profit(self, action: int, periods: int = 20_000, seed: int = 0) -> float:
        """Average per-maker profit when all makers post the same spread."""
        rng = np.random.RandomState(seed)
        actions = np.full(self.n_makers, int(action))
        inv = np.zeros(self.n_makers)
        total = 0.0
        for _ in range(periods):
            profit, inv = self.step(actions, inv, rng)
            total += profit.mean()
        return total / periods

    def benchmarks(self, periods: int = 20_000) -> dict[str, float]:
        profits = [self.symmetric_profit(a, periods=periods, seed=1) for a in range(self.n_actions)]
        mono_a = int(np.argmax(profits))
        # competitive: tightest spread that is still (weakly) profitable in symmetric play
        positive = [a for a in range(self.n_actions) if profits[a] > 0]
        nash_a = int(min(positive)) if positive else int(np.argmax(profits))
        return {
            "nash_action": nash_a, "nash_spread": self.spread_grid[nash_a], "nash_profit": profits[nash_a],
            "monopoly_action": mono_a, "monopoly_spread": self.spread_grid[mono_a], "monopoly_profit": profits[mono_a],
        }

    def collusion_index(self, observed_profit: float, benchmarks: dict | None = None) -> float:
        b = benchmarks or self.benchmarks()
        denom = b["monopoly_profit"] - b["nash_profit"]
        if abs(denom) < 1e-9:
            return float("nan")
        return float((observed_profit - b["nash_profit"]) / denom)
