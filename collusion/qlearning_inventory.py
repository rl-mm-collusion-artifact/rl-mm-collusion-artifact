"""Multi-agent Q-learning for the inventory + stochastic-mid game.

Same algorithm as ``collusion.qlearning`` but the per-agent state augments the
previous joint action profile with the agent's own (binned) inventory, so makers
can learn inventory-dependent quoting. Inventory is carried across periods and
fed back through ``InventoryGame.step``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collusion.env_inventory import InventoryGame
from collusion.qlearning import _profile_to_state


@dataclass
class InvQLearningConfig:
    periods: int = 800_000
    alpha: float = 0.125
    gamma: float = 0.95
    epsilon0: float = 1.0
    epsilon_decay: float = 1.5e-5
    eval_window: int = 80_000
    seed: int = 0


class InventoryQLearning:
    def __init__(self, game: InventoryGame, config: InvQLearningConfig) -> None:
        self.game = game
        self.cfg = config
        self.n = game.n_makers
        self.k = game.n_actions
        self.b = game.n_inv_bins
        self.n_states = (self.k ** self.n) * self.b
        rng = np.random.RandomState(config.seed)
        self.Q = rng.normal(0.0, 1e-3, size=(self.n, self.n_states, self.k))
        self.rng = rng
        self.benchmarks = game.benchmarks()

    def _states(self, actions: np.ndarray, inventory: np.ndarray) -> np.ndarray:
        profile = _profile_to_state(actions, self.k)
        return np.array([profile * self.b + self.game.inv_bin(inventory[i]) for i in range(self.n)], dtype=int)

    def run(self) -> dict:
        cfg = self.cfg
        eval_window = min(cfg.eval_window, cfg.periods)
        actions = self.rng.randint(0, self.k, size=self.n)
        inv = np.zeros(self.n, dtype=float)
        states = self._states(actions, inv)
        profit_trace = np.zeros(eval_window, dtype=float)
        spread_trace = np.zeros((eval_window, self.n), dtype=int)
        eval_start = cfg.periods - eval_window
        for t in range(cfg.periods):
            eps = cfg.epsilon0 * np.exp(-cfg.epsilon_decay * t)
            greedy = np.array([self.Q[i, states[i], :].argmax() for i in range(self.n)])
            explore = self.rng.random(self.n) < eps
            rand = self.rng.randint(0, self.k, size=self.n)
            actions = np.where(explore, rand, greedy)
            rewards, inv = self.game.step(actions, inv, self.rng)
            next_states = self._states(actions, inv)
            for i in range(self.n):
                a = int(actions[i])
                best_next = self.Q[i, next_states[i], :].max()
                td = rewards[i] + cfg.gamma * best_next - self.Q[i, states[i], a]
                self.Q[i, states[i], a] += cfg.alpha * td
            states = next_states
            if t >= eval_start:
                j = t - eval_start
                profit_trace[j] = rewards.mean()
                spread_trace[j] = actions

        avg_profit = float(profit_trace.mean())
        delta = self.game.collusion_index(avg_profit, self.benchmarks)
        mean_spread = float(np.mean([self.game.spread_grid[int(a)] for a in spread_trace.reshape(-1)]))
        return {
            "avg_profit": avg_profit,
            "collusion_index": delta,
            "mean_spread": mean_spread,
            "benchmarks": self.benchmarks,
        }
