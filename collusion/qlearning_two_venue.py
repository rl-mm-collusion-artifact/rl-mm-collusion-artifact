"""Tabular Q-learning for the two-venue routing extension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collusion.env_two_venue import TwoVenueGame
from collusion.qlearning import _profile_to_state


@dataclass
class TwoVenueQLearningConfig:
    periods: int = 500_000
    alpha: float = 0.125
    gamma: float = 0.95
    epsilon0: float = 1.0
    epsilon_decay: float = 2e-5
    eval_window: int = 50_000
    seed: int = 0


class TwoVenueQLearning:
    def __init__(self, game: TwoVenueGame, config: TwoVenueQLearningConfig) -> None:
        self.game = game
        self.cfg = config
        self.n = game.n_makers
        self.k = game.n_actions
        self.n_states = self.k ** self.n
        self.rng = np.random.RandomState(config.seed)
        self.Q = self.rng.normal(0.0, 1e-3, size=(self.n, self.n_states, self.k))
        self.benchmarks = game.benchmarks()

    def _state(self, actions: np.ndarray) -> int:
        return _profile_to_state(actions, self.k)

    def run(self) -> dict:
        cfg = self.cfg
        eval_window = min(cfg.eval_window, cfg.periods)
        actions = self.rng.randint(0, self.k, size=self.n)
        state = self._state(actions)
        profit_trace = np.zeros(eval_window, dtype=float)
        spread_trace = np.zeros(eval_window, dtype=float)
        effective_trace = np.zeros(eval_window, dtype=float)
        venue_a_trace = np.zeros(eval_window, dtype=float)
        eval_start = cfg.periods - eval_window

        for t in range(cfg.periods):
            eps = cfg.epsilon0 * np.exp(-cfg.epsilon_decay * t)
            greedy = self.Q[:, state, :].argmax(axis=1)
            explore = self.rng.random(self.n) < eps
            rand = self.rng.randint(0, self.k, size=self.n)
            actions = np.where(explore, rand, greedy)
            rewards, info = self.game.step(actions, self.rng)
            next_state = self._state(actions)
            for i in range(self.n):
                a = int(actions[i])
                best_next = self.Q[i, next_state, :].max()
                td = rewards[i] + cfg.gamma * best_next - self.Q[i, state, a]
                self.Q[i, state, a] += cfg.alpha * td
            state = next_state

            if t >= eval_start:
                j = t - eval_start
                profit_trace[j] = rewards.mean()
                spread_trace[j] = info["realized_spread"]
                effective_trace[j] = info["effective_spread"]
                venue_a_trace[j] = info["venue_a_flow_share"]

        avg_profit = float(profit_trace.mean())
        return {
            "avg_profit": avg_profit,
            "collusion_index": self.game.collusion_index(avg_profit, self.benchmarks),
            "realized_spread": float(spread_trace.mean()),
            "effective_spread": float(effective_trace.mean()),
            "venue_a_flow_share": float(venue_a_trace.mean()),
            "benchmarks": self.benchmarks,
        }
