"""Small dependency-free DQN sanity check for the duopoly game.

This is not meant to replace the tabular learner used in the paper. It is a
lightweight function-approximation check: independent agents learn Q(s, a) with
one-hidden-layer neural nets, replay, and target networks, using only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collusion.env import MarketMakingGame


def _profile_to_state(actions: np.ndarray, n_actions: int) -> int:
    state = 0
    for a in actions:
        state = state * n_actions + int(a)
    return state


@dataclass
class DQNConfig:
    periods: int = 200_000
    gamma: float = 0.95
    lr: float = 0.01
    epsilon0: float = 1.0
    epsilon_decay: float = 2e-5
    eval_window: int = 20_000
    memory: int = 1
    seed: int = 0
    hidden: int = 32
    replay_size: int = 20_000
    batch_size: int = 64
    warmup: int = 1_000
    train_every: int = 4
    target_update: int = 1_000
    grad_clip: float = 5.0


class MultiAgentDQN:
    def __init__(self, game: MarketMakingGame, config: DQNConfig) -> None:
        self.game = game
        self.cfg = config
        self.n = game.n_makers
        self.k = game.n_actions
        self.memory = int(config.memory)
        self.n_states = self.k ** self.n if self.memory == 1 else 1
        self.rng = np.random.RandomState(config.seed)

        h = config.hidden
        self.W1 = self.rng.normal(0.0, 1.0 / np.sqrt(self.n_states), size=(self.n, self.n_states, h))
        self.b1 = np.zeros((self.n, h), dtype=float)
        self.W2 = self.rng.normal(0.0, 1.0 / np.sqrt(h), size=(self.n, h, self.k))
        self.b2 = np.zeros((self.n, self.k), dtype=float)
        self._copy_target()

        size = config.replay_size
        self.buf_state = np.zeros(size, dtype=int)
        self.buf_actions = np.zeros((size, self.n), dtype=int)
        self.buf_rewards = np.zeros((size, self.n), dtype=float)
        self.buf_next_state = np.zeros(size, dtype=int)
        self.buf_pos = 0
        self.buf_size = 0

    def _copy_target(self) -> None:
        self.tW1 = self.W1.copy()
        self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy()
        self.tb2 = self.b2.copy()

    def _state(self, actions: np.ndarray) -> int:
        return _profile_to_state(actions, self.k) if self.memory == 1 else 0

    def _q(self, agent: int, states: np.ndarray, *, target: bool = False) -> np.ndarray:
        W1, b1, W2, b2 = (self.tW1, self.tb1, self.tW2, self.tb2) if target else (self.W1, self.b1, self.W2, self.b2)
        z = W1[agent, states] + b1[agent]
        h = np.maximum(z, 0.0)
        return h @ W2[agent] + b2[agent]

    def _greedy(self, state: int) -> np.ndarray:
        actions = np.zeros(self.n, dtype=int)
        states = np.array([state], dtype=int)
        for i in range(self.n):
            actions[i] = int(np.argmax(self._q(i, states)[0]))
        return actions

    def _store(self, state: int, actions: np.ndarray, rewards: np.ndarray, next_state: int) -> None:
        j = self.buf_pos
        self.buf_state[j] = state
        self.buf_actions[j] = actions
        self.buf_rewards[j] = rewards
        self.buf_next_state[j] = next_state
        self.buf_pos = (self.buf_pos + 1) % self.cfg.replay_size
        self.buf_size = min(self.buf_size + 1, self.cfg.replay_size)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.cfg.grad_clip, self.cfg.grad_clip)

    def _train_batch(self) -> None:
        cfg = self.cfg
        batch = min(cfg.batch_size, self.buf_size)
        idx = self.rng.randint(0, self.buf_size, size=batch)
        states = self.buf_state[idx]
        actions = self.buf_actions[idx]
        rewards = self.buf_rewards[idx]
        next_states = self.buf_next_state[idx]
        rows = np.arange(batch)

        for i in range(self.n):
            z = self.W1[i, states] + self.b1[i]
            h = np.maximum(z, 0.0)
            q = h @ self.W2[i] + self.b2[i]
            q_selected = q[rows, actions[:, i]]
            target_next = self._q(i, next_states, target=True).max(axis=1)
            target = rewards[:, i] + cfg.gamma * target_next
            err = np.clip(q_selected - target, -10.0, 10.0)

            grad_q = np.zeros_like(q)
            grad_q[rows, actions[:, i]] = err / batch
            grad_W2 = h.T @ grad_q
            grad_b2 = grad_q.sum(axis=0)
            grad_h = grad_q @ self.W2[i].T
            grad_z = grad_h * (z > 0.0)
            grad_W1 = np.zeros_like(self.W1[i])
            np.add.at(grad_W1, states, grad_z)
            grad_b1 = grad_z.sum(axis=0)

            self.W2[i] -= cfg.lr * self._clip(grad_W2)
            self.b2[i] -= cfg.lr * self._clip(grad_b2)
            self.W1[i] -= cfg.lr * self._clip(grad_W1)
            self.b1[i] -= cfg.lr * self._clip(grad_b1)

    def run(self) -> dict:
        cfg = self.cfg
        eval_window = min(cfg.eval_window, cfg.periods)
        eval_start = cfg.periods - eval_window

        last_actions = self.rng.randint(0, self.k, size=self.n)
        state = self._state(last_actions)
        profit_trace = np.zeros(eval_window, dtype=float)
        action_trace = np.zeros((eval_window, self.n), dtype=int)

        for t in range(cfg.periods):
            eps = cfg.epsilon0 * np.exp(-cfg.epsilon_decay * t)
            greedy = self._greedy(state)
            explore = self.rng.random(self.n) < eps
            rand = self.rng.randint(0, self.k, size=self.n)
            actions = np.where(explore, rand, greedy)
            rewards = self.game.step(actions, self.rng, prev_actions=last_actions)
            next_state = self._state(actions)
            self._store(state, actions, rewards, next_state)

            if self.buf_size >= cfg.warmup and t % cfg.train_every == 0:
                self._train_batch()
            if t > 0 and t % cfg.target_update == 0:
                self._copy_target()

            state = next_state
            last_actions = actions
            if t >= eval_start:
                j = t - eval_start
                profit_trace[j] = rewards.mean()
                action_trace[j] = actions

        avg_profit = float(profit_trace.mean())
        return {
            "avg_profit": avg_profit,
            "collusion_index": self.game.collusion_index(avg_profit),
            "mean_spread": float(np.mean([self.game.spread_grid[int(a)] for a in action_trace.reshape(-1)])),
            "greedy_profile": self._greedy(state).tolist(),
            "benchmarks": self.game.benchmarks(),
        }
