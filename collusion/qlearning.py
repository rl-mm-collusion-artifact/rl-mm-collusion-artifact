"""Multi-agent tabular Q-learning for the market-making collusion game.

Follows Calvano et al. (2020): N independent Q-learners, state = the previous
period's joint action profile (memory 1), epsilon-greedy with exponentially
decaying exploration. The collusion index Delta is read off the converged
average profit. Interpretable Q-tables also let us run deviation-and-punishment
experiments (the signature evidence of tacit collusion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collusion.env import MarketMakingGame


def _profile_to_state(actions: np.ndarray, n_actions: int) -> int:
    """Encode a joint action profile as a single state index (mixed radix)."""

    state = 0
    for a in actions:
        state = state * n_actions + int(a)
    return state


@dataclass
class QLearningConfig:
    periods: int = 500_000
    alpha: float = 0.125          # learning rate
    gamma: float = 0.95           # discount
    epsilon0: float = 1.0         # initial exploration
    epsilon_decay: float = 2e-5   # beta in epsilon = exp(-beta * t)
    eval_window: int = 50_000     # periods averaged at the end for the result
    memory: int = 1               # 1 = state is last joint profile; 0 = stateless
    seed: int = 0


class MultiAgentQLearning:
    def __init__(self, game: MarketMakingGame, config: QLearningConfig) -> None:
        self.game = game
        self.cfg = config
        self.n = game.n_makers
        self.k = game.n_actions
        self.memory = int(config.memory)
        self.n_states = self.k ** self.n if self.memory == 1 else 1
        rng = np.random.RandomState(config.seed)
        # Optimistic-ish small random init.
        self.Q = rng.normal(0.0, 1e-3, size=(self.n, self.n_states, self.k))
        self.rng = rng

    def _state(self, actions: np.ndarray) -> int:
        # Memory one: state is the previous joint profile. Memory zero: stateless.
        return _profile_to_state(actions, self.k) if self.memory == 1 else 0

    def _greedy(self, state: int) -> np.ndarray:
        return self.Q[:, state, :].argmax(axis=1)

    def impulse_response(self, deviator: int = 0, *, pre: int = 5, post: int = 25) -> dict:
        """Force one maker to undercut for a single period; record everyone's
        greedy spread before, during, and after. Retaliation then reversion is
        the punishment signature of tacit collusion (Calvano's key evidence).

        Run only after training. Uses greedy (no-exploration) play throughout
        except the forced one-period deviation.
        """

        # Settle into the greedy steady state.
        actions = self._greedy(0)
        state = self._state(actions)
        for _ in range(200):
            actions = self._greedy(state)
            state = self._state(actions)
        spreads = []
        # pre periods of normal greedy play
        for _ in range(pre):
            actions = self._greedy(state)
            spreads.append([self.game.spread_grid[int(a)] for a in actions])
            state = self._state(actions)
        # one forced deviation: deviator undercuts to the tightest spread
        actions = self._greedy(state)
        actions[deviator] = 0
        spreads.append([self.game.spread_grid[int(a)] for a in actions])
        state = self._state(actions)
        # post periods of greedy play (watch for retaliation + reversion)
        for _ in range(post):
            actions = self._greedy(state)
            spreads.append([self.game.spread_grid[int(a)] for a in actions])
            state = self._state(actions)
        spreads = np.array(spreads)
        return {
            "deviator": deviator,
            "deviation_period": pre,
            "spreads": spreads,                       # shape (pre+1+post, n_makers)
            "punisher_spread_pre": float(spreads[:pre, 1 - deviator].mean()) if self.n == 2 else None,
            "punisher_spread_after": float(spreads[pre + 1, 1 - deviator]) if self.n == 2 else None,
        }

    def run(self) -> dict:
        cfg = self.cfg
        eval_window = min(cfg.eval_window, cfg.periods)
        # Start from a random profile.
        last_actions = self.rng.randint(0, self.k, size=self.n)
        queue_order = self.rng.permutation(self.n) if self.game.tie_rule == "persistent_queue" else None
        state = self._state(last_actions)
        profit_trace = np.zeros(eval_window, dtype=float)
        action_trace = np.zeros((eval_window, self.n), dtype=int)
        eval_start = cfg.periods - eval_window
        for t in range(cfg.periods):
            eps = cfg.epsilon0 * np.exp(-cfg.epsilon_decay * t)
            # Epsilon-greedy action selection per agent.
            greedy = self.Q[:, state, :].argmax(axis=1)
            explore = self.rng.random(self.n) < eps
            rand = self.rng.randint(0, self.k, size=self.n)
            actions = np.where(explore, rand, greedy)
            rewards = self.game.step(actions, self.rng, prev_actions=last_actions, queue_order=queue_order)
            next_state = self._state(actions)
            # Q-update per agent.
            for i in range(self.n):
                a = int(actions[i])
                best_next = self.Q[i, next_state, :].max()
                td = rewards[i] + cfg.gamma * best_next - self.Q[i, state, a]
                self.Q[i, state, a] += cfg.alpha * td
            state = next_state
            queue_order = self.game.update_queue_order(actions, last_actions, queue_order)
            last_actions = actions
            if t >= eval_start:
                j = t - eval_start
                profit_trace[j] = rewards.mean()
                action_trace[j] = actions

        avg_profit = float(profit_trace.mean())
        delta = self.game.collusion_index(avg_profit)
        # Greedy (post-exploration) profile and its spread.
        greedy_actions = self._greedy(state)
        mean_spread = float(np.mean([self.game.spread_grid[int(a)] for a in action_trace.reshape(-1)]))
        return {
            "avg_profit": avg_profit,
            "collusion_index": delta,
            "mean_spread": mean_spread,
            "greedy_profile": greedy_actions.tolist(),
            "benchmarks": self.game.benchmarks(),
        }
