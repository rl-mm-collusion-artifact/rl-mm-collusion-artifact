"""Correctness tests for the collusion study (env, benchmarks, levers, learning)."""

from __future__ import annotations

import unittest

import numpy as np

from collusion.env import MarketMakingGame, grid_from_tick
from collusion.env_inventory import InventoryGame
from collusion.qlearning import MultiAgentQLearning, QLearningConfig
from collusion.qlearning_inventory import InventoryQLearning, InvQLearningConfig


class EnvTests(unittest.TestCase):
    def test_benchmarks_bound_and_index(self) -> None:
        g = MarketMakingGame(n_makers=2, elasticity=1.0, adverse_frac=0.5, adverse_cost=0.2)
        b = g.benchmarks()
        self.assertGreater(b["monopoly_spread"], b["nash_spread"])
        self.assertGreater(b["monopoly_profit"], b["nash_profit"])
        self.assertAlmostEqual(g.collusion_index(b["nash_profit"]), 0.0, places=6)
        self.assertAlmostEqual(g.collusion_index(b["monopoly_profit"]), 1.0, places=6)

    def test_tightest_wins_and_ties_split(self) -> None:
        g = MarketMakingGame(n_makers=2)
        # equal spreads -> split
        p = g.step(np.array([3, 3]))
        self.assertAlmostEqual(p[0], p[1])
        # one undercuts (tighter) -> wins all the flow, other gets zero
        p = g.step(np.array([2, 4]))
        self.assertGreater(p[0], 0.0)
        self.assertEqual(p[1], 0.0)

    def test_monopoly_tracks_elasticity(self) -> None:
        # interior monopoly spread ~ 1/elasticity in this demand form
        self.assertGreater(
            MarketMakingGame(elasticity=0.5).benchmarks()["monopoly_spread"],
            MarketMakingGame(elasticity=2.0).benchmarks()["monopoly_spread"],
        )

    def test_grid_from_tick_spacing(self) -> None:
        grid = grid_from_tick(0.5, lo=0.1, hi=2.0)
        self.assertAlmostEqual(grid[0], 0.1)
        self.assertAlmostEqual(grid[1] - grid[0], 0.5)
        self.assertLessEqual(grid[-1], 2.0 + 1e-9)

    def test_maker_rebate_raises_margin(self) -> None:
        base = MarketMakingGame(adverse_frac=0.5, adverse_cost=0.2)
        reb = MarketMakingGame(adverse_frac=0.5, adverse_cost=0.2, maker_rebate=0.3)
        self.assertGreater(reb._unit_margin(0.5), base._unit_margin(0.5))

    def test_winner_take_all_one_winner(self) -> None:
        g = MarketMakingGame(n_makers=2, tie_rule="winner_take_all")
        rng = np.random.RandomState(0)
        p = g.step(np.array([3, 3]), rng)
        # exactly one maker gets all the flow at a tie
        self.assertEqual(int((p > 0).sum()), 1)


class InventoryEnvTests(unittest.TestCase):
    def test_benchmarks_bound_and_index(self) -> None:
        g = InventoryGame(n_makers=2)
        b = g.benchmarks(periods=5_000)
        self.assertGreater(b["monopoly_spread"], b["nash_spread"])
        self.assertGreater(b["monopoly_profit"], 0.0)
        self.assertGreater(b["nash_profit"], 0.0)
        self.assertAlmostEqual(g.collusion_index(b["nash_profit"], b), 0.0, places=6)
        self.assertAlmostEqual(g.collusion_index(b["monopoly_profit"], b), 1.0, places=6)

    def test_inventory_mean_reverts_without_flow(self) -> None:
        # With no new flow (no winner trades), inventory decays toward zero.
        g = InventoryGame(n_makers=2, inventory_decay=0.2, sigma_mid=0.0)
        rng = np.random.RandomState(0)
        inv = np.array([10.0, -10.0])
        # post identical wide spreads so a tie splits flow; track magnitude shrinking
        prev = np.abs(inv).sum()
        for _ in range(5):
            _, inv = g.step(np.array([8, 8]), inv, rng)
            cur = np.abs(inv).sum()
            self.assertLess(cur, prev + 1e-9)
            prev = cur
        self.assertLess(np.abs(inv).sum(), np.abs(np.array([10.0, -10.0])).sum())

    def test_winner_accumulates_inventory(self) -> None:
        # A maker that wins the flow moves its inventory away from zero on average.
        g = InventoryGame(n_makers=2, sigma_mid=0.0, inventory_decay=0.0, imbalance_std=0.5)
        rng = np.random.RandomState(1)
        inv = np.zeros(2)
        moved = 0.0
        for _ in range(200):
            _, inv = g.step(np.array([2, 5]), inv, rng)  # maker 0 always undercuts
            moved += abs(inv[0])
        self.assertGreater(moved, 0.0)

    def test_inv_bin_monotone_and_bounded(self) -> None:
        g = InventoryGame(n_inv_bins=5, inventory_cap=20.0)
        self.assertEqual(g.inv_bin(-100.0), 0)
        self.assertEqual(g.inv_bin(100.0), 4)
        self.assertLessEqual(g.inv_bin(-5.0), g.inv_bin(5.0))


class InventoryLearningTests(unittest.TestCase):
    def test_run_returns_finite_index(self) -> None:
        g = InventoryGame(n_makers=2)
        r = InventoryQLearning(g, InvQLearningConfig(periods=20_000, eval_window=5_000, seed=0)).run()
        self.assertTrue(np.isfinite(r["collusion_index"]))
        self.assertIn("benchmarks", r)


class LearningTests(unittest.TestCase):
    def test_run_returns_finite_index(self) -> None:
        g = MarketMakingGame(n_makers=2)
        r = MultiAgentQLearning(g, QLearningConfig(periods=20_000, eval_window=5_000, seed=0)).run()
        self.assertTrue(np.isfinite(r["collusion_index"]))
        self.assertIn("benchmarks", r)

    def test_impulse_response_shape(self) -> None:
        g = MarketMakingGame(n_makers=2)
        m = MultiAgentQLearning(g, QLearningConfig(periods=20_000, eval_window=5_000, seed=0))
        m.run()
        ir = m.impulse_response(deviator=0, pre=3, post=5)
        self.assertEqual(ir["spreads"].shape, (3 + 1 + 5, 2))


if __name__ == "__main__":
    unittest.main()
