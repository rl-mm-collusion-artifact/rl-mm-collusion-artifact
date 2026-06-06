"""Run a small tabular-vs-DQN sanity check in the duopoly.

The aim is directional validation, not a production deep-RL benchmark. The
conditions mirror the main design extensions and keep the same collusion-index
measurement as the tabular sweeps.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from collusion.dqn import DQNConfig, MultiAgentDQN
from collusion.env import MarketMakingGame
from collusion.qlearning import MultiAgentQLearning, QLearningConfig

BASE = dict(n_makers=2, elasticity=1.0, adverse_frac=0.5, adverse_cost=0.2)

CONDITIONS = [
    ("baseline", {}),
    ("maker_rebate_only", {"maker_rebate": 0.20}),
    ("taker_fee_only", {"taker_fee": 0.20}),
    ("random_priority", {"tie_rule": "random_priority"}),
    ("latency_05", {"latency_cost": 0.05}),
]


def _eval_window(periods: int) -> int:
    return max(1_000, min(20_000, periods // 5, periods))


def run_cell(condition: str, params: dict, algorithm: str, seed: int, periods: int) -> dict:
    game = MarketMakingGame(**BASE, **params)
    if algorithm == "tabular":
        cfg = QLearningConfig(periods=periods, eval_window=_eval_window(periods), seed=seed)
        result = MultiAgentQLearning(game, cfg).run()
    elif algorithm == "dqn":
        cfg = DQNConfig(
            periods=periods,
            eval_window=_eval_window(periods),
            replay_size=max(1_000, min(20_000, periods // 2)),
            warmup=max(100, min(1_000, periods // 10)),
            target_update=max(250, min(1_000, periods // 100)),
            seed=seed,
        )
        result = MultiAgentDQN(game, cfg).run()
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return {
        "condition": condition,
        "algorithm": algorithm,
        "seed": seed,
        "periods": periods,
        "collusion_index": result["collusion_index"],
        "mean_spread": result["mean_spread"],
        "avg_profit": result["avg_profit"],
        "nash_spread": result["benchmarks"]["nash_spread"],
        "monopoly_spread": result["benchmarks"]["monopoly_spread"],
    }


def run_sanity(
    seeds=range(5),
    periods: int = 100_000,
    algorithms=("tabular", "dqn"),
    *,
    max_workers: int | None = None,
) -> pd.DataFrame:
    jobs = [
        (condition, params, algorithm, int(seed), periods)
        for condition, params in CONDITIONS
        for algorithm in algorithms
        for seed in seeds
    ]
    rows = []
    if max_workers == 1:
        for job in jobs:
            rows.append(run_cell(*job))
        return pd.DataFrame(rows)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_cell, *job) for job in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--periods", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--algorithms", nargs="+", choices=["tabular", "dqn"], default=["tabular", "dqn"])
    p.add_argument("--out", default="collusion/results/deep_sanity.csv")
    args = p.parse_args()

    df = run_sanity(
        seeds=range(args.seeds),
        periods=args.periods,
        algorithms=tuple(args.algorithms),
        max_workers=args.workers,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        df.groupby(["condition", "algorithm"])[["collusion_index", "mean_spread"]]
        .mean()
        .round(3)
        .to_string(),
        flush=True,
    )
    print(f"\nwrote {args.out}", flush=True)
