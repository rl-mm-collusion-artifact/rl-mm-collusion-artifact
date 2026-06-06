"""Parallel conditions sweep for the market-making collusion study.

For each market-structure condition (number of makers, adverse selection,
demand elasticity, exploration/learning) and seed, train the multi-agent
Q-learners and record the converged collusion index Delta. Maps the region
where tacit collusion emerges versus where competition wins. Cells are
independent and run across a process pool.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
import pandas as pd

from collusion.env import MarketMakingGame
from collusion.qlearning import MultiAgentQLearning, QLearningConfig


def run_cell(spec: dict) -> dict:
    from collusion.env import grid_from_tick

    grid_kwargs = {}
    if spec.get("tick_size"):
        grid_kwargs["spread_grid"] = grid_from_tick(spec["tick_size"])
    game = MarketMakingGame(
        n_makers=spec["n_makers"],
        elasticity=spec["elasticity"],
        adverse_frac=spec["adverse_frac"],
        adverse_cost=spec["adverse_cost"],
        maker_rebate=spec.get("maker_rebate", 0.0),
        tie_rule=spec.get("tie_rule", "split"),
        **grid_kwargs,
    )
    cfg = QLearningConfig(
        periods=spec["periods"],
        alpha=spec["alpha"],
        epsilon_decay=spec["epsilon_decay"],
        memory=spec.get("memory", 1),
        seed=spec["seed"],
    )
    learner = MultiAgentQLearning(game, cfg)
    result = learner.run()
    # Punishment magnitude (duopoly only): drop in the non-deviator's spread the
    # period after a forced undercut, normalized by the collusive spread.
    punish = float("nan")
    if spec["n_makers"] == 2:
        ir = learner.impulse_response(deviator=0, pre=4, post=4)
        pre_s = ir["punisher_spread_pre"]
        after_s = ir["punisher_spread_after"]
        if pre_s and pre_s > 0:
            punish = float((pre_s - after_s) / pre_s)
    out = {k: spec[k] for k in spec}
    out.update(
        {
            "collusion_index": result["collusion_index"],
            "mean_spread": result["mean_spread"],
            "avg_profit": result["avg_profit"],
            "nash_spread": result["benchmarks"]["nash_spread"],
            "monopoly_spread": result["benchmarks"]["monopoly_spread"],
            "punishment_magnitude": punish,
        }
    )
    return out


# Tabular state space is K^N, so high-N cells need more periods to converge.
PERIODS_FOR_N = {2: 500_000, 3: 1_000_000, 4: 2_500_000, 5: 5_000_000, 6: 8_000_000}


def _spec(sweep, n, af, ac, el, seed, eps=2e-5, alpha=0.125,
          maker_rebate=0.0, tie_rule="split", tick_size=None, memory=1):
    return {
        "sweep": sweep,
        "n_makers": n,
        "adverse_frac": af,
        "adverse_cost": ac,
        "elasticity": el,
        "epsilon_decay": eps,
        "alpha": alpha,
        "maker_rebate": maker_rebate,
        "tie_rule": tie_rule,
        "tick_size": tick_size,
        "memory": memory,
        "seed": seed,
        "periods": PERIODS_FOR_N.get(n, 500_000),
    }


def build_robustness_sweep(seeds=range(20)) -> list[dict]:
    """Robustness: learning rate, and memory (does collusion need memory?)."""

    specs = []
    for a in (0.05, 0.1, 0.125, 0.2):
        for s in seeds:
            specs.append(_spec("alpha", 2, 0.5, 0.2, 1.0, s, alpha=a))
    for m in (0, 1):
        for s in seeds:
            specs.append(_spec("memory", 2, 0.5, 0.2, 1.0, s, memory=m))
    return specs


def build_design_sweep(seeds=range(10)) -> list[dict]:
    """Market-design levers against collusion (the novel, prescriptive contribution)."""

    specs = []
    # Maker rebate: does a rebate break collusion?
    for rb in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        for s in seeds:
            specs.append(_spec("rebate", 2, 0.5, 0.2, 1.0, s, maker_rebate=rb))
    # Tick size: non-monotonic effect.
    for tk in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1):
        for s in seeds:
            specs.append(_spec("tick_size", 2, 0.5, 0.2, 1.0, s, tick_size=tk))
    # Tie-breaking rule: split vs winner-take-all.
    for tr in ("split", "winner_take_all"):
        for s in seeds:
            specs.append(_spec("tie_rule", 2, 0.5, 0.2, 1.0, s, tie_rule=tr))
    return specs


def build_full_sweep(seeds=range(8)) -> list[dict]:
    """Concentration headline plus duopoly parameter sweeps (one axis at a time)."""

    specs = []
    # 1) Concentration (the headline): number of makers, longer training for high N.
    for n in (2, 3, 4, 5):
        for s in seeds:
            specs.append(_spec("n_makers", n, 0.5, 0.2, 1.0, s))
    # 2) Adverse selection at duopoly: vary the adverse-selection cost.
    for ac in (0.0, 0.1, 0.2, 0.4, 0.6):
        for s in seeds:
            specs.append(_spec("adverse", 2, 0.5, ac, 1.0, s))
    # 3) Demand elasticity at duopoly.
    for el in (0.5, 0.75, 1.0, 1.5, 2.0):
        for s in seeds:
            specs.append(_spec("elasticity", 2, 0.5, 0.2, el, s))
    # 4) Exploration rate at duopoly (faster decay = less exploration).
    for eps in (1e-5, 2e-5, 5e-5, 1e-4):
        for s in seeds:
            specs.append(_spec("exploration", 2, 0.5, 0.2, 1.0, s, eps=eps))
    return specs


def build_specs(*, n_makers=(2, 3, 4, 5), seeds=range(8), periods=500_000, **_) -> list[dict]:
    specs = []
    for n, seed in itertools.product(n_makers, seeds):
        sp = _spec("n_makers", n, 0.5, 0.2, 1.0, seed)
        sp["periods"] = periods
        specs.append(sp)
    return specs


def run_sweep(specs: list[dict], *, max_workers: int | None = None) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_cell, s) for s in specs]
        for f in as_completed(futures):
            rows.append(f.result())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--periods", type=int, default=500_000)
    p.add_argument("--full", action="store_true", help="run the full conditions sweep")
    p.add_argument("--design", action="store_true", help="run the market-design lever sweep")
    p.add_argument("--robust", action="store_true", help="run the robustness sweep (alpha, memory)")
    p.add_argument("--out", default="collusion/results/sweep_nmakers.csv")
    args = p.parse_args()

    if args.robust:
        specs = build_robustness_sweep(seeds=range(args.seeds))
    elif args.design:
        specs = build_design_sweep(seeds=range(args.seeds))
    elif args.full:
        specs = build_full_sweep(seeds=range(args.seeds))
    else:
        specs = build_specs(seeds=range(args.seeds), periods=args.periods)
    print(f"running {len(specs)} cells ...", flush=True)
    df = run_sweep(specs, max_workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.robust:
        for name, axis in [("alpha", "alpha"), ("memory", "memory")]:
            sub = df[df["sweep"] == name]
            if not sub.empty:
                print(f"\n=== {name} ===", flush=True)
                print(sub.groupby(axis)[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
    elif args.design:
        for name, axis in [("rebate", "maker_rebate"), ("tick_size", "tick_size"), ("tie_rule", "tie_rule")]:
            sub = df[df["sweep"] == name]
            if not sub.empty:
                print(f"\n=== {name} ===", flush=True)
                print(sub.groupby(axis)[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
    elif args.full:
        for name, axis in [("n_makers", "n_makers"), ("adverse", "adverse_cost"),
                           ("elasticity", "elasticity"), ("exploration", "epsilon_decay")]:
            sub = df[df["sweep"] == name]
            if not sub.empty:
                print(f"\n=== {name} ===", flush=True)
                print(sub.groupby(axis)[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
    else:
        print(df.groupby("n_makers")[["collusion_index", "mean_spread", "punishment_magnitude"]].mean().round(3).to_string(), flush=True)
    print(f"\nwrote {args.out}", flush=True)
