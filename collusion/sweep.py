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
        taker_fee=spec.get("taker_fee", 0.0),
        tie_rule=spec.get("tie_rule", "split"),
        priority_share=spec.get("priority_share", 0.7),
        allocation_noise=spec.get("allocation_noise", 12.0),
        maker_rebates=spec.get("maker_rebates"),
        maker_costs=spec.get("maker_costs"),
        latency_cost=spec.get("latency_cost", 0.0),
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
          maker_rebate=0.0, taker_fee=0.0, tie_rule="split", tick_size=None,
          memory=1, priority_share=0.7, allocation_noise=12.0,
          maker_rebates=None, maker_costs=None, latency_cost=0.0,
          periods=None, **extra):
    out = {
        "sweep": sweep,
        "n_makers": n,
        "adverse_frac": af,
        "adverse_cost": ac,
        "elasticity": el,
        "epsilon_decay": eps,
        "alpha": alpha,
        "maker_rebate": maker_rebate,
        "taker_fee": taker_fee,
        "tie_rule": tie_rule,
        "tick_size": tick_size,
        "memory": memory,
        "priority_share": priority_share,
        "allocation_noise": allocation_noise,
        "maker_rebates": maker_rebates,
        "maker_costs": maker_costs,
        "latency_cost": latency_cost,
        "seed": seed,
        "periods": periods if periods is not None else PERIODS_FOR_N.get(n, 500_000),
    }
    out.update(extra)
    return out


def _base_ext_spec(sweep, seed, periods=None, **kwargs):
    return {
        **_spec(sweep, 2, 0.5, 0.2, 1.0, seed, periods=periods),
        **kwargs,
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


def build_extension_sweep(seeds=range(10), periods: int | None = None) -> list[dict]:
    """Higher-ROI extension experiments for the design claims.

    These are kept separate from ``build_design_sweep`` so older CSVs remain
    reproducible while the revision can add realistic priority, fee-split,
    heterogeneity, and latency variants.
    """

    specs = []
    # Intermediate allocation / priority rules. ``split`` is the proportional
    # pro-rata baseline; the others add stochastic or queue-based priority.
    for rule in ("split", "random_priority", "pro_rata_noise", "queue_priority", "winner_take_all"):
        for s in seeds:
            specs.append(_base_ext_spec("allocation_rule", s, periods, tie_rule=rule))

    # Maker/taker split. ``net_fee`` is taker fee minus maker rebate.
    fee_conditions = [
        ("baseline", 0.0, 0.0),
        ("maker_rebate_only", 0.20, 0.0),
        ("taker_fee_only", 0.0, 0.20),
        ("symmetric_fee_rebate", 0.20, 0.20),
        ("net_fee_10_taker10_rebate00", 0.00, 0.10),
        ("net_fee_10_taker20_rebate10", 0.10, 0.20),
        ("net_fee_10_taker30_rebate20", 0.20, 0.30),
    ]
    for label, rebate, taker_fee in fee_conditions:
        for s in seeds:
            specs.append(
                _base_ext_spec(
                    "fee_split",
                    s,
                    periods,
                    fee_condition=label,
                    maker_rebate=rebate,
                    taker_fee=taker_fee,
                    net_fee=taker_fee - rebate,
                )
            )

    # Heterogeneous makers: asymmetric per-fill costs or maker rebates.
    hetero_conditions = [
        ("symmetric", None, None),
        ("asymmetric_cost_05", None, (0.0, 0.05)),
        ("asymmetric_cost_10", None, (0.0, 0.10)),
        ("asymmetric_rebate_10", (0.10, 0.0), None),
        ("asymmetric_rebate_and_cost", (0.10, 0.0), (0.0, 0.05)),
    ]
    for label, rebates, costs in hetero_conditions:
        for s in seeds:
            specs.append(
                _base_ext_spec(
                    "heterogeneous_makers",
                    s,
                    periods,
                    heterogeneity=label,
                    maker_rebates=rebates,
                    maker_costs=costs,
                )
            )

    # Latency / stale-quote proxy: linear quote-update friction.
    for latency in (0.0, 0.01, 0.03, 0.05, 0.10):
        for s in seeds:
            specs.append(_base_ext_spec("latency", s, periods, latency_cost=latency))
    return specs


def build_queue_position_sweep(seeds=range(20), periods: int | None = None) -> list[dict]:
    """Persistent queue-position stress test.

    ``queue_priority`` is the static proxy used in the first extension sweep.
    ``persistent_queue`` carries priority across periods and moves quote
    changers to the back of the queue.
    """

    specs = []
    for rule in ("split", "queue_priority", "persistent_queue", "winner_take_all"):
        for s in seeds:
            specs.append(_base_ext_spec("queue_position", s, periods, tie_rule=rule))
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
    p.add_argument("--extensions", action="store_true", help="run extension experiments: allocation, fee split, heterogeneity, latency")
    p.add_argument("--queue-position", action="store_true", help="run persistent queue-position stress test")
    p.add_argument("--out", default="collusion/results/sweep_nmakers.csv")
    args = p.parse_args()

    if args.robust:
        specs = build_robustness_sweep(seeds=range(args.seeds))
    elif args.extensions:
        specs = build_extension_sweep(seeds=range(args.seeds), periods=args.periods)
    elif args.queue_position:
        specs = build_queue_position_sweep(seeds=range(args.seeds), periods=args.periods)
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
    elif args.extensions:
        for name, axis in [("allocation_rule", "tie_rule"), ("fee_split", "fee_condition"),
                           ("heterogeneous_makers", "heterogeneity"), ("latency", "latency_cost")]:
            sub = df[df["sweep"] == name]
            if not sub.empty:
                print(f"\n=== {name} ===", flush=True)
                print(sub.groupby(axis)[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
    elif args.queue_position:
        sub = df[df["sweep"] == "queue_position"]
        print(sub.groupby("tie_rule")[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
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
