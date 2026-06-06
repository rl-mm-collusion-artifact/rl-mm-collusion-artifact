"""Design-lever sweep in the extended (stochastic-mid + inventory) game.

Mirrors the duopoly design sweep in ``collusion.sweep`` but runs the
inventory-aware learner against ``InventoryGame``, to check that (a) collusion
still emerges with a stochastic mid-price and endogenous inventory, and (b) the
maker-rebate, tick-size and tie-rule levers act in the same direction as in the
stylized model. Cells are independent and run across a process pool.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from collusion.env import grid_from_tick
from collusion.env_inventory import InventoryGame
from collusion.qlearning_inventory import InventoryQLearning, InvQLearningConfig

# Inventory state multiplies K^N by n_inv_bins, so give the learner a longer
# horizon than the base duopoly run to converge.
PERIODS = 800_000


def run_cell(spec: dict) -> dict:
    grid_kwargs = {}
    if spec.get("tick_size"):
        grid_kwargs["spread_grid"] = grid_from_tick(spec["tick_size"])
    game = InventoryGame(
        n_makers=spec.get("n_makers", 2),
        maker_rebate=spec.get("maker_rebate", 0.0),
        taker_fee=spec.get("taker_fee", 0.0),
        tie_rule=spec.get("tie_rule", "split"),
        **grid_kwargs,
    )
    cfg = InvQLearningConfig(periods=spec.get("periods", PERIODS), seed=spec["seed"])
    result = InventoryQLearning(game, cfg).run()
    out = {k: spec[k] for k in spec}
    out.update(
        {
            "collusion_index": result["collusion_index"],
            "mean_spread": result["mean_spread"],
            "avg_profit": result["avg_profit"],
            "nash_spread": result["benchmarks"]["nash_spread"],
            "monopoly_spread": result["benchmarks"]["monopoly_spread"],
        }
    )
    return out


def build_inventory_sweep(seeds=range(20)) -> list[dict]:
    specs = []
    # Baseline: does collusion emerge at all in the extended game?
    for s in seeds:
        specs.append({"sweep": "baseline", "maker_rebate": 0.0, "tie_rule": "split",
                      "tick_size": None, "seed": s})
    # Maker rebate lever.
    for rb in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        for s in seeds:
            specs.append({"sweep": "rebate", "maker_rebate": rb, "tie_rule": "split",
                          "tick_size": None, "seed": s})
    # Tick-size lever.
    for tk in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1):
        for s in seeds:
            specs.append({"sweep": "tick_size", "maker_rebate": 0.0, "tie_rule": "split",
                          "tick_size": tk, "seed": s})
    # Tie-breaking lever.
    for tr in ("split", "winner_take_all"):
        for s in seeds:
            specs.append({"sweep": "tie_rule", "maker_rebate": 0.0, "tie_rule": tr,
                          "tick_size": None, "seed": s})
    return specs


def build_fee_incidence_sweep(seeds=range(20), periods: int = PERIODS) -> list[dict]:
    specs = []
    conditions = [
        ("baseline", 0.00, 0.00),
        ("maker_rebate_only", 0.20, 0.00),
        ("taker_fee_only", 0.00, 0.20),
        ("symmetric_fee_rebate", 0.20, 0.20),
        ("same_net_taker_only_10", 0.00, 0.10),
        ("same_net_maker_heavy_10", 0.20, 0.30),
    ]
    for label, rebate, taker_fee in conditions:
        for s in seeds:
            specs.append(
                {
                    "sweep": "fee_incidence",
                    "fee_condition": label,
                    "maker_rebate": rebate,
                    "taker_fee": taker_fee,
                    "net_fee": taker_fee - rebate,
                    "tie_rule": "split",
                    "tick_size": None,
                    "seed": s,
                    "periods": periods,
                }
            )
    return specs


def build_concentration_sweep(seeds=range(20), periods: int = PERIODS) -> list[dict]:
    specs = []
    for n_makers in (2, 3, 4):
        for s in seeds:
            specs.append(
                {
                    "sweep": "inventory_concentration",
                    "n_makers": n_makers,
                    "maker_rebate": 0.0,
                    "taker_fee": 0.0,
                    "tie_rule": "split",
                    "tick_size": None,
                    "seed": s,
                    "periods": periods,
                }
            )
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

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--periods", type=int, default=PERIODS)
    p.add_argument("--fee-incidence", action="store_true", help="run inventory x maker/taker fee-incidence cross")
    p.add_argument("--concentration", action="store_true", help="run inventory x maker-count concentration sweep")
    p.add_argument("--out", default="collusion/results/inventory_sweep.csv")
    args = p.parse_args()

    if args.fee_incidence:
        specs = build_fee_incidence_sweep(seeds=range(args.seeds), periods=args.periods)
    elif args.concentration:
        specs = build_concentration_sweep(seeds=range(args.seeds), periods=args.periods)
    else:
        specs = build_inventory_sweep(seeds=range(args.seeds))
    print(f"running {len(specs)} inventory cells ...", flush=True)
    df = run_sweep(specs, max_workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    for name, axis in [("baseline", "sweep"), ("rebate", "maker_rebate"),
                       ("tick_size", "tick_size"), ("tie_rule", "tie_rule"),
                       ("fee_incidence", "fee_condition"),
                       ("inventory_concentration", "n_makers")]:
        sub = df[df["sweep"] == name]
        if not sub.empty:
            print(f"\n=== {name} ===", flush=True)
            print(sub.groupby(axis)[["collusion_index", "mean_spread"]].mean().round(3).to_string(), flush=True)
    print(f"\nwrote {args.out}", flush=True)
