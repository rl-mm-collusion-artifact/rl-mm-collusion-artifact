"""Run the two-venue routing fee-incidence experiment."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.env_two_venue import TwoVenueGame
from collusion.qlearning_two_venue import TwoVenueQLearning, TwoVenueQLearningConfig

PERIODS = 500_000

CONDITIONS = [
    ("neutral_vs_neutral", "two neutral venues", (0.00, 0.00), (0.00, 0.00)),
    ("maker_rebate_vs_neutral", "maker-rebate venue vs neutral", (0.20, 0.00), (0.00, 0.00)),
    ("taker_fee_vs_neutral", "taker-fee venue vs neutral", (0.00, 0.00), (0.20, 0.00)),
    ("maker_taker_vs_neutral", "maker-taker venue vs neutral", (0.20, 0.00), (0.20, 0.00)),
    ("same_net_maker_heavy_vs_neutral", "same-net maker-heavy vs neutral", (0.20, 0.00), (0.30, 0.00)),
    ("same_net_taker_heavy_vs_neutral", "same-net taker-heavy vs neutral", (0.05, 0.00), (0.15, 0.00)),
    ("two_identical_maker_taker", "two identical maker-taker venues", (0.20, 0.20), (0.20, 0.20)),
]


def run_cell(spec: dict) -> dict:
    game = TwoVenueGame(
        n_makers=spec.get("n_makers", 2),
        maker_rebates=spec["maker_rebates"],
        taker_fees=spec["taker_fees"],
        tie_rule=spec.get("tie_rule", "split"),
    )
    cfg = TwoVenueQLearningConfig(periods=spec.get("periods", PERIODS), seed=spec["seed"])
    result = TwoVenueQLearning(game, cfg).run()
    row = dict(spec)
    row.update(
        {
            "maker_rebate_a": spec["maker_rebates"][0],
            "maker_rebate_b": spec["maker_rebates"][1],
            "taker_fee_a": spec["taker_fees"][0],
            "taker_fee_b": spec["taker_fees"][1],
            "avg_profit": result["avg_profit"],
            "collusion_index": result["collusion_index"],
            "realized_spread": result["realized_spread"],
            "effective_spread": result["effective_spread"],
            "venue_a_flow_share": result["venue_a_flow_share"],
            "nash_spread": result["benchmarks"]["nash_spread"],
            "nash_effective_spread": result["benchmarks"]["nash_effective_spread"],
            "nash_venue": result["benchmarks"]["nash_venue"],
            "monopoly_spread": result["benchmarks"]["monopoly_spread"],
            "monopoly_effective_spread": result["benchmarks"]["monopoly_effective_spread"],
            "monopoly_venue": result["benchmarks"]["monopoly_venue"],
        }
    )
    return row


def build_specs(seeds=range(20), periods: int = PERIODS) -> list[dict]:
    specs = []
    for condition, label, rebates, taker_fees in CONDITIONS:
        for seed in seeds:
            specs.append(
                {
                    "suite": "two_venue_routing",
                    "condition": condition,
                    "condition_label": label,
                    "maker_rebates": rebates,
                    "taker_fees": taker_fees,
                    "n_makers": 2,
                    "tie_rule": "split",
                    "seed": seed,
                    "periods": periods,
                }
            )
    return specs


def run_sweep(specs: list[dict], *, max_workers: int | None = None) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_cell, spec) for spec in specs]
        for f in as_completed(futures):
            rows.append(f.result())
    order = {condition: i for i, (condition, _, _, _) in enumerate(CONDITIONS)}
    return pd.DataFrame(rows).sort_values(["condition", "seed"], key=lambda s: s.map(order) if s.name == "condition" else s)


def _boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = {condition: i for i, (condition, _, _, _) in enumerate(CONDITIONS)}
    for condition, group in seed_df.groupby("condition", sort=False):
        d = group["collusion_index"].to_numpy(dtype=float)
        lo, hi = _boot_ci(d)
        rows.append(
            {
                "condition": condition,
                "condition_label": group["condition_label"].iloc[0],
                "mean_delta": float(d.mean()),
                "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
                "realized_spread": float(group["realized_spread"].mean()),
                "effective_spread": float(group["effective_spread"].mean()),
                "venue_a_flow_share": float(group["venue_a_flow_share"].mean()),
                "avg_profit": float(group["avg_profit"].mean()),
                "seeds": int(len(d)),
            }
        )
    return pd.DataFrame(rows).sort_values("condition", key=lambda s: s.map(order))


def latex_table(agg: pd.DataFrame) -> str:
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['condition_label']} & {r['mean_delta']:.3f} & {r['se']:.3f} "
            f"& $[{r['ci_low']:.3f},\\,{r['ci_high']:.3f}]$ "
            f"& {r['realized_spread']:.2f} & {r['effective_spread']:.2f} "
            f"& {r['venue_a_flow_share']:.2f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Two-venue routing experiment (duopoly, $20$ seeds). "
        "Venue A carries the non-neutral fee schedule where applicable; venue B is neutral unless noted.}\n"
        "\\label{tab:two-venue-routing}\n\\small\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Condition & $\\Delta$ & SE & 95\\% CI & spread & eff. spread & A flow \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--periods", type=int, default=PERIODS)
    p.add_argument("--seed-out", default="collusion/results/two_venue_routing20_seeds.csv")
    p.add_argument("--out", default="collusion/results/two_venue_routing20_aggregate.csv")
    p.add_argument("--table-out", default="collusion/results/table_two_venue_routing.tex")
    args = p.parse_args()

    specs = build_specs(seeds=range(args.seeds), periods=args.periods)
    print(f"running {len(specs)} two-venue cells ...", flush=True)
    seed_df = run_sweep(specs, max_workers=args.workers)
    agg = aggregate(seed_df)
    Path(args.seed_out).parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(args.seed_out, index=False)
    agg.to_csv(args.out, index=False)
    table = latex_table(agg)
    Path(args.table_out).write_text(table)
    print(agg[["condition_label", "mean_delta", "se", "realized_spread", "effective_spread", "venue_a_flow_share"]].round(3).to_string(index=False), flush=True)
    print(f"\nwrote {args.seed_out}", flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.table_out}", flush=True)
