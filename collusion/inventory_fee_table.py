"""Build table for inventory x maker/taker fee-incidence experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("collusion/results")

ORDER = [
    ("baseline", "Baseline"),
    ("maker_rebate_only", "Maker rebate only"),
    ("taker_fee_only", "Taker fee only"),
    ("symmetric_fee_rebate", "Symmetric fee/rebate"),
    ("same_net_taker_only_10", "Same net fee: taker-only"),
    ("same_net_maker_heavy_10", "Same net fee: maker-heavy"),
]


def build(csv: str = "collusion/results/inventory_fee_incidence20.csv") -> str:
    df = pd.read_csv(csv)
    rows = []
    for key, label in ORDER:
        sub = df[df["fee_condition"] == key]
        if sub.empty:
            continue
        d = sub["collusion_index"].to_numpy(dtype=float)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        spread = sub["mean_spread"].mean()
        profit = sub["avg_profit"].mean()
        rows.append(f"{label} & {d.mean():.3f} & {se:.3f} & {spread:.2f} & {profit:.3f} & {len(d)} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Fee incidence in the stochastic-mid-price inventory model "
        "(duopoly, $20$ seeds, $8\\times10^5$ periods per cell). Pure maker-rebate "
        "and taker-fee schedules are close to baseline, but symmetric and same-net "
        "maker-heavy schedules reduce learned collusion relative to taker-side "
        "fee incidence.}\n"
        "\\label{tab:inventory-fee}\n\\small\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "Condition & $\\Delta$ & SE & spread & profit & seeds \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="collusion/results/inventory_fee_incidence20.csv")
    p.add_argument("--out", default="collusion/results/table_inventory_fee.tex")
    args = p.parse_args()

    tex = build(args.csv)
    Path(args.out).write_text(tex)
    print(tex)
