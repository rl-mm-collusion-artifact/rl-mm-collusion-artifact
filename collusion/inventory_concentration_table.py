"""Build table for inventory x market-concentration experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build(csv: str = "collusion/results/inventory_concentration20.csv") -> str:
    df = pd.read_csv(csv)
    rows = []
    for n_makers, sub in df.groupby("n_makers"):
        d = sub["collusion_index"].to_numpy(dtype=float)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        spread = sub["mean_spread"].mean()
        profit = sub["avg_profit"].mean()
        rows.append(
            f"{int(n_makers)} & {d.mean():.3f} & {se:.3f} & "
            f"{spread:.2f} & {profit:.3f} & {len(d)} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Concentration in the stochastic-mid-price inventory model "
        "($20$ seeds, $8\\times10^5$ periods per cell).}\n"
        "\\label{tab:inventory-concentration}\n\\small\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "makers & $\\Delta$ & SE & spread & profit & seeds \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="collusion/results/inventory_concentration20.csv")
    p.add_argument("--out", default="collusion/results/table_inventory_concentration.tex")
    args = p.parse_args()

    tex = build(args.csv)
    Path(args.out).write_text(tex)
    print(tex)
