"""Generate compact LaTeX tables for the extension experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("collusion/results")


def _label(val) -> str:
    if isinstance(val, (int, float, np.floating)):
        return f"{val:.2f}"
    return str(val).replace("_", "-")


def _rows(df: pd.DataFrame, sweep: str, axis: str) -> list[str]:
    sub = df[df["sweep"] == sweep]
    rows = []
    for val, g in sub.groupby(axis):
        d = g["collusion_index"].to_numpy(dtype=float)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        rows.append(f"{_label(val)} & {d.mean():.3f} & {se:.3f} & {g['mean_spread'].mean():.2f} & {len(d)} \\\\")
    return rows


def extension_table(csv: str = "collusion/results/extension_sweep.csv") -> str:
    df = pd.read_csv(csv)
    rows = (
        ["\\multicolumn{5}{@{}l}{\\emph{Allocation / priority rule}}\\\\"]
        + _rows(df, "allocation_rule", "tie_rule")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Maker/taker fee split}}\\\\"]
        + _rows(df, "fee_split", "fee_condition")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Heterogeneous makers}}\\\\"]
        + _rows(df, "heterogeneous_makers", "heterogeneity")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Latency / quote-update cost}}\\\\"]
        + _rows(df, "latency", "latency_cost")
    )
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\caption{Extension experiments at the duopoly: mean $\\Delta$, SE, mean spread, and seeds.}\n"
        "\\label{tab:extensions}\n\\small\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "condition & $\\Delta$ & SE & spread & seeds \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )


def deep_sanity_table(csv: str = "collusion/results/deep_sanity.csv") -> str:
    df = pd.read_csv(csv)
    rows = []
    for (condition, algorithm), g in df.groupby(["condition", "algorithm"]):
        d = g["collusion_index"].to_numpy(dtype=float)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        rows.append(
            f"{_label(condition)} & {_label(algorithm)} & {d.mean():.3f} & {se:.3f} "
            f"& {g['mean_spread'].mean():.2f} & {len(d)} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\caption{Deep-RL sanity check: tabular Q-learning versus small DQN in the duopoly.}\n"
        "\\label{tab:deep-sanity}\n\\small\n"
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
        "condition & learner & $\\Delta$ & SE & spread & seeds \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--extension-csv", default="collusion/results/extension_sweep.csv")
    p.add_argument("--extension-out", default="collusion/results/table_extensions.tex")
    p.add_argument("--deep-csv", default="collusion/results/deep_sanity.csv")
    p.add_argument("--deep-out", default="collusion/results/table_deep_sanity.tex")
    args = p.parse_args()

    if Path(args.extension_csv).exists():
        tbl = extension_table(args.extension_csv)
        Path(args.extension_out).write_text(tbl)
        print(tbl)
    if Path(args.deep_csv).exists():
        tbl = deep_sanity_table(args.deep_csv)
        Path(args.deep_out).write_text(tbl)
        print(tbl)
