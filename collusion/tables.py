"""Generate LaTeX results tables (mean, SE, bootstrap 95% CI, n) from sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _rows(df: pd.DataFrame, sweep: str, axis: str, fmt="{:.2f}") -> list[str]:
    sub = df[df["sweep"] == sweep]
    out = []
    for val, g in sub.groupby(axis):
        d = g["collusion_index"].to_numpy()
        mean, se, n = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)
        lo, hi = _boot_ci(d)
        label = fmt.format(val) if isinstance(val, (int, float, np.floating)) else str(val).replace("_", "-")
        out.append(f"{label} & {mean:.3f} & {se:.3f} & $[{lo:.3f},\\,{hi:.3f}]$ & {n} \\\\")
    return out


def make_tables(full_csv: str, design_csv: str) -> str:
    full = pd.read_csv(full_csv)
    design = pd.read_csv(design_csv)
    blocks = []

    def tbl(caption, label, header, rows):
        body = "\n".join(rows)
        return (
            f"\\begin{{table}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"\\begin{{tabular}}{{@{{}}lcccc@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
        )

    blocks.append(tbl(
        "Collusion vs.\\ number of makers (mean $\\Delta$, standard error, bootstrap 95\\% CI, seeds).",
        "tab:concentration", "$N$ & $\\Delta$ & SE & 95\\% CI & seeds",
        _rows(full, "n_makers", "n_makers", "{:.0f}")))
    blocks.append(tbl(
        "Market-design levers at the duopoly (mean $\\Delta$, SE, bootstrap 95\\% CI, seeds).",
        "tab:design", "lever value & $\\Delta$ & SE & 95\\% CI & seeds",
        ["\\multicolumn{5}{@{}l}{\\emph{Maker rebate} $\\rho$}\\\\"] + _rows(design, "rebate", "maker_rebate")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Tick size} $\\delta$}\\\\"] + _rows(design, "tick_size", "tick_size")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Tie-breaking rule}}\\\\"] + _rows(design, "tie_rule", "tie_rule")))
    blocks.append(tbl(
        "Robustness at the duopoly: collusion is insensitive to these (mean $\\Delta$, SE, 95\\% CI, seeds).",
        "tab:robust", "axis value & $\\Delta$ & SE & 95\\% CI & seeds",
        ["\\multicolumn{5}{@{}l}{\\emph{Adverse-selection cost} $c$}\\\\"] + _rows(full, "adverse", "adverse_cost")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Demand elasticity} $\\eta$}\\\\"] + _rows(full, "elasticity", "elasticity")
        + ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Exploration decay} $\\beta$}\\\\"] + _rows(full, "exploration", "epsilon_decay", "{:.0e}")))
    return "\n".join(blocks)


if __name__ == "__main__":
    out = make_tables("collusion/results/full_sweep20.csv", "collusion/results/design_sweep20.csv")
    Path("collusion/results/tables.tex").write_text(out)
    print(out)
