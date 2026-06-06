"""Build the base-vs-inventory robustness table (results/table_inventory.tex).

Reads the base-model design sweep (``design_sweep20.csv``) and the extended-model
sweep (``inventory_sweep.csv``) and tabulates, for the baseline and for the
strongest setting of each design lever, the collusion index in each model. Shows
that collusion still emerges with a stochastic mid-price and inventory and that
the rebate and tie-rule levers act in the same direction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("collusion/results")


def _mean(df, mask, col="collusion_index"):
    sub = df[mask]
    return float(sub[col].mean()), int(len(sub))


def build() -> str:
    base = pd.read_csv(RES / "design_sweep20.csv")
    inv = pd.read_csv(RES / "inventory_sweep.csv")

    # baseline = no lever (rebate 0, split, fine grid)
    rows = []

    def add(label, bmask, imask):
        bD, _ = _mean(base, bmask)
        iD, n = _mean(inv, imask)
        rows.append((label, bD, iD))

    add("No lever (baseline)",
        (base["sweep"] == "rebate") & np.isclose(base["maker_rebate"], 0.0),
        (inv["sweep"] == "baseline"))
    add("Maker rebate $\\rho=0.40$",
        (base["sweep"] == "rebate") & np.isclose(base["maker_rebate"], 0.40),
        (inv["sweep"] == "rebate") & np.isclose(inv["maker_rebate"], 0.40))
    add("Coarse tick $=0.50$",
        (base["sweep"] == "tick_size") & np.isclose(base["tick_size"], 0.50),
        (inv["sweep"] == "tick_size") & np.isclose(inv["tick_size"], 0.50))
    add("Winner-take-all",
        (base["sweep"] == "tie_rule") & (base["tie_rule"] == "winner_take_all"),
        (inv["sweep"] == "tie_rule") & (inv["tie_rule"] == "winner_take_all"))

    body = "\n".join(
        f"{label} & {bD:.2f} & {iD:.2f} \\\\" for label, bD, iD in rows
    )
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Collusion index $\\Delta$ in the stylized model versus the "
        "extended model with a stochastic mid-price and endogenous inventory "
        "(duopoly, $20$ seeds each). Collusion still emerges, and the rebate and "
        "tie-rule levers move $\\Delta$ in the same direction; the tick-size effect "
        "weakens, consistent with its dependence on the discrete benchmark "
        "structure (\\S\\ref{sec:design}).}\n"
        "\\label{tab:inventory}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Condition & $\\Delta$ (stylized) & $\\Delta$ (mid+inventory) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    return tex


if __name__ == "__main__":
    out = RES / "table_inventory.tex"
    out.write_text(build())
    print(f"wrote {out}")
    print(build())
