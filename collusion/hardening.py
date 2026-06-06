"""Hardening artifacts: raw-outcome table, bootstrap comparison tests, and
forced-deviation metrics, for the submission revision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from collusion.env import MarketMakingGame, grid_from_tick
from collusion.qlearning import MultiAgentQLearning, QLearningConfig

OUT = Path("collusion/results")
BASE = dict(n_makers=2, elasticity=1.0, adverse_frac=0.5, adverse_cost=0.2)


def _bench(game: MarketMakingGame) -> tuple[float, float]:
    b = game.benchmarks()
    return b["nash_spread"], b["monopoly_spread"]


def _boot_diff_p(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> float:
    """Two-sided bootstrap p for difference of means (condition minus baseline)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    obs = a.mean() - b.mean()
    boot = np.array([rng.choice(a, len(a), replace=True).mean()
                     - rng.choice(b, len(b), replace=True).mean() for _ in range(n)])
    # fraction of bootstrap diffs on the opposite side of zero from the observed
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    return float(min(1.0, p))


def raw_and_tests(design_csv: str) -> str:
    df = pd.read_csv(design_csv)
    base = df[(df.sweep == "rebate") & (df.maker_rebate == 0.0)]["collusion_index"].to_numpy()

    def game_for(row_sweep, val):
        if row_sweep == "rebate":
            return MarketMakingGame(**BASE, maker_rebate=val)
        if row_sweep == "tick_size":
            return MarketMakingGame(**BASE, spread_grid=grid_from_tick(val))
        return MarketMakingGame(**BASE)  # tie rule does not change benchmarks

    rows = []
    spec = [("rebate", "maker_rebate", "$\\rho={:.2f}$"),
            ("tick_size", "tick_size", "$\\delta={:.2f}$"),
            ("tie_rule", "tie_rule", "{}")]
    for sweep, axis, fmt in spec:
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{sweep.replace('_',' ')}}}}}\\\\")
        for val, g in df[df.sweep == sweep].groupby(axis):
            d = g["collusion_index"].to_numpy()
            game = game_for(sweep, val)
            aN, aM = _bench(game)
            rs = g["mean_spread"].mean()
            rp = g["avg_profit"].mean()
            delta = d.mean()
            p = _boot_diff_p(d, base)
            sig = "$^{*}$" if p < 0.05 else ""
            label = fmt.format(val) if not isinstance(val, str) else str(val).replace("_", "-")
            rows.append(f"{label} & {aN:.2f} & {aM:.2f} & {rs:.2f} & {rp:.3f} & {delta:.3f}{sig} & {p:.3f} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\caption{Raw outcomes and significance for the design "
        "levers (duopoly, 20 seeds). $a^N$, $a^M$: competitive and monopoly spreads "
        "(recomputed per condition); realized spread and profit are means over seeds; "
        "$\\Delta$ is the collusion index; $p$ is a two-sided bootstrap test of $\\Delta$ "
        "against the no-rebate baseline. $^{*}$ marks $p<0.05$.}\n\\label{tab:raw}\n\\small\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "condition & $a^N$ & $a^M$ & spread & profit & $\\Delta$ & $p$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def deviation_metrics(seed: int = 0) -> dict:
    game = MarketMakingGame(**BASE)
    m = MultiAgentQLearning(game, QLearningConfig(periods=600_000, seed=seed))
    m.run()
    ir = m.impulse_response(deviator=0, pre=5, post=25)
    sp = ir["spreads"]
    dev = ir["deviation_period"]
    punisher = sp[:, 1]  # maker 1
    pre_spread = float(punisher[:dev].mean())
    forced = float(sp[dev, 0])  # deviator's forced spread
    post = punisher[dev + 1:]
    min_retaliation = float(post.min())
    # time to reversion: first period after deviation where punisher returns to pre level
    rev = next((i + 1 for i, v in enumerate(post) if v >= pre_spread - 1e-9), len(post))
    post_reversion = float(punisher[-1])
    return {
        "pre_spread": pre_spread, "forced_deviation": forced,
        "min_retaliation": min_retaliation, "time_to_reversion": rev,
        "post_reversion": post_reversion,
    }


if __name__ == "__main__":
    tbl = raw_and_tests("collusion/results/design_sweep20.csv")
    (OUT / "table_raw.tex").write_text(tbl)
    print(tbl)
    print("\n=== forced-deviation metrics ===")
    for k, v in deviation_metrics().items():
        print(f"  {k}: {v}")
