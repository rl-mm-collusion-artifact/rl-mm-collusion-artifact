"""Figure generation for the collusion paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collusion.env import MarketMakingGame
from collusion.qlearning import MultiAgentQLearning, QLearningConfig

OUT = Path("collusion/results")


def figure_punishment(seed: int = 0) -> Path:
    game = MarketMakingGame(n_makers=2, elasticity=1.0, adverse_frac=0.5, adverse_cost=0.2)
    learner = MultiAgentQLearning(game, QLearningConfig(periods=600_000, seed=seed))
    res = learner.run()
    ir = learner.impulse_response(deviator=0, pre=5, post=20)
    sp = ir["spreads"]
    b = res["benchmarks"]
    t = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, sp[:, 0], "-o", ms=4, label="maker 0 (forced deviator)", color="#c1121f")
    ax.plot(t, sp[:, 1], "-s", ms=4, label="maker 1 (punisher)", color="#1f4e79")
    ax.axhline(b["monopoly_spread"], ls=":", color="gray", label="monopoly (collusive) spread")
    ax.axhline(b["nash_spread"], ls="--", color="gray", label="competitive (Nash) spread")
    ax.axvline(ir["deviation_period"], color="black", lw=0.8, alpha=0.5)
    ax.annotate("forced\nundercut", (ir["deviation_period"], 0.12), fontsize=8, ha="center")
    ax.set_xlabel("period (greedy play; one forced deviation)")
    ax.set_ylabel("posted half-spread")
    ax.set_title("Deviation triggers punishment, then reversion to collusion")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    path = OUT / "figure_punishment.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def figure_conditions(csv: str = "collusion/results/full_sweep20.csv") -> Path | None:
    if not Path(csv).exists():
        return None
    df = pd.read_csv(csv)
    panels = [("n_makers", "n_makers", "number of market makers"),
              ("adverse", "adverse_cost", "adverse-selection cost"),
              ("elasticity", "elasticity", "demand elasticity"),
              ("exploration", "epsilon_decay", "exploration decay rate")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (sweep, axis, xlabel) in zip(axes, panels):
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            continue
        g = sub.groupby(axis)["collusion_index"].agg(["mean", "std", "count"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]), fmt="-o", color="#1f4e79", capsize=3)
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.1, 1.0)
        if sweep == "exploration":
            ax.set_xscale("log")
    axes[0].set_ylabel("collusion index Δ")
    fig.suptitle("What governs tacit collusion among RL market makers?")
    plt.tight_layout()
    path = OUT / "figure_conditions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def figure_design(csv: str = "collusion/results/design_sweep20.csv") -> Path | None:
    if not Path(csv).exists():
        return None
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # rebate
    sub = df[df["sweep"] == "rebate"]
    g = sub.groupby("maker_rebate")["collusion_index"].agg(["mean", "std", "count"])
    axes[0].errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]), fmt="-o", color="#c1121f", capsize=3)
    axes[0].set_xlabel("maker rebate"); axes[0].set_title("Rebates break collusion")
    # tick size
    sub = df[df["sweep"] == "tick_size"]
    g = sub.groupby("tick_size")["collusion_index"].agg(["mean", "std", "count"])
    axes[1].errorbar(g.index, g["mean"], yerr=g["std"] / np.sqrt(g["count"]), fmt="-o", color="#1f4e79", capsize=3)
    axes[1].set_xlabel("tick size"); axes[1].set_title("Coarse ticks suppress (this model)")
    # tie rule
    sub = df[df["sweep"] == "tie_rule"]
    g = sub.groupby("tie_rule")["collusion_index"].agg(["mean", "std", "count"])
    order = ["split", "winner_take_all"]
    g = g.reindex(order)
    axes[2].bar(range(len(g)), g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                color=["#1f4e79", "#c1121f"], capsize=4, width=0.5)
    axes[2].set_xticks(range(len(g))); axes[2].set_xticklabels(["split", "winner-\ntake-all"])
    axes[2].set_title("Tie rule matters")
    for ax in axes:
        ax.axhline(0, ls="--", color="gray", lw=0.8); ax.set_ylim(-0.05, 1.0)
    axes[0].set_ylabel("collusion index Δ")
    fig.suptitle("Design levers change collusion in the stylized duopoly")
    plt.tight_layout()
    path = OUT / "figure_design.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


if __name__ == "__main__":
    print("punishment:", figure_punishment())
    print("design:", figure_design())
    c = figure_conditions()
    print("conditions:", c if c else "(sweep csv not ready)")
