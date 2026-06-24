"""
Week 4: Plot scalability results from artifacts/scale_results.json.

Generates 3 figures in artifacts/:
  fig5_family_scale.png   — RL reward + training time vs N
  fig6_multigene.png      — RL reward vs number of genes (N=9)
  fig7_rl_vs_baselines.png — RL vs myopic + random baselines per N

Usage:
    python -m scripts.plot_scale [--inp artifacts/scale_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGS = "artifacts"
KNOWN_EXACT = {
    4: -0.07462713157894738,
    6: -0.11021914123180501,
    7: -0.13024273894687500,
    9: -0.16611911152031250,
}


def load(path: str):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 5 — Family size scale-up: reward + training time
# ---------------------------------------------------------------------------
def fig_family_scale(data):
    rows = data["family_size"]
    Ns = [r["N"] for r in rows]
    means = [r["rl_mean_reward"] for r in rows]
    stds  = [r["rl_std_reward"] for r in rows]
    times = [r["train_time_s"] for r in rows]

    exact_Ns = [n for n in Ns if n in KNOWN_EXACT]
    exact_Vs = [KNOWN_EXACT[n] for n in exact_Ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Reward vs N
    ax1.errorbar(Ns, means, yerr=stds, fmt="o-", color="#2196F3",
                 capsize=4, label="RL policy")
    ax1.plot(exact_Ns, exact_Vs, "D--", color="#FF5722", ms=8, label="Exact V*")
    ax1.axvline(9.5, ls=":", color="gray", lw=1.2)
    ax1.text(10.0, min(means) * 0.97, "Exact DP\ninfeasible →",
             fontsize=8, color="gray")
    ax1.set_xlabel("Family size N")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("RL reward vs. family size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(Ns)

    # Right: Training time
    ax2.bar(Ns, times, color="#4CAF50", width=0.6, edgecolor="white")
    for i, (n, t) in enumerate(zip(Ns, times)):
        ax2.text(n, t + max(times) * 0.01, f"{t:.0f}s",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Family size N")
    ax2.set_ylabel("Training time (s)")
    ax2.set_title("PPO training time vs. family size")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(Ns)

    fig.suptitle("Week 4: RL Scalability — Family Size", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig5_family_scale.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 6 — Multi-gene scale-up (N=9)
# ---------------------------------------------------------------------------
def fig_multigene(data):
    rows = data["multi_gene"]
    Gs = [r["G"] for r in rows]
    means = [r["rl_mean_reward"] for r in rows]
    stds  = [r["rl_std_reward"] for r in rows]
    times = [r["train_time_s"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: reward vs gene count
    ax1.errorbar(Gs, means, yerr=stds, fmt="s-", color="#9C27B0",
                 capsize=4, ms=8)
    ax1.set_xlabel("Number of genes G")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("RL reward vs. gene count (N=9)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(Gs)

    # Right: training time
    ax2.bar(Gs, times, color="#FF9800", width=0.4, edgecolor="white")
    for g, t in zip(Gs, times):
        ax2.text(g, t + max(times) * 0.01, f"{t:.0f}s",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Number of genes G")
    ax2.set_ylabel("Training time (s)")
    ax2.set_title("Training time vs. gene count (N=9)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(Gs)

    fig.suptitle("Week 4: RL Scalability — Multi-Gene (N=9)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig6_multigene.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 7 — RL vs baselines: optimality gap (where exact known)
# ---------------------------------------------------------------------------
def fig_optimality(data):
    rows = [r for r in data["family_size"] if r.get("optimality_gap_pct") is not None]
    if not rows:
        print("  No gap data — skipping fig7")
        return

    Ns = [r["N"] for r in rows]
    gaps = [r["optimality_gap_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(Ns, gaps, color="#2196F3", width=0.5, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(5, ls="--", color="#FF5722", lw=1.2, label="5% threshold")

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{gap:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Family size N")
    ax.set_ylabel("Optimality gap (%)")
    ax.set_title("RL vs. Exact DP: Optimality Gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(Ns)
    ax.set_ylim(bottom=0)

    fig.suptitle("Week 4: RL Quality vs. Exact DP", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig7_optimality_gap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="artifacts/scale_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGS, exist_ok=True)

    print(f"  Loading {args.inp}")
    data = load(args.inp)

    print("  Generating figures...")
    fig_family_scale(data)
    fig_multigene(data)
    fig_optimality(data)
    print("  Done — 3 figures saved to artifacts/")


if __name__ == "__main__":
    main()
