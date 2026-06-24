"""
Generate benchmark figures for Week 3 presentation.

Reads artifacts/benchmark_results.json and produces three figures:
  artifacts/fig1_compute_time.png   — Exact DP solve time vs N (log scale)
  artifacts/fig2_optimality_gap.png — RL optimality gap % by family size
  artifacts/fig3_scalability.png    — Combined: RL inference vs Exact DP time

Usage:
    python -m scripts.plot_benchmark
    python -m scripts.plot_benchmark --no-show   # save only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def state_space_bound(n: int) -> int:
    """Upper-bound on exact DP state space: 3^N configs × (N+1) test decisions."""
    return 3 ** n


# ---------------------------------------------------------------------------
# Figure 1 — Exact DP: compute time and state count vs N
# ---------------------------------------------------------------------------

def fig_compute_time(results, out_path: str) -> None:
    small = [r for r in results if r["exact_time_s"] is not None]
    large = [r for r in results if r["exact_time_s"] is None and r["N"] > 9]

    ns_small = [r["N"] for r in small]
    times = [r["exact_time_s"] for r in small]
    states = [r["exact_n_states"] for r in small]
    ns_large = [r["N"] for r in large]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ax1.semilogy(ns_small, times, "o-", color="#2166ac", linewidth=2,
                 markersize=8, label="Exact DP solve time (s)")
    ax2.semilogy(ns_small, states, "s--", color="#d6604d", linewidth=1.5,
                 markersize=7, label="State space size")

    for n in ns_large:
        ax1.axvline(n, color="gray", linestyle=":", alpha=0.6)
        ax1.text(n + 0.05, max(times) * 1.5, f"N={n}\n(infeasible)",
                 fontsize=8, color="gray", va="top")

    ax1.set_xlabel("Family size (N)", fontsize=12)
    ax1.set_ylabel("Exact DP solve time (s)", color="#2166ac", fontsize=11)
    ax2.set_ylabel("State space  |S|  (log scale)", color="#d6604d", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#d6604d")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("Exact Backward Induction: Compute Cost vs Family Size", fontsize=12)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Optimality gap and reward comparison
# ---------------------------------------------------------------------------

def fig_optimality_gap(results, out_path: str) -> None:
    trained = [r for r in results
               if r["rl_mean_reward"] is not None and r["exact_V0"] is not None]

    ns = [r["N"] for r in trained]
    exact_v = [abs(r["exact_V0"]) for r in trained]
    rl_v = [abs(r["rl_mean_reward"]) for r in trained]
    gaps = [r["optimality_gap_pct"] for r in trained]

    x = np.arange(len(ns))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: reward bars
    b1 = ax1.bar(x - width / 2, exact_v, width, label="Exact DP |V*(∅)|",
                 color="#2166ac", alpha=0.85)
    b2 = ax1.bar(x + width / 2, rl_v, width, label="RL mean |reward|",
                 color="#fc8d59", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Expected total cost (absolute value)")
    ax1.set_title("RL vs Exact DP: Policy Value")
    ax1.legend(fontsize=9)
    ax1.bar_label(b1, fmt="%.4f", fontsize=7, padding=2)
    ax1.bar_label(b2, fmt="%.4f", fontsize=7, padding=2)

    # Right: optimality gap
    colors = ["#91bfdb" if g < 5 else "#fc8d59" if g < 15 else "#d73027" for g in gaps]
    bars = ax2.bar(x, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_ylabel("Optimality gap (%)")
    ax2.set_title("RL Optimality Gap vs Exact DP")
    ax2.axhline(5, color="green", linestyle="--", linewidth=1, label="5% threshold")
    ax2.axhline(15, color="orange", linestyle="--", linewidth=1, label="15% threshold")
    ax2.legend(fontsize=9)
    ax2.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Scalability: RL inference vs Exact DP time
# ---------------------------------------------------------------------------

def fig_scalability(results, out_path: str) -> None:
    all_n = sorted(set(r["N"] for r in results))
    exact_times = {r["N"]: r["exact_time_s"] for r in results}
    rl_infer_ms = {r["N"]: r["rl_inference_ms"] for r in results if r.get("rl_inference_ms")}
    state_bounds = {n: state_space_bound(n) for n in all_n}

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ns_exact = [n for n in all_n if exact_times.get(n) is not None]
    ns_rl = sorted(rl_infer_ms.keys())

    # Exact DP time (left y)
    ax1.semilogy(
        ns_exact, [exact_times[n] for n in ns_exact],
        "o-", color="#2166ac", linewidth=2, markersize=9, label="Exact DP solve time (s)"
    )

    # RL inference time (left y, in seconds for same axis)
    rl_infer_s = [rl_infer_ms[n] / 1000 for n in ns_rl]
    ax1.semilogy(
        ns_rl, rl_infer_s,
        "D--", color="#fc8d59", linewidth=2, markersize=8, label="RL inference time (s/step)"
    )

    # State space bound (right y)
    ax2.semilogy(
        all_n, [state_bounds[n] for n in all_n],
        ":", color="#4dac26", linewidth=1.5, alpha=0.7, label="State space 3^N (bound)"
    )

    # Shade infeasible region
    infeasible_n = [n for n in all_n if exact_times.get(n) is None]
    if infeasible_n:
        ax1.axvspan(min(infeasible_n) - 0.5, max(infeasible_n) + 0.5,
                    alpha=0.08, color="red", label="Exact DP infeasible")

    ax1.set_xlabel("Family size (N)", fontsize=12)
    ax1.set_ylabel("Time (seconds, log scale)", fontsize=11)
    ax2.set_ylabel("State space bound 3^N", color="#4dac26", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#4dac26")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Scalability: RL vs Exact DP as Family Size Grows", fontsize=12)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Training curves (reward vs timesteps for each N)
# ---------------------------------------------------------------------------

def fig_training_curves(results, out_path: str) -> None:
    trained = [r for r in results if r.get("training_log") and r["rl_mean_reward"] is not None]
    if not trained:
        print("  No training log data — skipping training curve figure")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(trained)))

    for r, color in zip(trained, colors):
        log = r["training_log"]
        ts = log.get("timesteps", [])
        rewards = log.get("mean_ep_reward", [])
        if ts and rewards:
            ax.plot(ts, rewards, "o-", color=color, linewidth=1.5,
                    markersize=4, label=f"N={r['N']}", alpha=0.85)
        # Draw exact V* as horizontal dashed line
        if r.get("exact_V0") is not None:
            ax.axhline(r["exact_V0"], color=color, linestyle="--",
                       linewidth=1, alpha=0.5)

    ax.set_xlabel("Training timesteps", fontsize=11)
    ax.set_ylabel("Mean episode reward", fontsize=11)
    ax.set_title("PPO Training Curves (dashed = exact optimal)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    print("\n" + "=" * 80)
    print(f"{'N':>4} | {'Exact V*':>12} | {'RL Mean':>12} | {'Gap %':>8} | "
          f"{'Exact Time':>12} | {'RL ms/step':>12}")
    print("-" * 80)
    for r in results:
        ev = f"{r['exact_V0']:.6f}" if r['exact_V0'] is not None else "     N/A"
        et = f"{r['exact_time_s']:.2f}s" if r['exact_time_s'] is not None else "  infeasible"
        rl = f"{r['rl_mean_reward']:.6f}" if r['rl_mean_reward'] is not None else "     N/A"
        gp = f"{r['optimality_gap_pct']:.2f}%" if r['optimality_gap_pct'] is not None else "     N/A"
        ms = f"{r['rl_inference_ms']:.3f}" if r.get('rl_inference_ms') else "     N/A"
        print(f"{r['N']:>4} | {ev:>12} | {rl:>12} | {gp:>8} | {et:>12} | {ms:>12}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="artifacts/benchmark_results.json")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if not HAS_MPL:
        print("matplotlib not available — install it to generate figures")
        sys.exit(1)

    results = load_results(args.results)
    os.makedirs(args.out_dir, exist_ok=True)

    print_summary(results)

    print("\nGenerating figures...")
    fig_compute_time(results, f"{args.out_dir}/fig1_compute_time.png")
    fig_optimality_gap(results, f"{args.out_dir}/fig2_optimality_gap.png")
    fig_scalability(results, f"{args.out_dir}/fig3_scalability.png")
    fig_training_curves(results, f"{args.out_dir}/fig4_training_curves.png")

    print("\nDone. Figures saved to", args.out_dir)


if __name__ == "__main__":
    main()
