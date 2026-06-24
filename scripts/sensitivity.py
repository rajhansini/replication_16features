"""
Week 5: Sensitivity analysis — how RL tracks exact V* across allele frequencies.

Sweeps allele_freq in [0.05, 0.1, 0.15, 0.2, 0.3] on the N=9 family.
For each frequency: trains RL + runs exact backward induction.

Shows the method isn't over-fitted to allele_freq=0.1.

Outputs:
  artifacts/sensitivity_results.json
  artifacts/fig9_sensitivity.png

Usage:
    python -u -m scripts.sensitivity 2>&1 | tee artifacts/sensitivity_run.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv
from genetic_dp.rl.networks import MLPActorCritic
from genetic_dp.rl.ppo import PPOAgent
from genetic_dp.exact_dp.backward_induction import BackwardInductionSolver


def make_n9_pedigree() -> Pedigree:
    ped = Pedigree()
    for x in ["GF1", "GM1", "GF2", "GM2"]:
        ped.add_individual(x)
    ped.add_individual("P1", parents=("GF1", "GM1"))
    ped.add_individual("P2", parents=("GF2", "GM2"))
    for i in range(1, 4):
        ped.add_individual(f"C{i}", parents=("P1", "P2"))
    return ped


def run_one(freq: float, timesteps: int, n_eval: int, seed: int, device: str) -> dict:
    ped = make_n9_pedigree()
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=freq)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=seed)

    # Exact DP
    t0 = time.time()
    dp = BackwardInductionSolver(ped, config, genes=("gene",), verbose=0)
    exact_v = dp.solve()
    exact_time = time.time() - t0

    # RL
    policy = MLPActorCritic(obs_dim=env.obs_dim, n_actions=env.n_actions,
                            hidden=[256, 128])
    agent = PPOAgent(policy=policy, lr=3e-4, clip_ratio=0.2,
                     entropy_coef=0.01, n_epochs=4, batch_size=64, device=device)
    t0 = time.time()
    agent.train(env, total_timesteps=timesteps, rollout_steps=512,
                log_every=max(timesteps // (512 * 5), 1))
    train_time = time.time() - t0

    eval_stats = agent.evaluate_policy(env, n_episodes=n_eval, deterministic=True)
    gap = abs(exact_v - eval_stats["mean_reward"]) / abs(exact_v) * 100

    print(f"  freq={freq:.2f}  exact={exact_v:.6f}  RL={eval_stats['mean_reward']:.6f}"
          f"  gap={gap:.2f}%  exact_t={exact_time:.1f}s  rl_t={train_time:.0f}s")

    return {
        "allele_freq": freq,
        "exact_V0": float(exact_v),
        "exact_time_s": float(exact_time),
        "rl_mean": eval_stats["mean_reward"],
        "rl_std": eval_stats["std_reward"],
        "rl_train_time_s": float(train_time),
        "optimality_gap_pct": float(gap),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freqs", nargs="+", type=float,
                   default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/sensitivity_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  Week 5: Sensitivity Analysis (N=9)")
    print(f"  Allele freqs: {args.freqs}")
    print(f"  Timesteps: {args.timesteps:,}  |  Device: {device}")
    print("=" * 60)

    results = []
    for freq in args.freqs:
        r = run_one(freq, args.timesteps, args.n_eval, args.seed, device)
        results.append(r)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {args.out}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Freq':>6} | {'Exact V*':>10} | {'RL Mean':>10} | {'Gap%':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['allele_freq']:>6.2f} | {r['exact_V0']:>10.6f} | "
              f"{r['rl_mean']:>10.6f} | {r['optimality_gap_pct']:>6.2f}%")
    print("=" * 60)

    # Figure
    freqs = [r["allele_freq"] for r in results]
    exact_vals = [r["exact_V0"] for r in results]
    rl_means = [r["rl_mean"] for r in results]
    rl_stds = [r["rl_std"] for r in results]
    gaps = [r["optimality_gap_pct"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(freqs, exact_vals, "D--", color="#FF5722", ms=8, label="Exact V*")
    ax1.errorbar(freqs, rl_means, yerr=rl_stds, fmt="o-", color="#2196F3",
                 capsize=4, label="RL (PPO)")
    ax1.set_xlabel("Allele frequency"); ax1.set_ylabel("Episode reward")
    ax1.set_title("Reward vs allele frequency (N=9)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xticks(freqs)

    ax2.bar(freqs, gaps, width=0.03, color="#4CAF50", edgecolor="white")
    ax2.axhline(5, ls="--", color="red", lw=1.2, label="5% threshold")
    for f, g in zip(freqs, gaps):
        ax2.text(f, g + 0.1, f"{g:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Allele frequency"); ax2.set_ylabel("Optimality gap (%)")
    ax2.set_title("RL gap from exact V* vs allele frequency")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0); ax2.set_xticks(freqs)

    fig.suptitle("Week 5: Sensitivity to Allele Frequency (N=9)", fontweight="bold")
    fig.tight_layout()
    out = "artifacts/fig9_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


if __name__ == "__main__":
    main()
