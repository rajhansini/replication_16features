"""
Week 4 Benchmark: RL Scalability beyond exact DP.

Two axes of scale:
  1. Family size:  N = 4, 6, 7, 9, 12, 15  (single gene)
  2. Gene count:   G = 1, 2, 3              (fixed N=9 family)

For N <= 9 we compare against the known exact V* from Week 3.
For N > 9  exact DP is infeasible; we report RL reward, training time,
and inference speed to show the method still runs.

Outputs (all in artifacts/):
  scale_results.json   — raw numbers
  scale_run.log        — copy of stdout (pipe with tee)

Usage:
    python -u -m scripts.benchmark_scale 2>&1 | tee artifacts/scale_run.log
    python -u -m scripts.benchmark_scale --timesteps 100000 --n-eval 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv
from genetic_dp.rl.networks import MLPActorCritic
from genetic_dp.rl.ppo import PPOAgent

# Known exact V* from Week 3 (BackwardInduction, single gene, allele_freq=0.1)
KNOWN_EXACT = {
    4: -0.07462713157894738,
    6: -0.11021914123180501,
    7: -0.13024273894687500,
    9: -0.16611911152031250,
}

ALLELE_FREQ = 0.1


# ---------------------------------------------------------------------------
# Pedigree builders
# ---------------------------------------------------------------------------

def make_pedigree(n: int) -> Pedigree:
    ped = Pedigree()
    if n == 4:
        ped.add_individual("F1"); ped.add_individual("F2")
        ped.add_individual("C1", parents=("F1", "F2"))
        ped.add_individual("C2", parents=("F1", "F2"))
    elif n == 6:
        ped.add_individual("F1"); ped.add_individual("F2")
        for i in range(1, 5):
            ped.add_individual(f"C{i}", parents=("F1", "F2"))
    elif n == 7:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1", "GM1"))
        ped.add_individual("P2", parents=("GF2", "GM2"))
        ped.add_individual("C1", parents=("P1", "P2"))
    elif n == 9:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1", "GM1"))
        ped.add_individual("P2", parents=("GF2", "GM2"))
        for i in range(1, 4):
            ped.add_individual(f"C{i}", parents=("P1", "P2"))
    elif n == 12:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1", "GM1"))
        ped.add_individual("P2", parents=("GF2", "GM2"))
        for i in range(1, 7):
            ped.add_individual(f"C{i}", parents=("P1", "P2"))
    elif n == 15:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1", "GM1"))
        ped.add_individual("P2", parents=("GF2", "GM2"))
        ped.add_individual("P3", parents=("GF1", "GM1"))
        ped.add_individual("M3", parents=("GF2", "GM2"))
        for i in range(1, 5):
            ped.add_individual(f"C{i}", parents=("P1", "P2"))
        for i in range(5, 8):
            ped.add_individual(f"C{i}", parents=("P3", "M3"))
    else:
        ped.add_individual("F1"); ped.add_individual("F2")
        for i in range(1, n - 1):
            ped.add_individual(f"C{i}", parents=("F1", "F2"))
    return ped


# ---------------------------------------------------------------------------
# Single-run training + evaluation
# ---------------------------------------------------------------------------

def train_and_eval(
    ped: Pedigree,
    genes: tuple,
    timesteps: int,
    n_eval: int,
    seed: int,
    device: str,
    log_every: int = 5,
) -> Dict:
    individuals = ped.to_list()
    N = len(individuals)
    config = get_config(individuals, pedigree=ped, allele_freq=ALLELE_FREQ)
    env = GeneticTestingEnv(ped, config, genes=genes, seed=seed)

    policy = MLPActorCritic(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=[256, 128],
    )
    agent = PPOAgent(
        policy=policy,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        n_epochs=4,
        batch_size=64,
        device=device,
    )

    t0 = time.time()
    agent.train(env, total_timesteps=timesteps, rollout_steps=512, log_every=log_every)
    train_time = time.time() - t0

    eval_stats = agent.evaluate_policy(env, n_episodes=n_eval, deterministic=True)

    # Inference speed
    obs_t = torch.zeros(1, env.obs_dim, device=device)
    with torch.no_grad():
        for _ in range(100): policy.forward(obs_t)   # warm up
    t0 = time.time()
    with torch.no_grad():
        for _ in range(1000): policy.forward(obs_t)
    infer_ms = (time.time() - t0) / 1000 * 1000

    return {
        "N": N,
        "G": len(genes),
        "genes": list(genes),
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "train_time_s": float(train_time),
        "rl_mean_reward": eval_stats["mean_reward"],
        "rl_std_reward": eval_stats["std_reward"],
        "rl_inference_ms": float(infer_ms),
        "cache_size": len(env._infer._posterior_cache),
        "state_space_bound": 4 ** N,
        "config_space": 3 ** N,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[4, 6, 7, 9, 12, 15],
                   help="Family sizes for single-gene scale-up")
    p.add_argument("--gene-counts", nargs="+", type=int,
                   default=[1, 2, 3],
                   help="Gene counts for multi-gene experiment (fixed N=9)")
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="PPO timesteps for N <= large-n-threshold")
    p.add_argument("--large-n-timesteps", type=int, default=8_000,
                   help="PPO timesteps for N > large-n-threshold (inference is slow per step)")
    p.add_argument("--large-n-threshold", type=int, default=12,
                   help="N above which --large-n-timesteps kicks in")
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="artifacts/scale_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  Week 4: RL Scalability Benchmark")
    print(f"  Sizes: {args.sizes}  |  Gene counts (N=9): {args.gene_counts}")
    print(f"  Timesteps: {args.timesteps:,}  |  Device: {device}")
    print("=" * 65)

    results = {"family_size": [], "multi_gene": []}

    # ------------------------------------------------------------------
    # Part 1: Scale by family size (single gene)
    # ------------------------------------------------------------------
    print("\n── Part 1: Family size scale-up (single gene) ──────────────")
    for n in args.sizes:
        ts = args.large_n_timesteps if n > args.large_n_threshold else args.timesteps
        print(f"\n  N={n}  (state space bound: 4^{n} = {4**n:,})  timesteps={ts:,}")
        ped = make_pedigree(n)
        r = train_and_eval(ped, ("gene",), ts, args.n_eval, args.seed, device)

        exact_v = KNOWN_EXACT.get(n)
        if exact_v is not None:
            gap = abs(exact_v - r["rl_mean_reward"]) / abs(exact_v) * 100
            r["exact_V0"] = exact_v
            r["optimality_gap_pct"] = gap
            print(f"    RL mean={r['rl_mean_reward']:.6f}  exact={exact_v:.6f}  "
                  f"gap={gap:.2f}%  train={r['train_time_s']:.1f}s  "
                  f"cache={r['cache_size']} states")
        else:
            r["exact_V0"] = None
            r["optimality_gap_pct"] = None
            print(f"    RL mean={r['rl_mean_reward']:.6f}  [exact DP infeasible]  "
                  f"train={r['train_time_s']:.1f}s  cache={r['cache_size']} states")

        results["family_size"].append(r)

    # ------------------------------------------------------------------
    # Part 2: Scale by gene count (fixed N=9)
    # ------------------------------------------------------------------
    print("\n── Part 2: Multi-gene scale-up (N=9 family) ────────────────")
    ped9 = make_pedigree(9)
    for g in args.gene_counts:
        genes = tuple(f"gene{i+1}" for i in range(g))
        print(f"\n  G={g} genes  obs_dim={9*g + 9}  "
              f"state space: 4^9 = {4**9:,} (per gene independent)")
        r = train_and_eval(ped9, genes, args.timesteps, args.n_eval, args.seed, device)
        exact_v = KNOWN_EXACT.get(9) if g == 1 else None
        if exact_v is not None:
            gap = abs(exact_v - r["rl_mean_reward"]) / abs(exact_v) * 100
            r["exact_V0"] = exact_v
            r["optimality_gap_pct"] = gap
        else:
            r["exact_V0"] = None
            r["optimality_gap_pct"] = None
        print(f"    RL mean={r['rl_mean_reward']:.6f}  train={r['train_time_s']:.1f}s  "
              f"cache={r['cache_size']} states")
        results["multi_gene"].append(r)

    # ------------------------------------------------------------------
    # Save + print summary
    # ------------------------------------------------------------------
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    print("\n" + "=" * 70)
    print("  FAMILY SIZE SCALE-UP")
    print(f"{'N':>4} | {'RL Mean':>10} | {'Gap%':>7} | {'Train(s)':>9} | "
          f"{'Infer(ms)':>10} | {'Cache':>8}")
    print("-" * 70)
    for r in results["family_size"]:
        gp = f"{r['optimality_gap_pct']:.2f}%" if r["optimality_gap_pct"] is not None else "  N/A"
        print(f"{r['N']:>4} | {r['rl_mean_reward']:>10.6f} | {gp:>7} | "
              f"{r['train_time_s']:>9.1f} | {r['rl_inference_ms']:>10.3f} | "
              f"{r['cache_size']:>8,}")

    print("\n  MULTI-GENE (N=9)")
    print(f"{'G':>3} | {'RL Mean':>10} | {'Gap%':>7} | {'Train(s)':>9} | {'Infer(ms)':>10}")
    print("-" * 50)
    for r in results["multi_gene"]:
        gp = f"{r['optimality_gap_pct']:.2f}%" if r["optimality_gap_pct"] is not None else "  N/A"
        print(f"{r['G']:>3} | {r['rl_mean_reward']:>10.6f} | {gp:>7} | "
              f"{r['train_time_s']:>9.1f} | {r['rl_inference_ms']:>10.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
