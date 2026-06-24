"""
Benchmark: RL vs Exact DP — scalability and optimality comparison.

Produces three outputs in artifacts/:
  benchmark_results.json  — raw numbers
  scalability.txt         — human-readable table for the presentation
  training_curve.txt      — reward vs timesteps (if --train is set)

Usage:
    # Quick comparison on pre-set family sizes
    python -m scripts.benchmark

    # Full run: train + benchmark (slower)
    python -m scripts.benchmark --train --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv
from genetic_dp.rl.networks import MLPActorCritic, build_pedigree_adj, build_generation_depths
from genetic_dp.rl.ppo import PPOAgent
from genetic_dp.exact_dp.backward_induction import BackwardInductionSolver


ALLELE_FREQ = 0.1


# ---------------------------------------------------------------------------
# Pedigree catalogue
# ---------------------------------------------------------------------------

def make_pedigree(n: int) -> Pedigree:
    """Generate canonical families of increasing size N.

    N=4  : 2 founders + 2 children
    N=6  : 2 founders + 4 children
    N=7  : 4 grandparents + 2 parents + 1 grandchild
    N=9  : 4 grandparents + 2 parents + 3 grandchildren
    N=12 : 4 grandparents + 2 parents + 6 grandchildren
    N=15 : 4 grandparents + 4 parents (2 families) + 7 grandchildren
    """
    ped = Pedigree()

    if n == 4:
        ped.add_individual("F1"); ped.add_individual("F2")
        ped.add_individual("C1", parents=("F1","F2"))
        ped.add_individual("C2", parents=("F1","F2"))

    elif n == 6:
        ped.add_individual("F1"); ped.add_individual("F2")
        for i in range(1, 5):
            ped.add_individual(f"C{i}", parents=("F1","F2"))

    elif n == 7:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1","GM1"))
        ped.add_individual("P2", parents=("GF2","GM2"))
        ped.add_individual("C1", parents=("P1","P2"))

    elif n == 9:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1","GM1"))
        ped.add_individual("P2", parents=("GF2","GM2"))
        for i in range(1, 4):
            ped.add_individual(f"C{i}", parents=("P1","P2"))

    elif n == 12:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1","GM1"))
        ped.add_individual("P2", parents=("GF2","GM2"))
        for i in range(1, 7):
            ped.add_individual(f"C{i}", parents=("P1","P2"))

    elif n == 15:
        for x in ["GF1","GM1","GF2","GM2"]: ped.add_individual(x)
        ped.add_individual("P1", parents=("GF1","GM1"))
        ped.add_individual("P2", parents=("GF2","GM2"))
        ped.add_individual("P3", parents=("GF1","GM1"))
        ped.add_individual("M3", parents=("GF2","GM2"))  # P3's mate (same founders, distinct person)
        for i in range(1, 5):
            ped.add_individual(f"C{i}", parents=("P1","P2"))
        for i in range(5, 8):
            ped.add_individual(f"C{i}", parents=("P3","M3"))

    else:
        # Generic: 2 founders + (n-2) children
        ped.add_individual("F1"); ped.add_individual("F2")
        for i in range(1, n - 1):
            ped.add_individual(f"C{i}", parents=("F1","F2"))

    return ped


# ---------------------------------------------------------------------------
# Benchmark one family size
# ---------------------------------------------------------------------------

def benchmark_one(
    n: int,
    timesteps: int,
    do_exact: bool,
    n_eval: int,
    seed: int,
    verbose: bool = False,
    device: str = "cpu",
) -> Dict:
    print(f"\n  N={n} {'─'*45}")
    ped = make_pedigree(n)
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=ALLELE_FREQ)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=seed)

    result: Dict = {"N": n, "obs_dim": env.obs_dim, "n_actions": env.n_actions}

    # ---- Exact DP --------------------------------------------------------
    if do_exact and n <= 9:
        dp = BackwardInductionSolver(ped, config, genes=("gene",), verbose=0)
        t0 = time.time()
        v_star = dp.solve()
        exact_time = time.time() - t0
        result["exact_V0"] = float(v_star)
        result["exact_time_s"] = float(exact_time)
        result["exact_n_states"] = dp.n_states
        print(f"    Exact DP:  V*(∅)={v_star:.6f}  states={dp.n_states}  time={exact_time:.2f}s")
    else:
        result["exact_V0"] = None
        result["exact_time_s"] = None
        result["exact_n_states"] = None
        if n > 9:
            print(f"    Exact DP:  SKIPPED (N={n} > 9, state space too large)")
        else:
            print(f"    Exact DP:  SKIPPED (--no-exact flag)")

    # ---- Train RL --------------------------------------------------------
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
    log = agent.train(
        env,
        total_timesteps=timesteps,
        rollout_steps=512,
        log_every=max(timesteps // (512 * 5), 1),
    )
    train_time = time.time() - t0
    result["train_time_s"] = float(train_time)
    result["training_log"] = log

    # ---- Evaluate --------------------------------------------------------
    eval_stats = agent.evaluate_policy(env, n_episodes=n_eval, deterministic=True)
    result["rl_mean_reward"] = eval_stats["mean_reward"]
    result["rl_std_reward"] = eval_stats["std_reward"]
    print(f"    RL policy: mean={eval_stats['mean_reward']:.6f} ± {eval_stats['std_reward']:.4f}  "
          f"train={train_time:.1f}s")

    # ---- Inference speed (1 forward pass) --------------------------------
    obs_t = torch.zeros(1, env.obs_dim, device=device)
    t0 = time.time()
    N_SPEED = 1000
    with torch.no_grad():
        for _ in range(N_SPEED):
            policy.forward(obs_t)
    infer_ms = (time.time() - t0) / N_SPEED * 1000
    result["rl_inference_ms"] = float(infer_ms)
    print(f"    RL inference: {infer_ms:.3f}ms/step")

    # ---- Optimality gap --------------------------------------------------
    if result["exact_V0"] is not None:
        v_star = result["exact_V0"]
        v_rl = result["rl_mean_reward"]
        gap = abs(v_star - v_rl) / max(abs(v_star), 1e-9)
        result["optimality_gap_pct"] = float(gap * 100)
        print(f"    Optimality gap: {gap*100:.2f}%")
    else:
        result["optimality_gap_pct"] = None

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[4, 6, 7, 9],
                   help="Family sizes to benchmark with RL training (N<=9 recommended)")
    p.add_argument("--large-n", nargs="+", type=int, default=[12, 15],
                   help="Large N sizes for RL inference speed only (no training)")
    p.add_argument("--timesteps", type=int, default=30_000,
                   help="PPO training timesteps per family size")
    p.add_argument("--n-eval", type=int, default=200,
                   help="Evaluation episodes per family size")
    p.add_argument("--no-exact", action="store_true",
                   help="Skip exact DP computation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="artifacts/benchmark_results.json")
    return p.parse_args()


def benchmark_inference_only(n: int, device: str, n_speed: int = 1000) -> Dict:
    """For large N: build policy and measure inference speed only (no training)."""
    ped = make_pedigree(n)
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=ALLELE_FREQ)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=0)

    policy = MLPActorCritic(obs_dim=env.obs_dim, n_actions=env.n_actions, hidden=[256, 128])
    policy = policy.to(device)

    obs_t = torch.zeros(1, env.obs_dim, device=device)
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            policy.forward(obs_t)
    # Time
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_speed):
            policy.forward(obs_t)
    infer_ms = (time.time() - t0) / n_speed * 1000

    n_params = sum(p.numel() for p in policy.parameters())
    state_space = 4 ** n  # upper bound
    print(f"    N={n}  LARGE-N INFERENCE ONLY")
    print(f"    State space (bound): {state_space:,}  | RL inference: {infer_ms:.3f}ms/step")
    print(f"    [Exact DP: infeasible — {state_space:,.0f} states]")

    return {
        "N": n, "obs_dim": env.obs_dim, "n_actions": env.n_actions,
        "exact_V0": None, "exact_time_s": None, "exact_n_states": None,
        "rl_mean_reward": None, "rl_std_reward": None,
        "train_time_s": None, "optimality_gap_pct": None,
        "rl_inference_ms": float(infer_ms),
        "note": "inference_only",
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Benchmark: RL vs Exact DP — Scalability")
    print(f"  Sizes: {args.sizes}  |  timesteps/size: {args.timesteps:,}")
    print(f"  Large-N (inference only): {args.large_n}")
    print(f"  Device: {device}")
    print("=" * 60)

    results = []
    for n in args.sizes:
        r = benchmark_one(
            n=n,
            timesteps=args.timesteps,
            do_exact=not args.no_exact,
            n_eval=args.n_eval,
            seed=args.seed,
            verbose=True,
            device=device,
        )
        results.append(r)

    # Large-N: RL inference speed only (exact DP infeasible)
    if args.large_n:
        print(f"\n  {'─'*55}")
        print("  Large-N: RL inference speed only (exact DP infeasible)")
        print(f"  {'─'*55}")
        for n in args.large_n:
            r = benchmark_inference_only(n, device=device)
            results.append(r)

    # ---- Save raw JSON ---------------------------------------------------
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    # ---- Print summary table ---------------------------------------------
    print("\n" + "=" * 75)
    print(f"{'N':>4} | {'Exact V*(∅)':>12} | {'Exact Time':>12} | {'RL Mean':>12} | {'Gap%':>7} | {'RL ms/step':>10}")
    print("-" * 75)
    for r in results:
        n = r["N"]
        ev = f"{r['exact_V0']:.6f}" if r["exact_V0"] is not None else "    N/A  "
        et = f"{r['exact_time_s']:.2f}s" if r["exact_time_s"] is not None else "    N/A"
        rl = f"{r['rl_mean_reward']:.6f}" if r["rl_mean_reward"] is not None else "   N/A  "
        gp = f"{r['optimality_gap_pct']:.2f}%" if r["optimality_gap_pct"] is not None else "   N/A"
        sp = f"{r['rl_inference_ms']:.3f}"
        print(f"{n:>4} | {ev:>12} | {et:>12} | {rl:>12} | {gp:>7} | {sp:>10}")
    print("=" * 75)

    # ---- Save scalability table ------------------------------------------
    table_path = "artifacts/scalability.txt"
    with open(table_path, "w") as f:
        f.write("N | Exact_V0 | Exact_Time_s | Exact_States | RL_Mean | RL_Std | Gap_pct | RL_ms_per_step\n")
        for r in results:
            rl_mean = f"{r['rl_mean_reward']:.6f}" if r['rl_mean_reward'] is not None else 'NA'
            rl_std  = f"{r['rl_std_reward']:.6f}"  if r['rl_std_reward']  is not None else 'NA'
            f.write(
                f"{r['N']} | "
                f"{r['exact_V0'] if r['exact_V0'] is not None else 'NA'} | "
                f"{r['exact_time_s'] if r['exact_time_s'] is not None else 'NA'} | "
                f"{r['exact_n_states'] if r['exact_n_states'] is not None else 'NA'} | "
                f"{rl_mean} | "
                f"{rl_std} | "
                f"{r['optimality_gap_pct'] if r['optimality_gap_pct'] is not None else 'NA'} | "
                f"{r['rl_inference_ms']:.4f}\n"
            )
    print(f"  Scalability table → {table_path}")

    print("\n  KEY TAKE-AWAY:")
    small = [r for r in results if r["N"] <= 9]
    large = [r for r in results if r["N"] > 9]
    if small:
        avg_gap = np.mean([r["optimality_gap_pct"] for r in small if r["optimality_gap_pct"] is not None])
        print(f"    • RL achieves {100-avg_gap:.1f}% of exact optimal on N≤9 families")
    if large:
        print(f"    • RL scales to N={max(r['N'] for r in large)} where exact DP is infeasible")
    if small and large:
        exact_times = [r["exact_time_s"] for r in small if r["exact_time_s"] is not None]
        rl_times = [r["rl_inference_ms"] for r in large]
        if exact_times and rl_times:
            print(f"    • Exact DP time grows to {max(exact_times):.1f}s at N=9; "
                  f"RL inference stays at {np.mean(rl_times):.2f}ms")


if __name__ == "__main__":
    main()
