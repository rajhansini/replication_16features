"""
Train an RL agent on the sequential genetic testing MDP.

Usage (from repo root):
    python -m scripts.train_rl [--model mlp|gnn] [--family nuclear|extended|large]
                                [--timesteps 100000] [--seed 0]

Trains on a single family pedigree (for demonstration). In practice, train on
a distribution of randomly generated families for generalisation.
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

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv
from genetic_dp.rl.networks import (
    MLPActorCritic,
    GNNActorCritic,
    build_pedigree_adj,
    build_generation_depths,
)
from genetic_dp.rl.ppo import PPOAgent


# ---------------------------------------------------------------------------
# Pedigree factories
# ---------------------------------------------------------------------------

def make_nuclear() -> Pedigree:
    """4 individuals: 2 founders, 2 children."""
    ped = Pedigree()
    ped.add_individual("F1"); ped.add_individual("F2")
    ped.add_individual("C1", parents=("F1", "F2"))
    ped.add_individual("C2", parents=("F1", "F2"))
    return ped


def make_extended() -> Pedigree:
    """7 individuals: 2 grandparents + 2 parents (with mates) + 3 grandchildren."""
    ped = Pedigree()
    for ind in ["GF1", "GM1", "GF2", "GM2"]:
        ped.add_individual(ind)
    ped.add_individual("P1", parents=("GF1", "GM1"))
    ped.add_individual("P2", parents=("GF2", "GM2"))
    ped.add_individual("C1", parents=("P1", "P2"))
    return ped


def make_large() -> Pedigree:
    """9 individuals: grandparents → parents → 3 children."""
    ped = Pedigree()
    for ind in ["GF1", "GM1", "GF2", "GM2"]:
        ped.add_individual(ind)
    ped.add_individual("P1", parents=("GF1", "GM1"))
    ped.add_individual("P2", parents=("GF2", "GM2"))
    ped.add_individual("C1", parents=("P1", "P2"))
    ped.add_individual("C2", parents=("P1", "P2"))
    ped.add_individual("C3", parents=("P1", "P2"))
    return ped


PEDIGREES = {
    "nuclear": make_nuclear,
    "extended": make_extended,
    "large": make_large,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "gnn"], default="mlp")
    p.add_argument("--family", choices=list(PEDIGREES), default="nuclear")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allele-freq", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-path", type=str, default="artifacts/trained_policy.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    ped = PEDIGREES[args.family]()
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=args.allele_freq)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  Training PPO ({args.model.upper()}) on {args.family} family")
    print(f"  N={env.N} individuals | obs_dim={env.obs_dim} | actions={env.n_actions}")
    print(f"  timesteps={args.timesteps:,} | seed={args.seed}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    gnn_kwargs = None
    if args.model == "mlp":
        policy = MLPActorCritic(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            hidden=[256, 128],
        )
    else:
        adj_norm = build_pedigree_adj(ped, individuals)
        gen_depths = build_generation_depths(ped, individuals)
        policy = GNNActorCritic(
            n_individuals=env.N,
            n_genes=len(env.genes),
            pedigree=ped,
            emb_dim=64,
            gnn_layers=2,
        )
        gnn_kwargs = {"adj_norm": adj_norm, "gen_depths": gen_depths}

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {n_params:,}\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    agent = PPOAgent(
        policy=policy,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        value_coef=0.5,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gnn_kwargs=gnn_kwargs,
        device=device,
    )

    log = agent.train(
        env,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        log_every=args.log_every,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Evaluation (100 episodes, deterministic policy)")
    eval_stats = agent.evaluate_policy(env, n_episodes=100, deterministic=True)
    print(f"  Mean reward : {eval_stats['mean_reward']:.6f}")
    print(f"  Std reward  : {eval_stats['std_reward']:.6f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Compare to exact DP (only for small families)
    # ------------------------------------------------------------------
    if env.N <= 7:
        print("  Running exact DP baseline for comparison...")
        from genetic_dp.exact_dp.backward_induction import BackwardInductionSolver
        dp = BackwardInductionSolver(ped, config, genes=("gene",), verbose=0)
        v_star = dp.solve()
        print(f"  Exact DP  V*(∅) = {v_star:.6f}  (solved {dp.n_states} states in {dp.solve_time:.2f}s)")
        print(f"  RL policy mean  = {eval_stats['mean_reward']:.6f}")
        gap = abs(v_star - eval_stats['mean_reward']) / max(abs(v_star), 1e-9)
        print(f"  Optimality gap  = {gap*100:.2f}%\n")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "policy_state": policy.state_dict(),
        "model": args.model,
        "family": args.family,
        "N": env.N,
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "genes": env.genes,
        "eval": eval_stats,
        "log": log,
    }, args.save_path)
    print(f"  Saved policy → {args.save_path}")


if __name__ == "__main__":
    main()
