"""
Week 5: Baseline comparison — RL vs Myopic vs Random.

Myopic policy: at each step, test the individual where
  _test_reward(i) - _no_test_reward(i) is maximised; stop when that
  gain is <= 0 for all untested individuals.

Random policy: pick uniformly from all valid actions (including stop).

RL policy: trained PPO; also saved to artifacts/rl_policy_N{n}.pt for
  use by visualize_policy.py.

Evaluates on N=4,6,7,9 (all have known exact V*).

Outputs:
  artifacts/baselines_results.json
  artifacts/fig8_baselines.png
  artifacts/rl_policy_N{n}.pt   (for each N)

Usage:
    python -u -m scripts.eval_baselines 2>&1 | tee artifacts/baselines_run.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

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

ALLELE_FREQ = 0.1
KNOWN_EXACT = {
    4: -0.07462713157894738,
    6: -0.11021914123180501,
    7: -0.13024273894687500,
    9: -0.16611911152031250,
}


# ---------------------------------------------------------------------------
# Pedigree builder (same as benchmark_scale)
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
    return ped


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniform random over all valid actions (including stop)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: GeneticTestingEnv) -> int:
        mask = env.action_mask()
        valid = np.where(mask)[0]
        return int(self.rng.choice(valid))


class MyopicPolicy:
    """Test the individual with the highest marginal benefit; stop when none is positive.

    At each step, for each untested individual i computes:
        gain(i) = _test_reward(i) - _no_test_reward(i)

    If max(gain) > 0, test the best i. Otherwise stop.
    This is the one-step greedy approximation to the full DP value.
    """

    def act(self, env: GeneticTestingEnv) -> int:
        untested = env.untested
        if not untested:
            return env.N  # forced stop

        best_gain = -np.inf
        best_action = env.N

        for ind in untested:
            gain = env._test_reward(ind) - env._no_test_reward(ind)
            if gain > best_gain:
                best_gain = gain
                best_action = env._idx[ind]

        return best_action if best_gain > 0 else env.N


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_policy(policy, env: GeneticTestingEnv, n_episodes: int) -> Dict:
    rewards = []
    n_tests = []

    for _ in range(n_episodes):
        obs = env.reset()
        ep_r = 0.0
        while not env.done:
            if isinstance(policy, MLPActorCritic):
                dev = next(policy.parameters()).device
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(dev)
                mask_t = torch.tensor(
                    env.action_mask(), dtype=torch.bool
                ).unsqueeze(0).to(dev)
                with torch.no_grad():
                    action, _, _ = policy.act(obs_t, mask=mask_t, deterministic=True)
                a = int(action.item())
            else:
                a = policy.act(env)
            obs, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
        n_tests.append(len(env.tested))

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "mean_tests": float(np.mean(n_tests)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[4, 6, 7, 9])
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/baselines_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  Week 5: Baseline Comparison")
    print(f"  Sizes: {args.sizes}  |  timesteps: {args.timesteps:,}")
    print(f"  Eval episodes: {args.n_eval}  |  Device: {device}")
    print("=" * 60)

    results = []

    for n in args.sizes:
        print(f"\n── N={n} {'─'*45}")
        ped = make_pedigree(n)
        individuals = ped.to_list()
        config = get_config(individuals, pedigree=ped, allele_freq=ALLELE_FREQ)
        env = GeneticTestingEnv(ped, config, genes=("gene",), seed=args.seed)
        exact_v = KNOWN_EXACT.get(n)

        # ---- Random baseline ---------------------------------------------
        rand_pol = RandomPolicy(seed=args.seed)
        rand_stats = evaluate_policy(rand_pol, env, args.n_eval)
        print(f"  Random:  mean={rand_stats['mean']:.6f} ± {rand_stats['std']:.4f}"
              f"  avg_tests={rand_stats['mean_tests']:.1f}")

        # ---- Myopic baseline ---------------------------------------------
        myopic_pol = MyopicPolicy()
        myopic_stats = evaluate_policy(myopic_pol, env, args.n_eval)
        print(f"  Myopic:  mean={myopic_stats['mean']:.6f} ± {myopic_stats['std']:.4f}"
              f"  avg_tests={myopic_stats['mean_tests']:.1f}")

        # ---- Train RL ----------------------------------------------------
        policy = MLPActorCritic(
            obs_dim=env.obs_dim, n_actions=env.n_actions, hidden=[256, 128]
        )
        agent = PPOAgent(
            policy=policy, lr=3e-4, clip_ratio=0.2,
            entropy_coef=0.01, n_epochs=4, batch_size=64, device=device,
        )
        t0 = time.time()
        agent.train(env, total_timesteps=args.timesteps, rollout_steps=512,
                    log_every=max(args.timesteps // (512 * 5), 1))
        train_time = time.time() - t0

        rl_stats = evaluate_policy(policy, env, args.n_eval)
        print(f"  RL:      mean={rl_stats['mean']:.6f} ± {rl_stats['std']:.4f}"
              f"  avg_tests={rl_stats['mean_tests']:.1f}  train={train_time:.0f}s")

        # Save policy for visualizer
        policy_path = f"artifacts/rl_policy_N{n}.pt"
        torch.save(policy.state_dict(), policy_path)
        print(f"  Saved policy → {policy_path}")

        if exact_v is not None:
            for label, stats in [("random", rand_stats), ("myopic", myopic_stats), ("rl", rl_stats)]:
                gap = abs(exact_v - stats["mean"]) / abs(exact_v) * 100
                print(f"    {label:>6} gap: {gap:.2f}%")

        results.append({
            "N": n,
            "exact_V0": exact_v,
            "random": rand_stats,
            "myopic": myopic_stats,
            "rl": rl_stats,
            "rl_train_time_s": float(train_time),
        })

    # ---- Save results ------------------------------------------------
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {args.out}")

    # ---- Summary table -----------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'N':>3} | {'Exact V*':>10} | {'Random':>10} | {'Myopic':>10} | {'RL':>10}")
    print("-" * 72)
    for r in results:
        ev = f"{r['exact_V0']:.6f}" if r["exact_V0"] else "    N/A"
        print(f"{r['N']:>3} | {ev:>10} | "
              f"{r['random']['mean']:>10.6f} | "
              f"{r['myopic']['mean']:>10.6f} | "
              f"{r['rl']['mean']:>10.6f}")
    print("=" * 72)

    # ---- Figure -------------------------------------------------------
    _plot_baselines(results)


def _plot_baselines(results: List[Dict]):
    Ns = [r["N"] for r in results]
    labels = ["Exact V*", "RL (PPO)", "Myopic", "Random"]
    colors = ["#FF5722", "#2196F3", "#4CAF50", "#9E9E9E"]

    x = np.arange(len(Ns))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute reward
    for j, (label, color, key) in enumerate(zip(
        labels, colors,
        ["exact_V0", "rl", "myopic", "random"]
    )):
        vals = []
        errs = []
        for r in results:
            if key == "exact_V0":
                vals.append(r["exact_V0"] if r["exact_V0"] else np.nan)
                errs.append(0)
            else:
                vals.append(r[key]["mean"])
                errs.append(r[key]["std"])
        offset = (j - 1.5) * width
        bars = ax1.bar(x + offset, vals, width, label=label, color=color,
                       edgecolor="white", alpha=0.9)
        ax1.errorbar(x + offset, vals, yerr=errs, fmt="none",
                     color="black", capsize=2, lw=0.8)

    ax1.set_xticks(x); ax1.set_xticklabels([f"N={n}" for n in Ns])
    ax1.set_ylabel("Episode reward"); ax1.set_title("Reward by policy")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

    # Right: optimality gap %
    gap_labels = ["RL (PPO)", "Myopic", "Random"]
    gap_colors = ["#2196F3", "#4CAF50", "#9E9E9E"]
    gap_keys   = ["rl", "myopic", "random"]

    for j, (label, color, key) in enumerate(zip(gap_labels, gap_colors, gap_keys)):
        gaps = []
        for r in results:
            ev = r["exact_V0"]
            if ev:
                gaps.append(abs(ev - r[key]["mean"]) / abs(ev) * 100)
            else:
                gaps.append(np.nan)
        offset = (j - 1) * width
        ax2.bar(x + offset, gaps, width, label=label, color=color,
                edgecolor="white", alpha=0.9)

    ax2.axhline(5, ls="--", color="red", lw=1, label="5% threshold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"N={n}" for n in Ns])
    ax2.set_ylabel("Optimality gap (%)"); ax2.set_title("Gap from exact V*")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0)

    fig.suptitle("Week 5: RL vs Baselines", fontweight="bold")
    fig.tight_layout()
    out = "artifacts/fig8_baselines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


if __name__ == "__main__":
    main()
