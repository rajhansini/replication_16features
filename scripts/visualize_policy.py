"""
Week 5: Policy visualization for the trained N=9 RL policy.

Loads artifacts/rl_policy_N9.pt (saved by eval_baselines.py) and runs
many episodes to answer:
  1. Who does the policy test first, second, ... ?  (testing-order heatmap)
  2. How many tests before stopping?               (stopping distribution)
  3. Which individuals does it skip most often?    (skip frequency)

Also runs the Myopic policy for visual comparison.

Outputs:
  artifacts/fig10_policy_viz.png

Usage:
    python -m scripts.visualize_policy
    python -m scripts.visualize_policy --policy artifacts/rl_policy_N9.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv
from genetic_dp.rl.networks import MLPActorCritic

ALLELE_FREQ = 0.1
N_EPISODES = 2000


def make_n9_pedigree() -> Pedigree:
    ped = Pedigree()
    for x in ["GF1", "GM1", "GF2", "GM2"]:
        ped.add_individual(x)
    ped.add_individual("P1", parents=("GF1", "GM1"))
    ped.add_individual("P2", parents=("GF2", "GM2"))
    for i in range(1, 4):
        ped.add_individual(f"C{i}", parents=("P1", "P2"))
    return ped


class MyopicPolicy:
    def act(self, env: GeneticTestingEnv) -> int:
        untested = env.untested
        if not untested:
            return env.N
        best_gain, best_action = -np.inf, env.N
        for ind in untested:
            gain = env._test_reward(ind) - env._no_test_reward(ind)
            if gain > best_gain:
                best_gain, best_action = gain, env._idx[ind]
        return best_action if best_gain > 0 else env.N


def collect_episodes(policy, env: GeneticTestingEnv, n: int, is_nn: bool):
    """Return list of episode dicts: {tested_order: [ind, ...], n_tests: int}."""
    episodes = []
    for _ in range(n):
        obs = env.reset()
        order = []
        while not env.done:
            if is_nn:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(env.action_mask(), dtype=torch.bool).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = policy.act(obs_t, mask=mask_t, deterministic=True)
                a = int(action.item())
            else:
                a = policy.act(env)
            obs, _, _, info = env.step(a)
            if info.get("action") == "test":
                order.append(info["individual"])
        episodes.append({"tested_order": order, "n_tests": len(order)})
    return episodes


def build_heatmap(episodes, individuals):
    """Return matrix [N_individuals, max_step+1]: freq of being tested at step t."""
    max_steps = max((e["n_tests"] for e in episodes), default=1)
    mat = np.zeros((len(individuals), max_steps))
    ind_idx = {ind: i for i, ind in enumerate(individuals)}
    for ep in episodes:
        for t, ind in enumerate(ep["tested_order"]):
            mat[ind_idx[ind], t] += 1
    mat /= len(episodes)
    return mat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="artifacts/rl_policy_N9.pt")
    p.add_argument("--n-episodes", type=int, default=N_EPISODES)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    ped = make_n9_pedigree()
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=ALLELE_FREQ)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=args.seed)

    # Load RL policy
    if not os.path.exists(args.policy):
        print(f"  Policy file not found: {args.policy}")
        print("  Run eval_baselines.py first to generate it.")
        sys.exit(1)

    rl_policy = MLPActorCritic(obs_dim=env.obs_dim, n_actions=env.n_actions,
                               hidden=[256, 128])
    rl_policy.load_state_dict(torch.load(args.policy, map_location="cpu"))
    rl_policy.eval()

    myopic = MyopicPolicy()

    print(f"  Collecting {args.n_episodes} episodes per policy...")
    rl_eps = collect_episodes(rl_policy, env, args.n_episodes, is_nn=True)
    myopic_eps = collect_episodes(myopic, env, args.n_episodes, is_nn=False)
    print("  Done.")

    # Stats
    rl_n_tests = [e["n_tests"] for e in rl_eps]
    myopic_n_tests = [e["n_tests"] for e in myopic_eps]
    print(f"  RL:     avg tests = {np.mean(rl_n_tests):.2f} ± {np.std(rl_n_tests):.2f}")
    print(f"  Myopic: avg tests = {np.mean(myopic_n_tests):.2f} ± {np.std(myopic_n_tests):.2f}")

    rl_heat = build_heatmap(rl_eps, individuals)
    myopic_heat = build_heatmap(myopic_eps, individuals)

    # ---- Figure -----------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Row 0: heatmaps
    ax_rl = fig.add_subplot(gs[0, :2])
    ax_my = fig.add_subplot(gs[1, :2])

    vmax = max(rl_heat.max(), myopic_heat.max())
    for ax, heat, title in [
        (ax_rl, rl_heat, "RL Policy — testing-order frequency"),
        (ax_my, myopic_heat, "Myopic Policy — testing-order frequency"),
    ]:
        im = ax.imshow(heat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(individuals)))
        ax.set_yticklabels(individuals, fontsize=8)
        ax.set_xlabel("Test step (0 = first test)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Fraction of episodes")

    # Column 2: stopping distributions
    ax_stop = fig.add_subplot(gs[0, 2])
    bins = range(0, env.N + 2)
    ax_stop.hist(rl_n_tests, bins=bins, alpha=0.6, color="#2196F3",
                 label=f"RL (μ={np.mean(rl_n_tests):.1f})", density=True)
    ax_stop.hist(myopic_n_tests, bins=bins, alpha=0.6, color="#4CAF50",
                 label=f"Myopic (μ={np.mean(myopic_n_tests):.1f})", density=True)
    ax_stop.set_xlabel("# tests before stopping")
    ax_stop.set_ylabel("Density")
    ax_stop.set_title("Stopping distribution (N=9)")
    ax_stop.legend(fontsize=8)
    ax_stop.grid(True, alpha=0.3)

    # Column 2 row 2: first-test preference
    ax_first = fig.add_subplot(gs[1, 2])
    rl_first = np.zeros(len(individuals))
    myopic_first = np.zeros(len(individuals))
    for ep in rl_eps:
        if ep["tested_order"]:
            rl_first[individuals.index(ep["tested_order"][0])] += 1
    for ep in myopic_eps:
        if ep["tested_order"]:
            myopic_first[individuals.index(ep["tested_order"][0])] += 1
    rl_first /= len(rl_eps)
    myopic_first /= len(myopic_eps)

    x = np.arange(len(individuals))
    ax_first.bar(x - 0.2, rl_first, 0.4, label="RL", color="#2196F3", alpha=0.8)
    ax_first.bar(x + 0.2, myopic_first, 0.4, label="Myopic", color="#4CAF50", alpha=0.8)
    ax_first.set_xticks(x)
    ax_first.set_xticklabels(individuals, rotation=45, ha="right", fontsize=7)
    ax_first.set_ylabel("P(tested first)")
    ax_first.set_title("First individual tested")
    ax_first.legend(fontsize=8)
    ax_first.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Week 5: Policy Visualization — N=9 Family", fontweight="bold", y=1.01)
    out = "artifacts/fig10_policy_viz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


if __name__ == "__main__":
    main()
