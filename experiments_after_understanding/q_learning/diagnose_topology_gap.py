"""Why does GNN-Q (25%) underperform MLP-Q (58%) on action agreement, despite
GNN-Q having LOWER training loss (it fits Trio/Nuclear better than MLP does)?

Hypothesis: GNN-Q overfits to the *specific message-passing topology* of
Trio/Nuclear (3-4 people, max depth 1) and that doesn't transfer to
ThreeGeneration (5 people, depth 2 — an extra generation never seen in
training). MLP's mean-pool is topology-blind, so it has nothing
topology-specific to overfit to, and transfers more robustly by default.

This script does NOT retrain anything — loads the already-saved mlp_q.pt /
gnn_q.pt checkpoints and traces DP vs MLP-Q vs GNN-Q pick-by-pick on a couple
of ThreeGeneration test configs, flagging whether Grandfather/Grandmother
(the generation absent from training) are involved at each divergence.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE        = Path(__file__).resolve().parent
EXP_ROOT    = HERE.parent
ROOT        = EXP_ROOT.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlp_base = _load_module("mlp_run_base2", EXP_ROOT / "mlp" / "run.py")
gnn_base = _load_module("gnn_run_base2", EXP_ROOT / "gnn" / "run.py")
mlp_q_mod = _load_module("mlp_q_mod", HERE / "mlp_q.py")
gnn_q_mod = _load_module("gnn_q_mod", HERE / "gnn_q.py")

from exputils.eval import _get_entry  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

CACHE_DIR = mlp_base.CACHE_DIR
DEVICE = "cpu"

TEST_CONFIGS = mlp_base.TEST_KEYS  # all 12 held-out configs, not a hand-picked pair

GRANDPARENT_GEN = {"Grandfather", "Grandmother"}


def main():
    fam = "ThreeGeneration"
    pedigree = generate_deterministic_pedigree(mlp_base.FAMILY_CASES[fam])

    with open(CACHE_DIR / f"{fam}_LowHigh_Base_3gene.pkl", "rb") as f:
        sample_ds = pickle.load(f)
    individuals = sample_ds["individuals"]
    struct_feats = mlp_base.compute_structural_features(pedigree, individuals)
    edge_index = gnn_base.build_edge_index(pedigree, individuals)

    mlp_model = mlp_q_mod.MLPQ().to(DEVICE)
    mlp_model.load_state_dict(torch.load(HERE / "results" / "mlp_q.pt", map_location=DEVICE))
    mlp_model.eval()

    gnn_model = gnn_q_mod.GNNQ().to(DEVICE)
    gnn_model.load_state_dict(torch.load(HERE / "results" / "gnn_q.pt", map_location=DEVICE))
    gnn_model.eval()

    print(f"Individuals ({fam}): {individuals}")
    print(f"Grandparent generation (absent from Trio/Nuclear training): {sorted(GRANDPARENT_GEN)}")
    print()

    grand_step_mlp_match, grand_step_mlp_total = 0, 0
    grand_step_gnn_match, grand_step_gnn_total = 0, 0
    other_step_mlp_match, other_step_mlp_total = 0, 0
    other_step_gnn_match, other_step_gnn_total = 0, 0

    for key in TEST_CONFIGS:
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        config, belief, genes = ds["config"], ds["belief"], ds.get("genes", mlp_base.GENES)
        policy_dp = ds["policy_dp"]
        cost_vec_mlp = mlp_base.config_to_cost_vec(config)
        cost_vec_gnn = gnn_base.config_to_cost_vec(config)

        print(f"{'='*90}\n{key}\n{'='*90}")
        print(f"{'step':<5}{'state':<48}{'DP':<16}{'MLP-Q':<16}{'GNN-Q':<16}{'grandparent?'}")

        state = frozenset()
        for step in range(15):
            dp_action, dp_person, _ = policy_dp[state]
            dp_pick = "STOP" if dp_action == "stop" else f"TEST {dp_person}"

            _, mlp_person, mlp_qvals = mlp_q_mod.greedy_pick(
                mlp_model, state, belief, individuals, config, genes, struct_feats, cost_vec_mlp, DEVICE)
            mlp_pick = "STOP" if mlp_person is None else f"TEST {mlp_person}"

            gnn_kind, gnn_person = gnn_q_mod.greedy_pick(
                gnn_model, state, belief, individuals, config, genes, struct_feats, edge_index, cost_vec_gnn, DEVICE)
            gnn_pick = "STOP" if gnn_kind == "STOP" else f"TEST {gnn_person}"

            involves_grandparent = dp_person in GRANDPARENT_GEN if dp_person else False
            tested_str = "{" + ", ".join(sorted(p for p, _ in state)) + "}" if state else "root"
            flag = "YES" if involves_grandparent else "no"
            print(f"{step:<5}{tested_str:<48}{dp_pick:<16}{mlp_pick:<16}{gnn_pick:<16}{flag}")

            if involves_grandparent:
                grand_step_mlp_total += 1
                grand_step_mlp_match += int(dp_pick == mlp_pick)
                grand_step_gnn_total += 1
                grand_step_gnn_match += int(dp_pick == gnn_pick)
            else:
                other_step_mlp_total += 1
                other_step_mlp_match += int(dp_pick == mlp_pick)
                other_step_gnn_total += 1
                other_step_gnn_match += int(dp_pick == gnn_pick)

            if dp_action == "stop":
                break
            per_gene, tuple_pmfs = _get_entry(belief, state, genes)
            pmf = tuple_pmfs.get(dp_person, {})
            outcome = max(pmf.items(), key=lambda kv: kv[1])[0]
            state = frozenset(state | {(dp_person, outcome)})
        print()

    print(f"{'='*90}\nSUMMARY\n{'='*90}")
    print(f"Steps where DP's action IS a grandparent (untrained generation):")
    print(f"  MLP-Q match: {grand_step_mlp_match}/{grand_step_mlp_total}")
    print(f"  GNN-Q match: {grand_step_gnn_match}/{grand_step_gnn_total}")
    print(f"Steps where DP's action is NOT a grandparent (parent/child/root, seen in training):")
    print(f"  MLP-Q match: {other_step_mlp_match}/{other_step_mlp_total}")
    print(f"  GNN-Q match: {other_step_gnn_match}/{other_step_gnn_total}")


if __name__ == "__main__":
    main()
