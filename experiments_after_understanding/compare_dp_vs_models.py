"""DP vs MLP vs GNN: which person each policy picks, at every step of one held-out test config.

For a fixed test config, walks the trajectory starting at root, following the DP-optimal
outcome at each test. At each visited state, prints what DP actually did (from the cached
policy_dp, computed by exact dynamic programming) side-by-side with what the trained MLP
and GNN would greedily pick at that same state (via their own V_hat).

Usage:
    python compare_dp_vs_models.py > dp_vs_mlp_vs_gnn.txt
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "mlp"))
sys.path.insert(0, str(HERE / "gnn"))

import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlp_mod = _load_module("mlp_run", HERE / "mlp" / "run.py")
gnn_mod = _load_module("gnn_run", HERE / "gnn" / "run.py")

from exputils.eval import _get_entry, _stop_val, _q_hat  # noqa: E402

DEVICE = "cpu"


def build_struct(fam, mod):
    from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
    sample_key = f"{fam}_LowHigh_Base_3gene"
    with open(mod.CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
        sample_ds = pickle.load(f)
    pedigree = generate_deterministic_pedigree(mod.FAMILY_CASES[fam])
    return mod.compute_structural_features(pedigree, sample_ds["individuals"])


def greedy_pick(state, belief, individuals, config, genes, v_hat):
    per_gene, tuple_pmfs = _get_entry(belief, state, genes)
    tested = {p for p, _ in state}
    untested = [p for p in individuals if p not in tested]
    if not untested:
        return "STOP", None
    v_stop = _stop_val(per_gene, individuals, tested, config)
    q_hats = {p: _q_hat(state, p, per_gene, tuple_pmfs, v_hat, belief) for p in untested}
    best_person = max(untested, key=lambda p: q_hats[p])
    if q_hats[best_person] <= v_stop:
        return "STOP", None
    return "TEST", best_person


def most_likely_outcome(state, person, belief, genes):
    entry = belief[state]
    from genetic_dp.models.belief import InferenceResult
    if isinstance(entry, InferenceResult):
        tuple_pmfs = entry.get_tuple_pmfs()
    else:
        from genetic_dp.exact_dp.utils import lift_tuple_posteriors_to_genes
        _, tuple_pmfs = _get_entry(belief, state, genes)
    pmf = tuple_pmfs.get(person, {})
    return max(pmf.items(), key=lambda kv: kv[1])[0]


def compare_one_config(key: str, fam: str, out):
    with open(mlp_mod.CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)

    individuals = ds["individuals"]
    config = ds["config"]
    belief = ds["belief"]
    genes = ds.get("genes", mlp_mod.GENES)
    policy_dp = ds["policy_dp"]

    struct_mlp = build_struct(fam, mlp_mod)
    struct_gnn = build_struct(fam, gnn_mod)

    mlp_model = mlp_mod.MLP().to(DEVICE)
    mlp_model.load_state_dict(torch.load(mlp_mod.HERE / "results" / "mlp.pt", map_location=DEVICE))
    mlp_model.eval()

    gnn_model = gnn_mod.GNN().to(DEVICE)
    gnn_model.load_state_dict(torch.load(gnn_mod.HERE / "results" / "gnn.pt", map_location=DEVICE))
    gnn_model.eval()

    pedigree = gnn_mod.generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
    edge_index = gnn_mod.build_edge_index(pedigree, individuals)

    v_hat_mlp = mlp_mod.precompute_vhat(mlp_model, ds, struct_mlp, DEVICE)
    v_hat_gnn = gnn_mod.precompute_vhat(gnn_model, ds, struct_gnn, edge_index, DEVICE)

    belief["config"] = config

    out.write(f"\n{'='*70}\n{key}\n{'='*70}\n")
    out.write(f"{'step':<5}{'state':<45}{'DP':<18}{'MLP':<18}{'GNN':<18}{'agree?'}\n")

    state = frozenset()
    step = 0
    while True:
        dp_action, dp_person, _ = policy_dp[state]
        dp_pick = "STOP" if dp_action == "stop" else f"TEST {dp_person}"

        mlp_kind, mlp_person = greedy_pick(state, belief, individuals, config, genes, v_hat_mlp)
        mlp_pick = "STOP" if mlp_kind == "STOP" else f"TEST {mlp_person}"

        gnn_kind, gnn_person = greedy_pick(state, belief, individuals, config, genes, v_hat_gnn)
        gnn_pick = "STOP" if gnn_kind == "STOP" else f"TEST {gnn_person}"

        tested_str = "{" + ", ".join(sorted(p for p, _ in state)) + "}" if state else "root"
        agree = "ALL MATCH" if dp_pick == mlp_pick == gnn_pick else (
            "dp=mlp" if dp_pick == mlp_pick else ("dp=gnn" if dp_pick == gnn_pick else "all differ")
        )
        out.write(f"{step:<5}{tested_str:<45}{dp_pick:<18}{mlp_pick:<18}{gnn_pick:<18}{agree}\n")

        if dp_action == "stop":
            break
        outcome = most_likely_outcome(state, dp_person, belief, genes)
        state = frozenset(state | {(dp_person, outcome)})
        step += 1
        if step > 20:
            out.write("  ... (truncated)\n")
            break

    del belief["config"]


if __name__ == "__main__":
    out = sys.stdout
    for key in mlp_mod.TEST_KEYS:
        fam = key.split("_")[0]
        compare_one_config(key, fam, out)
