"""One concrete, fully-traced example of the testing/eval workflow, using the
real trained GNN checkpoint (seed0) on one real ThreeGeneration test config.
No new code path -- this calls the exact same precompute_vhat() / rollout()
used by run_eval() in gnn/run.py, just with trace=True and a single config so
every intermediate number is visible.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

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


gnn_mod = _load_module("gnn_run_base_trace", EXP_ROOT / "gnn" / "run.py")
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402
from exputils.eval import rollout  # noqa: E402

KEY = "ThreeGeneration_LowHigh_Base_3gene"
SEED = 0


def main():
    device = torch.device("cpu")

    with open(gnn_mod.CACHE_DIR / f"{KEY}.pkl", "rb") as f:
        ds = pickle.load(f)
    individuals = ds["individuals"]

    pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES["ThreeGeneration"])
    struct_feats = gnn_mod.compute_structural_features(pedigree, individuals)
    edge_index = gnn_mod.build_edge_index(pedigree, individuals)

    model = gnn_mod.GNN().to(device)
    ckpt = EXP_ROOT / "gnn" / "results" / "seed_runs" / f"seed{SEED}" / "gnn.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print(f"config: {KEY}")
    print(f"individuals (fixed order): {individuals}")
    print(f"true edges (parent->child): "
          f"{[(individuals[a], individuals[b]) for a, b in zip(*edge_index)]}")
    print(f"total states in this config: {len(ds['states']):,}")
    print(f"V* (true optimum from exact DP): {ds['V_root']:.4f}")
    print(f"V_stop(root) (cost of testing nobody): {ds['V_stop_root']:.4f}")
    print()

    v_hat = gnn_mod.precompute_vhat(model, ds, struct_feats, edge_index, device)
    print(f"V_hat computed for all {len(v_hat):,} states in this config "
          f"(one forward pass per state, batched)")
    print(f"V_hat range across all states: [{min(v_hat.values()):.4f}, {max(v_hat.values()):.4f}]")
    print()
    print("=" * 70)
    print("ROLLOUT TRACE -- the actual greedy policy this model induces,")
    print("scored by recursing through the REAL exact-DP state graph:")
    print("=" * 70)

    ratio2, L = rollout(v_hat, ds, log=print, trace=True)

    print("=" * 70)
    print(f"L (true expected value of following this policy)      = {L:.4f}")
    print(f"V* (true optimum)                                      = {ds['V_root']:.4f}")
    print(f"V_stop(root) (value of testing nobody)                 = {ds['V_stop_root']:.4f}")
    print(f"ratio2 = (V* - L) / (V* - V_stop(root)) = "
          f"({ds['V_root']:.4f} - {L:.4f}) / ({ds['V_root']:.4f} - {ds['V_stop_root']:.4f}) "
          f"= {ratio2:.4f}")


if __name__ == "__main__":
    main()
