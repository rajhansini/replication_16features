"""
Re-verification of the myopic baseline for ONE (genes, regime, preset) config,
using Kanix's ORIGINAL genetic_dp/policy/{myopic.py,baselines.py}
(myopic_greedy = zero-lookahead, pure argmax of immediate r_test vs stop),
built on Kanix's original belief-map builder
(genetic_dp/experiments/core.py::_build_factorized_multigene_belief_snapshot)
and exact DP solver (genetic_dp/exact_dp/solver.py::solve_exact_dp_primal,
pure backward induction, no LP solver needed).

Run one config per process (SLURM array task) so memory is released between
configs instead of accumulating in one long-lived interpreter -- the earlier
all-in-one-process version grew to 16GB+ RSS and stalled.

Usage:
    python verify_myopic_one.py --genes 3 --regime LowHigh --preset Aggressive
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path("/net/projects/ranalab/rajhansini/replication_16features")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ground-up-experiments"))

from genetic_dp.config import get_config
from genetic_dp.exact_dp.utils import GENOTYPE_STATES
from genetic_dp.exact_dp.solver import solve_exact_dp_primal
from genetic_dp.experiments.core import _build_factorized_multigene_belief_snapshot
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree

from search_multigene_myopic_vs_stop import _build_child_cpds, _evaluate_myopic_root, _stop_value_from_state
from shared.data_gen import FAMILY_CASES as FAMILY_CASES_ALL, PRESETS as PRESETS_3GENE

ALLELE_REGIMES_2GENE = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12},
}
sys.path.insert(0, str(ROOT / "ground-up-experiments" / "step9_gnn_3gene"))
from build_datasets import ALLELE_FREQS as ALLELE_REGIMES_3GENE  # noqa: E402 -- single source of truth, not a hand copy
COEF_PRESETS_2GENE = {
    "Base": {
        "a_gene": {"GeneA": -0.08, "GeneB": -0.06},
        "b_gene": {"GeneA": -0.04, "GeneB": -0.03},
        "delta_gene": {"GeneA": 0.60, "GeneB": 0.70},
    },
    "Aggressive": {
        "a_gene": {"GeneA": -0.12, "GeneB": -0.09},
        "b_gene": {"GeneA": -0.06, "GeneB": -0.045},
        "delta_gene": {"GeneA": 0.70, "GeneB": 0.80},
    },
}


def build_config_n(pedigree, genes, allele_freqs, preset_label, fixed_cost, variable_cost, preset_table):
    preset = preset_table[preset_label]
    config = get_config(
        pedigree.to_list(), pedigree=pedigree, genes=genes,
        allele_freqs=allele_freqs,
        per_gene_a=dict(preset["a_gene"]),
        per_gene_b=dict(preset["b_gene"]),
        per_gene_c={g: 0.0 for g in genes},
        per_gene_delta=dict(preset["delta_gene"]),
    )
    config.fixed_cost = float(fixed_cost)
    config.variable_cost = float(variable_cost)
    return config


def run_one(family_label, genes, allele_freqs, preset_label, preset_table, fixed_cost=0.01, variable_cost=0.02):
    pedigree = generate_deterministic_pedigree(FAMILY_CASES_ALL[family_label])
    config = build_config_n(pedigree, genes, allele_freqs, preset_label, fixed_cost, variable_cost, preset_table)

    belief = _build_factorized_multigene_belief_snapshot(
        pedigree=pedigree, config=config, genes=genes,
        child_cpds=_build_child_cpds(pedigree), belief_parallelism=1,
        progress_label=None,
    )

    mu0 = {frozenset(): 1.0}
    gen_states_full = list(product(GENOTYPE_STATES, repeat=len(genes)))
    V, policy_dp = solve_exact_dp_primal(
        pedigree.to_list(), gen_states_full, mu0, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        genes=genes, a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
    )
    V_root = float(V[frozenset()])
    V_stop_root = _stop_value_from_state(
        frozenset(), pedigree=pedigree, config=config, belief=belief, belief_gene={},
    )

    myopic_root_value, myopic_policy = _evaluate_myopic_root(
        pedigree=pedigree, config=config, belief=belief,
    )

    denom = V_root - V_stop_root
    ratio2 = (V_root - myopic_root_value) / denom if abs(denom) > 1e-12 else 0.0
    root_action = myopic_policy.get(frozenset())
    return {
        "V_root": V_root, "V_stop_root": V_stop_root,
        "L_myopic_TRUE": myopic_root_value, "ratio2_TRUE": ratio2,
        "root_action_myopic": [root_action[0], root_action[1]] if root_action else None,
        "root_action_dp": [policy_dp.get(frozenset())[0], policy_dp.get(frozenset())[1]] if policy_dp.get(frozenset()) else None,
        "n_states": len(belief),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--regime", required=True)
    p.add_argument("--preset", required=True, choices=["Base", "Aggressive"])
    args = p.parse_args()

    out_dir = ROOT / "incremental_experiments" / "results" / "myopic_TRUE"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.genes == 2:
        genes = ("GeneA", "GeneB")
        allele_freqs = ALLELE_REGIMES_2GENE[args.regime]
        preset_table = COEF_PRESETS_2GENE
        key = f"ThreeGeneration_{args.regime}_{args.preset}_2gene"
    else:
        genes = ("GeneA", "GeneB", "GeneC")
        allele_freqs = ALLELE_REGIMES_3GENE[args.regime]
        preset_table = PRESETS_3GENE
        key = f"ThreeGeneration_{args.regime}_{args.preset}_3gene"

    result = run_one("ThreeGeneration", genes, allele_freqs, args.preset, preset_table)
    result["key"] = key
    out_path = out_dir / f"{key}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{key}] ratio2_TRUE={result['ratio2_TRUE']:.4f}  L={result['L_myopic_TRUE']:.4f}  "
          f"V_root={result['V_root']:.4f}  n_states={result['n_states']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
