"""
Step 8: Scaling experiment — exact DP with k genes.

Phase 1 (always runs): BFS state count only — fast, no DP.
  Reports how many states exist for k=1,2,3 genes on each family.
  This tells us whether full DP is feasible before committing.

Phase 2 (if state count < MAX_STATES): Full exact DP.
  Builds belief map, runs backward induction, reports V_root and timing.

Goal: understand how the state space scales with k (number of genes)
and N (family size). If k=3, N=5 is feasible, we can generate V*
labels for 3-gene GNN training.

Upper bound on states:
  k genes, N people: (1 + 3^k)^N
  k=2, N=5:  10^5  = 100,000  (actual ~20,816)
  k=3, N=5:  28^5  = 17.2M    (actual ~3-5M expected)
  k=2, N=6:  10^6  = 1,000,000 (actual ~107,728)
  k=3, N=6:  28^6  = 481M     (likely intractable)
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from collections import deque
from itertools import product
from pathlib import Path

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import (
    FAMILY_CASES, PRESETS,
    _build_child_cpds, _build_factorized_belief_map,
    build_multigene_dataset,
    ALLELE_FREQ_REGIMES,
)
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from genetic_dp.exact_dp.utils import GENOTYPE_STATES

MAX_STATES = 10_000_000  # skip full DP above this threshold

ALLELE_FREQS_3GENE = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15, "GeneC": 0.10},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08, "GeneC": 0.08},
}

EXPERIMENTS_MATRIX = [
    # (family_label, regime, n_genes, genes_tuple)
    ("ThreeGeneration", "LowHigh",    2, ("GeneA", "GeneB")),
    ("ThreeGeneration", "LowHigh",    3, ("GeneA", "GeneB", "GeneC")),
    ("ThreeGeneration", "MediumEven", 2, ("GeneA", "GeneB")),
    ("ThreeGeneration", "MediumEven", 3, ("GeneA", "GeneB", "GeneC")),
    ("Extended",        "LowHigh",    2, ("GeneA", "GeneB")),
    ("Extended",        "LowHigh",    3, ("GeneA", "GeneB", "GeneC")),
    ("Extended",        "MediumEven", 2, ("GeneA", "GeneB")),
    ("Extended",        "MediumEven", 3, ("GeneA", "GeneB", "GeneC")),
]


def count_states_bfs(family_label: str, allele_freqs: dict, genes: tuple) -> tuple[int, float]:
    """BFS to count reachable states — no DP, just counting. Returns (n_states, seconds)."""
    pedigree    = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    individuals = pedigree.to_list()
    child_cpds  = _build_child_cpds(pedigree)
    k_states    = list(product(GENOTYPE_STATES, repeat=len(genes)))

    # Build factorized single-gene belief maps (needed to get tuple PMFs for BFS)
    from genetic_dp.exact_dp.utils import build_full_joint, build_belief_map
    single_gene_beliefs = {}
    for gene in genes:
        joint_g = build_full_joint(pedigree, GENOTYPE_STATES, allele_freqs[gene], child_cpds, genes=None)
        single_gene_beliefs[gene] = build_belief_map(pedigree, GENOTYPE_STATES, joint_g)

    t0      = time.time()
    seen    = {frozenset()}
    frontier = deque([frozenset()])
    n       = 0

    while frontier:
        state = frontier.popleft()
        n    += 1

        observed = {person for person, _ in state}
        if len(observed) >= len(individuals):
            continue

        # Get tuple PMF for each untested person to expand
        for person in individuals:
            if person in observed:
                continue
            # Build this person's tuple PMF from single-gene marginals
            per_gene_marg = {}
            for gene_idx, gene in enumerate(genes):
                projected = frozenset(
                    (p, outcome[gene_idx]) for p, outcome in state
                )
                per_gene_marg[gene] = single_gene_beliefs[gene][projected]

            for outcome in k_states:
                prob = 1.0
                for idx, gene in enumerate(genes):
                    prob *= per_gene_marg[gene][person].get(outcome[idx], 0.0)
                    if prob <= 0:
                        break
                if prob <= 0:
                    continue
                next_state = frozenset(state | {(person, outcome)})
                if next_state not in seen:
                    seen.add(next_state)
                    frontier.append(next_state)

    return n, time.time() - t0


def run_full_dp(family_label: str, regime: str, genes: tuple, preset_label: str = "Base") -> dict:
    """Run the full exact DP and return timing + key results."""
    allele_freqs = ALLELE_FREQS_3GENE[regime] if len(genes) == 3 else ALLELE_FREQ_REGIMES[regime]
    allele_freqs = {g: allele_freqs[g] for g in genes}

    tracemalloc.start()
    t0 = time.time()
    ds = build_multigene_dataset(
        family_label=family_label,
        allele_freqs=allele_freqs,
        preset_label=preset_label,
        genes=genes,
    )
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n_states":     len(ds["states"]),
        "V_root":       ds["V_root"],
        "V_stop_root":  ds["V_stop_root"],
        "elapsed_sec":  elapsed,
        "peak_mem_mb":  peak / 1e6,
        "feat_dim":     ds["X"].shape[1],
        "node_feat_dim_gnn": len(genes) * 3 + 1,
    }


def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("SCALING EXPERIMENT: exact DP with k genes")
    print(f"MAX_STATES threshold for full DP: {MAX_STATES:,}")
    print("=" * 80)

    all_results = []

    # ── Phase 1: BFS state counts ──────────────────────────────────────────────
    print("\n[PHASE 1] BFS state counts (no DP)")
    print(f"{'Family':<20} {'Regime':<12} {'Genes':>5}  {'States':>12}  {'BFS time':>10}")
    print("-" * 65)

    for family_label, regime, n_genes, genes in EXPERIMENTS_MATRIX:
        allele_freqs = ALLELE_FREQS_3GENE[regime] if n_genes == 3 else ALLELE_FREQ_REGIMES[regime]
        allele_freqs_k = {g: allele_freqs[g] for g in genes}

        n_states, bfs_time = count_states_bfs(family_label, allele_freqs_k, genes)
        feasible = n_states < MAX_STATES

        print(f"{family_label:<20} {regime:<12} {n_genes:>5}  {n_states:>12,}  {bfs_time:>8.1f}s  {'✓ feasible' if feasible else '✗ too large'}")

        all_results.append({
            "family": family_label, "regime": regime,
            "n_genes": n_genes, "genes": list(genes),
            "n_states_bfs": n_states,
            "bfs_time_sec": bfs_time,
            "feasible_for_dp": feasible,
        })

    # ── Phase 2: Full exact DP for feasible cases ──────────────────────────────
    print("\n[PHASE 2] Full exact DP (feasible cases only)")
    print(f"{'Family':<20} {'Regime':<12} {'Genes':>5}  {'States':>10}  {'Time':>8}  {'Mem MB':>8}  {'V*':>10}  {'GNN feat/node':>14}")
    print("-" * 95)

    for row in all_results:
        if not row["feasible_for_dp"]:
            print(f"{row['family']:<20} {row['regime']:<12} {row['n_genes']:>5}  {'SKIPPED (too large)':>58}")
            continue

        genes  = tuple(row["genes"])
        regime = row["regime"]
        family = row["family"]

        try:
            dp = run_full_dp(family, regime, genes)
            row["dp"] = dp
            print(
                f"{family:<20} {regime:<12} {len(genes):>5}  "
                f"{dp['n_states']:>10,}  {dp['elapsed_sec']:>6.1f}s  "
                f"{dp['peak_mem_mb']:>7.0f}  {dp['V_root']:>10.4f}  "
                f"{dp['node_feat_dim_gnn']:>14}"
            )
        except Exception as e:
            row["dp_error"] = str(e)
            print(f"{family:<20} {regime:<12} {len(genes):>5}  ERROR: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n[SUMMARY]")
    print("  For 3-gene GNN: node_feat_dim = 3*k + 1 (3 probs per gene + tested flag)")
    print("  If ThreeGeneration 3-gene DP ran: retrain GNN with node_feat_dim=10")
    print("  Next step: train on N=5, 3 genes → test generalisation on N=6 or N=10\n")

    out_path = results_dir / "scaling_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
