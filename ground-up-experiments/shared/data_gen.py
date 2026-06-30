"""Generate (belief_state_vector, V*) training pairs from the exact DP.

Single gene: state = frozenset({(person, int)}), int in {0,1,2}
Two genes:   state = frozenset({(person, (int,int))}), tuple in {0,1,2}^2

X = flattened posterior probabilities per person per gene
Y = V*(s) from backward-induction exact DP
"""
from __future__ import annotations

import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from genetic_dp.config import get_config
from genetic_dp.exact_dp.utils import (
    GENOTYPE_STATES,
    build_belief_map,
    build_full_joint,
    lift_tuple_posteriors_to_genes,
)
from genetic_dp.exact_dp.solver import solve_exact_dp_primal
from genetic_dp.models.belief import InferenceResult
from genetic_dp.models.genetics_cpd import make_inheritance_genotype_cpd_with_table
from genetic_dp.models.reward import r_reward
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree

# ─── family definitions ───────────────────────────────────────────────────────

FAMILY_CASES = {
    "ThreeGeneration": [
        ("Father", "Grandfather", "Grandmother"),
        ("Child", "Father", "Mother"),
    ],
    "Extended": [
        ("Father", "Grandfather", "Grandmother"),
        ("Uncle", "Grandfather", "Grandmother"),
        ("Child", "Father", "Mother"),
    ],
}

# GeneA=0.02 (rare), GeneB=0.15 (common) — from original8 "LowHigh"
# GeneA=0.08, GeneB=0.08 — from original8 "MediumEven"
ALLELE_FREQ_REGIMES = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
}

# Per-gene coefficients matching PI's COEF_PRESETS exactly.
# GeneB has different (smaller-magnitude) a/b and higher delta than GeneA.
PRESETS = {
    "Base": {
        "a_gene":     {"GeneA": -0.08, "GeneB": -0.06},
        "b_gene":     {"GeneA": -0.04, "GeneB": -0.03},
        "delta_gene": {"GeneA":  0.60, "GeneB":  0.70},
    },
    "Aggressive": {
        "a_gene":     {"GeneA": -0.12, "GeneB": -0.09},
        "b_gene":     {"GeneA": -0.06, "GeneB": -0.045},
        "delta_gene": {"GeneA":  0.70, "GeneB":  0.80},
    },
}

# ─── helpers ─────────────────────────────────────────────────────────────────

def _build_child_cpds(pedigree) -> dict:
    child_cpds = {}
    for child in pedigree.get_offspring():
        p1, p2 = pedigree.get_parents(child)
        _, table = make_inheritance_genotype_cpd_with_table(child, p1, p2)
        child_cpds[child] = table
    return child_cpds


def _build_factorized_belief_map(
    pedigree, individuals, genes, allele_freqs, child_cpds, two_gene_states
) -> dict:
    """
    BFS over reachable states, building per-gene beliefs by projection.

    For each state s = frozenset({(person, (g_A,g_B)), ...}):
      - Project s to per-gene single-gene states
      - Look up per-gene single-gene beliefs (fast, already computed)
      - Combine per-gene marginals as joint tuple PMFs (independent genes)
      - Store as InferenceResult

    This mirrors _build_factorized_multigene_belief_snapshot in experiments/core.py
    but without importing that module (which pulls in gurobipy at module level).
    """
    # Build single-gene belief maps (fast: 3^n × |states| ops)
    single_gene_beliefs = {}
    for gene_idx, gene in enumerate(genes):
        joint_g = build_full_joint(
            pedigree, GENOTYPE_STATES, allele_freqs[gene], child_cpds, genes=None
        )
        single_gene_beliefs[gene] = build_belief_map(pedigree, GENOTYPE_STATES, joint_g)

    belief: dict = {}
    frontier = deque([frozenset()])
    seen: set = {frozenset()}

    while frontier:
        state = frontier.popleft()

        # Project state to per-gene single-gene states and get marginals
        per_gene_marg: dict[str, dict] = {}
        for gene_idx, gene in enumerate(genes):
            projected = frozenset(
                (person, outcome[gene_idx])
                for person, outcome in state
            )
            per_gene_marg[gene] = single_gene_beliefs[gene][projected]  # {person: {g: p}}

        # Build joint tuple PMFs (independent assortment between genes)
        tuple_pmfs: dict = {}
        for person in individuals:
            pmf: dict = {}
            for outcome in two_gene_states:
                prob = 1.0
                for idx, gene in enumerate(genes):
                    prob *= per_gene_marg[gene][person].get(outcome[idx], 0.0)
                    if prob <= 0.0:
                        break
                if prob > 0.0:
                    pmf[outcome] = prob
            total = sum(pmf.values())
            if total > 0.0 and abs(total - 1.0) > 1e-12:
                pmf = {k: v / total for k, v in pmf.items()}
            tuple_pmfs[person] = pmf

        gene_first = {
            gene: {person: dict(per_gene_marg[gene][person]) for person in individuals}
            for gene in genes
        }
        belief[state] = InferenceResult(
            marginals={person: dict(per_gene_marg[genes[0]][person]) for person in individuals},
            tuple_pmfs=tuple_pmfs,
            per_gene=gene_first,
            gene_order=genes,
            gen_states=GENOTYPE_STATES,
        )

        # Expand: test each untested person under each possible outcome
        observed = {person for person, _ in state}
        if len(observed) >= len(individuals):
            continue
        for person in individuals:
            if person in observed:
                continue
            for outcome, prob in tuple_pmfs[person].items():
                if prob <= 0.0:
                    continue
                next_state = frozenset(state | {(person, outcome)})
                if next_state not in seen:
                    seen.add(next_state)
                    frontier.append(next_state)

    return belief


def _is_reachable(state, belief, individuals) -> bool:
    """A state is reachable if its marginals sum to 1 (not an impossible observation)."""
    p_s = belief[state]
    first = individuals[0]
    return sum(p_s[first].values()) > 1e-9


def state_to_vector_single_gene(state, belief, individuals) -> np.ndarray:
    """Single gene: 3 floats per person → vector of length n_people*3."""
    p_s = belief[state]
    x = []
    for person in individuals:
        dist = p_s[person]
        x.extend([dist[0], dist[1], dist[2]])
    return np.array(x, dtype=np.float32)


def state_to_vector_two_genes(state, belief, individuals, genes=("GeneA", "GeneB")) -> np.ndarray:
    """Two genes: 6 floats per person → vector of length n_people*6."""
    from genetic_dp.models.belief import InferenceResult
    entry = belief[state]
    if isinstance(entry, InferenceResult):
        per_gene = entry.get_per_gene_probs()
    else:
        per_gene = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
    x = []
    for person in individuals:
        for gene in genes:
            dist = per_gene[gene][person]
            x.extend([dist[0], dist[1], dist[2]])
    return np.array(x, dtype=np.float32)


# ─── single-gene dataset ─────────────────────────────────────────────────────

def build_single_gene_dataset(
    family_label: str = "ThreeGeneration",
    allele_freq: float = 0.02,
    preset_label: str = "Base",
    fixed_cost: float = 0.01,
    variable_cost: float = 0.02,
) -> dict:
    """
    Build (X, Y) dataset for a single-gene exact DP run.

    Returns a dict with keys:
      states, X, Y, belief, V_star, policy_dp, config, individuals,
      pedigree, V_root, V_stop_root, family_label, allele_freq, preset_label
    """
    preset = PRESETS[preset_label]
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    individuals = pedigree.to_list()

    child_cpds = _build_child_cpds(pedigree)

    config = get_config(
        individuals,
        a_base=preset["a_gene"]["GeneA"],
        b_base=preset["b_gene"]["GeneA"],
        c_base=0.0,
        delta_base=preset["delta_gene"]["GeneA"],
        pedigree=pedigree,
        allele_freq=allele_freq,
    )
    config.fixed_cost = fixed_cost
    config.variable_cost = variable_cost

    joint = build_full_joint(pedigree, GENOTYPE_STATES, allele_freq, child_cpds, genes=None)
    belief = build_belief_map(pedigree, GENOTYPE_STATES, joint)

    mu0 = {frozenset(): 1.0}
    V, policy = solve_exact_dp_primal(
        individuals, GENOTYPE_STATES, mu0, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
    )

    p_root = belief[frozenset()]
    v_stop_root = float(sum(
        r_reward(k, p_root, config.a, config.b, config.c, config.delta)
        for k in individuals
    ))

    states, X_list, Y_list = [], [], []
    for state, v_star in V.items():
        if not _is_reachable(state, belief, individuals):
            continue
        states.append(state)
        X_list.append(state_to_vector_single_gene(state, belief, individuals))
        Y_list.append(float(v_star))

    return {
        "states":        states,
        "X":             np.stack(X_list),
        "Y":             np.array(Y_list, dtype=np.float32),
        "belief":        belief,
        "V_star":        V,
        "policy_dp":     policy,
        "config":        config,
        "individuals":   individuals,
        "pedigree":      pedigree,
        "V_root":        float(V[frozenset()]),
        "V_stop_root":   v_stop_root,
        "family_label":  family_label,
        "allele_freq":   allele_freq,
        "preset_label":  preset_label,
        "n_genes":       1,
        "genes":         None,
    }


# ─── two-gene dataset ─────────────────────────────────────────────────────────

def build_two_gene_dataset(
    family_label: str = "ThreeGeneration",
    allele_freqs: Optional[dict] = None,
    preset_label: str = "Base",
    fixed_cost: float = 0.01,
    variable_cost: float = 0.02,
    genes: tuple = ("GeneA", "GeneB"),
) -> dict:
    """
    Build (X, Y) dataset for a two-gene exact DP run.

    Uses the factorized BFS belief-map builder from experiments/core.py —
    the same approach the existing ADP solver uses. This is orders of magnitude
    faster than brute-force joint enumeration.

    State outcomes are tuples (g_GeneA, g_GeneB) ∈ {0,1,2}^2.
    X has n_people * 6 features (3 per gene per person).
    """
    if allele_freqs is None:
        allele_freqs = {"GeneA": 0.02, "GeneB": 0.15}

    preset = PRESETS[preset_label]
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    individuals = pedigree.to_list()

    child_cpds = _build_child_cpds(pedigree)

    config = get_config(
        individuals,
        a_base=0.0,
        b_base=0.0,
        c_base=0.0,
        delta_base=0.0,
        pedigree=pedigree,
        genes=genes,
        allele_freqs=allele_freqs,
        per_gene_a={g: preset["a_gene"][g] for g in genes},
        per_gene_b={g: preset["b_gene"][g] for g in genes},
        per_gene_c={g: 0.0 for g in genes},
        per_gene_delta={g: preset["delta_gene"][g] for g in genes},
    )
    config.fixed_cost = fixed_cost
    config.variable_cost = variable_cost

    two_gene_states = list(product(GENOTYPE_STATES, repeat=len(genes)))

    belief = _build_factorized_belief_map(
        pedigree, individuals, genes, allele_freqs, child_cpds, two_gene_states
    )

    mu0 = {frozenset(): 1.0}
    V, policy = solve_exact_dp_primal(
        individuals, two_gene_states, mu0, belief,
        config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        genes=genes,
        a_gene=config.a_gene,
        b_gene=config.b_gene,
        c_gene=config.c_gene,
        delta_gene=config.delta_gene,
    )

    root_entry = belief[frozenset()]
    per_gene_root = root_entry.get_per_gene_probs()
    p_root = root_entry.marginals
    v_stop_root = float(sum(
        r_reward(
            k, p_root, config.a, config.b, config.c, config.delta,
            per_gene_probs=per_gene_root,
            a_gene=config.a_gene, b_gene=config.b_gene,
            c_gene=config.c_gene, delta_gene=config.delta_gene,
        )
        for k in individuals
    ))

    states, X_list, Y_list = [], [], []
    for state, v_star in V.items():
        if state not in belief:
            continue
        states.append(state)
        X_list.append(state_to_vector_two_genes(state, belief, individuals, genes))
        Y_list.append(float(v_star))

    return {
        "states":          states,
        "X":               np.stack(X_list),
        "Y":               np.array(Y_list, dtype=np.float32),
        "belief":          belief,
        "V_star":          V,
        "policy_dp":       policy,
        "config":          config,
        "individuals":     individuals,
        "pedigree":        pedigree,
        "V_root":          float(V[frozenset()]),
        "V_stop_root":     v_stop_root,
        "family_label":    family_label,
        "allele_freqs":    allele_freqs,
        "preset_label":    preset_label,
        "n_genes":         2,
        "genes":           genes,
        "two_gene_states": two_gene_states,
    }
