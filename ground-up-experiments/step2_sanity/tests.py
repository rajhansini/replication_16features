"""Step 2: Protective sanity tests — hallucination guards on V*(s).

Run:
    python step2_sanity/tests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_single_gene_dataset, build_two_gene_dataset
from shared.evaluate import sanity_checks


def _run_suite(label: str, ds: dict) -> dict[str, bool]:
    """Run all sanity tests on dataset ds. Returns {test_name: passed}."""
    from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
    from genetic_dp.models.belief  import InferenceResult
    from genetic_dp.models.reward  import r_reward

    def _marg(entry):
        return entry.marginals if isinstance(entry, InferenceResult) else entry

    belief      = ds["belief"]
    V_star      = ds["V_star"]
    individuals = ds["individuals"]
    config      = ds["config"]
    genes       = ds.get("genes")
    two_gene    = genes is not None

    results: dict[str, bool] = {}

    # ── T1: root state exists ────────────────────────────────────────────────
    t1 = (frozenset() in V_star) and (frozenset() in belief)
    results["root_state_exists"] = t1
    _report("root_state_exists", t1)

    # ── T2: V*(root) >= V_stop(root) ────────────────────────────────────────
    t2 = ds["V_root"] >= ds["V_stop_root"] - 1e-9
    results["v_root_geq_v_stop_root"] = t2
    _report("v_root_geq_v_stop_root", t2,
            f"V*={ds['V_root']:.6f}  V_stop={ds['V_stop_root']:.6f}")

    # ── T3: V*(s) >= V_stop(s) everywhere ───────────────────────────────────
    fails = 0
    for state, v_star in V_star.items():
        entry = belief[state]
        marg  = _marg(entry)
        if sum(marg[individuals[0]].values()) < 1e-9:
            continue
        tested = {i for i, _ in state}
        if two_gene:
            per_gene = (entry.get_per_gene_probs() if isinstance(entry, InferenceResult)
                        else lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES))
            v_stop = float(sum(
                r_reward(k, marg, config.a, config.b, config.c, config.delta,
                         per_gene_probs=per_gene,
                         a_gene=config.a_gene, b_gene=config.b_gene,
                         c_gene=config.c_gene, delta_gene=config.delta_gene)
                for k in individuals if k not in tested
            ))
        else:
            v_stop = float(sum(
                r_reward(k, marg, config.a, config.b, config.c, config.delta)
                for k in individuals if k not in tested
            ))
        if v_star < v_stop - 1e-9:
            fails += 1
    t3 = fails == 0
    results["v_star_geq_v_stop_everywhere"] = t3
    _report("v_star_geq_v_stop_everywhere", t3,
            f"{fails} violation(s)" if not t3 else "")

    # ── T4: leaf states have V*(s) = 0 ──────────────────────────────────────
    n_people = len(individuals)
    fails = 0
    for state, v_star in V_star.items():
        if len(state) == n_people and abs(v_star) > 1e-9:
            fails += 1
    t4 = fails == 0
    results["leaf_value_zero"] = t4
    _report("leaf_value_zero", t4, f"{fails} violation(s)" if not t4 else "")

    # ── T5: tested person's belief is degenerate ─────────────────────────────
    fails = 0
    for state, entry in belief.items():
        marg = _marg(entry)
        if sum(marg[individuals[0]].values()) < 1e-9:
            continue
        for person, obs_g in state:
            primary_g = obs_g[0] if isinstance(obs_g, tuple) else obs_g
            dist = marg[person]
            if abs(dist.get(primary_g, 0.0) - 1.0) > 1e-6:
                fails += 1
    t5 = fails == 0
    results["tested_belief_degenerate"] = t5
    _report("tested_belief_degenerate", t5, f"{fails} violation(s)" if not t5 else "")

    # ── T6: V*(s) decreases with more tests (monotonicity sanity) ────────────
    # Not strictly required but a useful smell test: testing more people can
    # only help, so V*(s') <= V*(s) when s' has one more observation, IF the
    # new observation is positive (informative). We skip this test for now
    # as it's not always monotone — just log count.
    n_total = sum(
        1 for s, entry in belief.items()
        if sum(_marg(entry)[individuals[0]].values()) > 1e-9
    )
    results["n_reachable_states"] = n_total  # informational, not a bool

    n_pass = sum(v for k, v in results.items() if isinstance(v, bool))
    n_tests = sum(1 for v in results.values() if isinstance(v, bool))
    print(f"\n  [{label}]  {n_pass}/{n_tests} tests passed")
    return results


def _report(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    msg    = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    all_results = {}

    configs = [
        ("single_GeneA", lambda: build_single_gene_dataset(
            "ThreeGeneration", allele_freq=0.02, preset_label="Base")),
        ("single_GeneB", lambda: build_single_gene_dataset(
            "ThreeGeneration", allele_freq=0.15, preset_label="Base")),
        ("two_genes_LowHigh", lambda: build_two_gene_dataset(
            "ThreeGeneration",
            allele_freqs={"GeneA": 0.02, "GeneB": 0.15},
            preset_label="Base")),
    ]

    for label, builder in configs:
        print(f"\n{'='*60}")
        print(f"Sanity tests: {label}")
        print(f"{'='*60}")
        ds = builder()
        r  = _run_suite(label, ds)
        all_results[label] = {k: (bool(v) if isinstance(v, bool) else v) for k, v in r.items()}

    (results_dir / "sanity_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {results_dir / 'sanity_results.json'}")


if __name__ == "__main__":
    main()
