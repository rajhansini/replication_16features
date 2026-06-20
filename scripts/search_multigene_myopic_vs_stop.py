#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Mapping

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from genetic_dp.config import get_config
from genetic_dp.exact_dp.policy import extract_policy as extract_exact_policy
from genetic_dp.exact_dp.solver import solve_exact_dual_pulp
from genetic_dp.exact_dp.utils import GENOTYPE_STATES
from genetic_dp.experiments.core import _build_factorized_multigene_belief_snapshot
from genetic_dp.models.belief import InferenceResult, lift_single_gene_posteriors_to_genes
from genetic_dp.models.genetics_cpd import make_multigene_inheritance_cpds_with_tables
from genetic_dp.models.reward import r_reward, r_reward_test
from genetic_dp.policy.baselines import myopic_greedy
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree


GENES = ("GeneA", "GeneB")

FAMILY_CASES: dict[str, list[tuple[str, str, str]]] = {
    "Extended": [
        ("Father", "Grandfather", "Grandmother"),
        ("Uncle", "Grandfather", "Grandmother"),
        ("Child", "Father", "Mother"),
    ],
    "ThreeGeneration": [
        ("Father", "Grandfather", "Grandmother"),
        ("Child", "Father", "Mother"),
    ],
}

COEF_PRESETS = {
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

_BELIEF_CACHE: dict[tuple[str, float, float], dict] = {}


def _posterior_entry(entry):
    return entry[0] if isinstance(entry, tuple) else entry


def _tested_set(state: frozenset[tuple[str, object]]) -> set[str]:
    return {person for person, _ in state}


def _merge_state(state: frozenset[tuple[str, object]], person: str, outcome: object) -> frozenset[tuple[str, object]]:
    evidence = dict(state)
    evidence[person] = outcome
    return frozenset(evidence.items())


def _build_child_cpds(pedigree) -> dict[str, object]:
    child_cpds = {}
    for child in pedigree.get_offspring():
        parent1, parent2 = pedigree.get_parents(child)
        tables = make_multigene_inheritance_cpds_with_tables(child, parent1, parent2, GENES)
        child_cpds[child] = tables[GENES[0]][1]
    return child_cpds


def _get_factorized_belief_snapshot(
    *,
    family_label: str,
    pedigree,
    config,
) -> dict:
    key = (
        family_label,
        round(float(config.allele_freqs["GeneA"]), 8),
        round(float(config.allele_freqs["GeneB"]), 8),
    )
    cached = _BELIEF_CACHE.get(key)
    if cached is not None:
        return cached

    belief_exact = _build_factorized_multigene_belief_snapshot(
        pedigree=pedigree,
        config=config,
        genes=GENES,
        child_cpds=_build_child_cpds(pedigree),
        belief_parallelism=1,
        progress_label=f"myopic-stop-{family_label}",
    )
    _BELIEF_CACHE[key] = belief_exact
    return belief_exact


def _get_gene_probs(
    state: frozenset[tuple[str, object]],
    *,
    belief: dict,
    belief_gene: dict,
    genes: tuple[str, ...],
):
    if state in belief_gene:
        return belief_gene[state]
    entry = _posterior_entry(belief[state])
    if isinstance(entry, InferenceResult):
        gene_probs = entry.get_per_gene_probs()
    else:
        gene_probs = lift_single_gene_posteriors_to_genes(entry, genes)
    belief_gene[state] = gene_probs
    return gene_probs


def _tuple_dist_for_person(
    state: frozenset[tuple[str, object]],
    person: str,
    *,
    belief: dict,
) -> Mapping[object, float]:
    entry = _posterior_entry(belief[state])
    if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
        return entry.get_tuple_pmfs().get(person, {})
    raise RuntimeError(f"Missing tuple PMFs for state={state!r}, person={person!r}")


def _stop_value_from_state(
    state: frozenset[tuple[str, object]],
    *,
    pedigree,
    config,
    belief: dict,
    belief_gene: dict,
) -> float:
    individuals = pedigree.to_list()
    entry = _posterior_entry(belief[state])
    gene_probs = _get_gene_probs(
        state,
        belief=belief,
        belief_gene=belief_gene,
        genes=tuple(config.genes),
    )
    tested = _tested_set(state)
    return float(
        sum(
            r_reward(
                person,
                entry,
                config.a,
                config.b,
                config.c,
                config.delta,
                per_gene_probs=gene_probs,
                a_gene=config.a_gene if config.a_gene else None,
                b_gene=config.b_gene if config.b_gene else None,
                c_gene=config.c_gene if config.c_gene else None,
                delta_gene=config.delta_gene if config.delta_gene else None,
            )
            for person in individuals
            if person not in tested
        )
    )


def _evaluate_myopic_root(
    *,
    pedigree,
    config,
    belief: dict,
) -> tuple[float, dict]:
    individuals = pedigree.to_list()
    genes = tuple(config.genes) if config.genes else tuple()
    belief_gene: dict = {}
    policy: dict = {}
    values: dict = {}

    common_kwargs = dict(
        belief=belief,
        individuals=individuals,
        gen_states=GENOTYPE_STATES,
        infer=None,
        a=config.a,
        b=config.b,
        c=config.c,
        delta=config.delta,
        fixed_cost=config.fixed_cost,
        variable_cost=config.variable_cost,
        belief_gene=belief_gene,
        genes=genes or None,
        a_gene=config.a_gene if config.a_gene else None,
        b_gene=config.b_gene if config.b_gene else None,
        c_gene=config.c_gene if config.c_gene else None,
        delta_gene=config.delta_gene if config.delta_gene else None,
        tuple_mode=bool(genes),
    )

    def value_at(state: frozenset[tuple[str, object]]) -> float:
        if state in values:
            return values[state]

        action = myopic_greedy(state, **common_kwargs)
        policy[state] = action

        if action[0] == "stop":
            values[state] = _stop_value_from_state(
                state,
                pedigree=pedigree,
                config=config,
                belief=belief,
                belief_gene=belief_gene,
            )
            return values[state]

        entry = _posterior_entry(belief[state])
        gene_probs = _get_gene_probs(
            state,
            belief=belief,
            belief_gene=belief_gene,
            genes=genes,
        )
        _, who, _ = action
        test_reward = r_reward_test(
            who,
            entry,
            config.a,
            config.b,
            config.c,
            config.delta,
            config.fixed_cost,
            config.variable_cost,
            per_gene_probs=gene_probs,
            a_gene=config.a_gene if config.a_gene else None,
            c_gene=config.c_gene if config.c_gene else None,
            delta_gene=config.delta_gene if config.delta_gene else None,
        )

        expected_successor = 0.0
        tuple_dist = _tuple_dist_for_person(state, who, belief=belief)
        for outcome, prob in tuple_dist.items():
            if prob <= 0.0:
                continue
            succ = _merge_state(state, who, outcome)
            if len(succ) == len(individuals):
                continue
            expected_successor += float(prob) * value_at(succ)

        values[state] = float(test_reward + expected_successor)
        return values[state]

    root = frozenset()
    return value_at(root), policy


def _build_config(
    pedigree,
    *,
    allele_freqs: Mapping[str, float],
    preset_label: str,
    a_scale: float,
    b_scale: float,
    delta_shift: float,
    fixed_cost: float,
    variable_cost: float,
):
    preset = COEF_PRESETS[preset_label]
    a_gene = {gene: a_scale * float(value) for gene, value in preset["a_gene"].items()}
    b_gene = {gene: b_scale * float(value) for gene, value in preset["b_gene"].items()}
    delta_gene = {
        gene: min(0.99, max(0.0, float(value) + float(delta_shift)))
        for gene, value in preset["delta_gene"].items()
    }

    config = get_config(
        pedigree.to_list(),
        pedigree=pedigree,
        genes=GENES,
        allele_freqs=allele_freqs,
        per_gene_a=a_gene,
        per_gene_b=b_gene,
        per_gene_c={gene: 0.0 for gene in GENES},
        per_gene_delta=delta_gene,
    )
    config.fixed_cost = float(fixed_cost)
    config.variable_cost = float(variable_cost)
    return config


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multigene Myopic vs Stop Search",
        "",
        f"- Cases searched: {len(rows)}",
        f"- Confirmed hits: {sum(1 for row in rows if row.get('hit'))}",
        "",
        "| family | preset | af_a | af_b | a_scale | b_scale | delta_shift | fixed_cost | variable_cost | exact | stop | myopic | myopic-stop | hit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("error"):
            continue
        lines.append(
            "| {family} | {preset} | {af_a:.4f} | {af_b:.4f} | {a_scale:.2f} | {b_scale:.2f} | {delta_shift:+.2f} | {fixed_cost:.4f} | {variable_cost:.4f} | {exact_root_value:.6f} | {stop_value:.6f} | {myopic_root_value:.6f} | {myopic_minus_stop:.6f} | {hit} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search multigene benchmark-family settings where myopic is worse than stop."
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=list(FAMILY_CASES.keys()),
        choices=sorted(FAMILY_CASES.keys()),
        help="Pedigree families to search.",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=list(COEF_PRESETS.keys()),
        choices=sorted(COEF_PRESETS.keys()),
        help="Coefficient presets to use as anchors.",
    )
    parser.add_argument("--af-a-values", nargs="*", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--af-b-values", nargs="*", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--a-scales", nargs="*", type=float, default=[1.0])
    parser.add_argument("--b-scales", nargs="*", type=float, default=[1.0])
    parser.add_argument("--delta-shifts", nargs="*", type=float, default=[0.0])
    parser.add_argument("--fixed-cost-values", nargs="*", type=float, default=[0.01])
    parser.add_argument("--variable-cost-values", nargs="*", type=float, default=[0.02])
    parser.add_argument(
        "--allow-genea-greater",
        action="store_true",
        help="Do not enforce GeneA allele frequency <= GeneB allele frequency.",
    )
    parser.add_argument(
        "--output-json",
        default="tests/output/multigene_myopic_vs_stop_search.json",
        help="Full JSON results path.",
    )
    parser.add_argument(
        "--output-md",
        default="tests/output/multigene_myopic_vs_stop_search.md",
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--stop-after-hit",
        action="store_true",
        help="Stop once the first hit is confirmed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    rows: list[dict] = []
    case_index = 0
    mu0 = {frozenset(): 1.0}

    for family_label, preset_label, af_a, af_b, a_scale, b_scale, delta_shift, fixed_cost, variable_cost in product(
        args.families,
        args.presets,
        args.af_a_values,
        args.af_b_values,
        args.a_scales,
        args.b_scales,
        args.delta_shifts,
        args.fixed_cost_values,
        args.variable_cost_values,
    ):
        if not args.allow_genea_greater and af_a > af_b:
            continue

        pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
        allele_freqs = {"GeneA": float(af_a), "GeneB": float(af_b)}
        config = _build_config(
            pedigree,
            allele_freqs=allele_freqs,
            preset_label=preset_label,
            a_scale=float(a_scale),
            b_scale=float(b_scale),
            delta_shift=float(delta_shift),
            fixed_cost=float(fixed_cost),
            variable_cost=float(variable_cost),
        )

        case_index += 1
        label = (
            f"{family_label}|{preset_label}|af=({af_a:.4f},{af_b:.4f})|"
            f"a={a_scale:.2f}|b={b_scale:.2f}|d={delta_shift:+.2f}|"
            f"fc={fixed_cost:.4f}|vc={variable_cost:.4f}"
        )
        print(f"[{case_index}] {label}")

        try:
            belief_exact = _get_factorized_belief_snapshot(
                family_label=family_label,
                pedigree=pedigree,
                config=config,
            )
            phi_exact_maybe = solve_exact_dual_pulp(
                pedigree.to_list(),
                list(product(GENOTYPE_STATES, repeat=len(GENES))),
                mu0,
                belief_exact,
                config.a,
                config.b,
                config.c,
                config.delta,
                config.fixed_cost,
                config.variable_cost,
                genes=GENES,
                a_gene=config.a_gene,
                b_gene=config.b_gene,
                c_gene=config.c_gene,
                delta_gene=config.delta_gene,
                base_gen_states=GENOTYPE_STATES,
            )
            if isinstance(phi_exact_maybe, tuple):
                phi_exact, phi_exact_gene = phi_exact_maybe
            else:
                phi_exact = phi_exact_maybe
                phi_exact_gene = None

            exact_policy = extract_exact_policy(
                pedigree.to_list(),
                list(product(GENOTYPE_STATES, repeat=len(GENES))),
                config.a,
                config.b,
                config.c,
                config.delta,
                phi_exact,
                belief_exact,
                config.fixed_cost,
                config.variable_cost,
                genes=GENES,
                Phi_star_gene=phi_exact_gene,
                a_gene=config.a_gene,
                b_gene=config.b_gene,
                c_gene=config.c_gene,
                delta_gene=config.delta_gene,
                base_gen_states=GENOTYPE_STATES,
            )
            myopic_root_value, myopic_policy = _evaluate_myopic_root(
                pedigree=pedigree,
                config=config,
                belief=belief_exact,
            )
            belief_gene = {}
            stop_value = _stop_value_from_state(
                frozenset(),
                pedigree=pedigree,
                config=config,
                belief=belief_exact,
                belief_gene=belief_gene,
            )
            exact_root_value = float(phi_exact[frozenset()])
            myopic_minus_stop = float(myopic_root_value) - stop_value
            row = {
                "family": family_label,
                "preset": preset_label,
                "af_a": float(af_a),
                "af_b": float(af_b),
                "a_scale": float(a_scale),
                "b_scale": float(b_scale),
                "delta_shift": float(delta_shift),
                "fixed_cost": float(fixed_cost),
                "variable_cost": float(variable_cost),
                "exact_root_value": exact_root_value,
                "stop_value": float(stop_value),
                "myopic_root_value": float(myopic_root_value),
                "myopic_minus_stop": float(myopic_minus_stop),
                "hit": bool(myopic_minus_stop < 0.0),
                "myopic_root_action": list(myopic_policy.get(frozenset(), (None, None, None))),
                "exact_root_action": list(exact_policy.get(frozenset(), (None, None, None))),
            }
            rows.append(row)
            print(
                "    exact={exact_root_value:.6f} stop={stop_value:.6f} myopic={myopic_root_value:.6f} diff={myopic_minus_stop:.6f} hit={hit}".format(
                    **row
                )
            )
            rows.sort(key=lambda item: (item.get("myopic_minus_stop", float("inf")), item["family"], item["preset"]))
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            _write_markdown(output_md, rows)
            if row["hit"] and args.stop_after_hit:
                print("Stopping after first confirmed hit.")
                return 0
        except Exception as exc:
            row = {
                "family": family_label,
                "preset": preset_label,
                "af_a": float(af_a),
                "af_b": float(af_b),
                "a_scale": float(a_scale),
                "b_scale": float(b_scale),
                "delta_shift": float(delta_shift),
                "fixed_cost": float(fixed_cost),
                "variable_cost": float(variable_cost),
                "error": repr(exc),
            }
            rows.append(row)
            print(f"    ERROR: {exc!r}")
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    hits = [row for row in rows if row.get("hit")]
    print(f"Completed {len(rows)} cases; confirmed hits={len(hits)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
