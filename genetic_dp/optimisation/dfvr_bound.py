from __future__ import annotations

import itertools
import heapq
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .bellman_rowgen import bellman_violation
from .postprocess import phi_hat
from ..models.belief import (
    InferenceResult,
    lift_single_gene_posteriors_to_genes,
    propagate_all_marginals_safe,
    propagate_multigene_marginals,
)
from ..models.genetics_cpd import genotype_node_name
from ..models.outcomes import canonical_gene_order
from ..models.reward import r_reward, r_reward_testp
from ..utils.state_indexer import StateIndexer


def serialize_dfvr_state(state: frozenset) -> list[list[object]]:
    items = []
    for person, outcome in sorted(state, key=lambda pair: pair[0]):
        payload = list(outcome) if isinstance(outcome, tuple) else outcome
        items.append([person, payload])
    return items


def deserialize_dfvr_state(items: Sequence[Sequence[object]]) -> frozenset:
    if not isinstance(items, Sequence):
        raise ValueError(f"State payload must be a sequence, got {type(items).__name__}.")
    parsed = {}
    for pair in items:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"Invalid state pair payload: {pair!r}")
        person = pair[0]
        outcome = pair[1]
        if not isinstance(person, str):
            raise ValueError(f"State person must be str, got {type(person).__name__}.")
        if isinstance(outcome, list):
            outcome = tuple(outcome)
        parsed[person] = outcome
    return frozenset(parsed.items())


def load_dfvr_stateset(path: Union[str, Path]) -> list[frozenset]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        states_payload = payload.get("states")
    else:
        states_payload = payload
    if not isinstance(states_payload, list):
        raise ValueError(f"Invalid DFVR stateset payload in {path}: missing list of states.")
    return [deserialize_dfvr_state(state_items) for state_items in states_payload]


def save_dfvr_stateset(
    path: Union[str, Path],
    states: Sequence[Union[frozenset, Sequence[Sequence[object]]]],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    serial_states = []
    for state in states:
        if isinstance(state, frozenset):
            serial_states.append(serialize_dfvr_state(state))
        else:
            serial_states.append(serialize_dfvr_state(deserialize_dfvr_state(state)))
    out_payload: Dict[str, object] = {"states": serial_states}
    if metadata:
        out_payload["metadata"] = dict(metadata)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_dfvr_bound(
    *,
    Phi_star: Optional[Mapping[frozenset, float]],
    W_star: Mapping,
    belief: Dict,
    theta_star: Union[float, Sequence[float], Mapping],
    individuals: Sequence[str],
    gen_states: Sequence[int],
    infer,
    config,
    theta_mode: Optional[str] = None,
    tuple_mode: bool = False,
    genes: Optional[Sequence[str]] = None,
    state_mode: str = "belief",
    max_states: Optional[int] = None,
    random_seed: int = 0,
    rho_fn: Optional[Callable[[frozenset], float]] = None,
    successor_selector: Optional[Callable[[Dict], Dict]] = None,
    renormalize: bool = True,
    filter_infeasible_states: Optional[bool] = None,
    feasibility_eps: float = 1e-12,
    top_k: int = 0,
    max_outcomes_per_action: int = 10,
    aaub_star: Optional[Mapping[str, object]] = None,
    W_edge_star: Optional[Mapping] = None,
    pedigree_edges: Optional[Sequence] = None,
    W_trio_star: Optional[Mapping] = None,
    pedigree_trios: Optional[Sequence] = None,
    myopic_adp_star: Optional[Mapping] = None,
    oracle_adp_star: Optional[Mapping] = None,
    regime_residual_star: Optional[Mapping] = None,
    no_mutation: bool = False,
    fixed_states: Optional[Sequence[Union[frozenset, Sequence[Sequence[object]]]]] = None,
    enforce_fixed_states: bool = False,
    collect_state_list: bool = False,
) -> Dict[str, object]:
    """
    Compute the DF&VR residual bound for a candidate value function.

    This implementation targets Gap1-style overestimation (Φ̂ - Φ*), so the
    residual norm is computed as the Bellman *slack*:
        slack(s) = Φ̂(s) - (TΦ̂)(s),
    scaled by 1/rho(s), where (TΦ̂)(s) = max_a { r(s,a) + E[Φ̂(s')] }.

    Returns a dict with dfvr_bound, beta_rho, residual_norm, coverage, and argmax info.
    """
    gene_list = canonical_gene_order(genes) if genes else tuple()
    multi_gene = bool(gene_list)
    tuple_mode = bool(tuple_mode and multi_gene)

    a_gene = getattr(config, "a_gene", None)
    b_gene = getattr(config, "b_gene", None)
    c_gene = getattr(config, "c_gene", None)
    delta_gene = getattr(config, "delta_gene", None)

    theta_mode_resolved = (theta_mode or os.getenv("THETA_MODE", "scalar")).strip().lower()
    if theta_mode_resolved not in {"scalar", "stage", "person", "person_stage", "stage_gene"}:
        raise ValueError(
            "Unknown theta mode for DFVR="
            f"{theta_mode_resolved!r} (expected 'scalar', 'stage', 'person', 'person_stage', or 'stage_gene')."
        )
    if theta_mode_resolved == "stage_gene":
        per_gene_phi_env = os.getenv("ENABLE_PER_GENE_PHI", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
        if not multi_gene:
            raise ValueError("THETA_MODE='stage_gene' requires multi-gene config (genes must be configured).")
        if not tuple_mode:
            raise ValueError("THETA_MODE='stage_gene' requires tuple row generation (ENABLE_TUPLE_ROWGEN=1).")
        if not per_gene_phi_env:
            raise ValueError("THETA_MODE='stage_gene' requires ENABLE_PER_GENE_PHI=1.")

    def _evidence(state):
        if not isinstance(state, frozenset):
            raise AssertionError(
                f"State must be evidence-only frozenset[(person,outcome)], got {type(state).__name__}: {state!r}"
            )
        return state

    if rho_fn is None:
        def rho_fn(state) -> float:
            return 1.0 + (len(individuals) - len(_evidence(state)))

    rng = random.Random(random_seed)
    bn_model = getattr(infer, "model", None)
    if filter_infeasible_states is None:
        filter_infeasible_states = state_mode == "exhaustive"

    tuple_pmfs_by_state: Dict[frozenset, Dict[str, Dict[Tuple[int, ...], float]]] = {}
    local_belief: Dict[frozenset, Tuple[InferenceResult, Dict[str, Dict[int, float]]]] = {}
    value_cache: Dict[frozenset, float] = {}
    feasible_cache: Dict[frozenset, bool] = {}
    top_residuals_heap: list[tuple[float, int, dict]] = []
    top_beta_heap: list[tuple[float, int, dict]] = []
    heap_counter = itertools.count()
    belief_size_before = len(belief)

    def _state_key(state: frozenset) -> Tuple[Tuple[object, object], ...]:
        canonical_pairs = []
        for person, outcome in sorted(_evidence(state), key=lambda pair: pair[0]):
            if isinstance(outcome, list):
                outcome = tuple(outcome)
            canonical_pairs.append((person, outcome))
        return tuple(canonical_pairs)

    fixed_state_list = None
    fixed_state_keys = None
    if fixed_states is not None:
        fixed_state_list = []
        fixed_state_keys = set()
        for raw_state in fixed_states:
            if isinstance(raw_state, frozenset):
                state = raw_state
            else:
                state = deserialize_dfvr_state(raw_state)
            state = frozenset(_evidence(state))
            key = _state_key(state)
            if key in fixed_state_keys:
                continue
            fixed_state_list.append(state)
            fixed_state_keys.add(key)

    def _state_to_bn_evidence(state) -> Dict[str, int]:
        evidence = dict(_evidence(state))
        bn_evidence: Dict[str, int] = {}
        if multi_gene:
            for person, outcome in evidence.items():
                if isinstance(outcome, list):
                    outcome = tuple(outcome)
                outcome_tuple = outcome if isinstance(outcome, tuple) else None
                for idx, gene in enumerate(gene_list):
                    value = outcome_tuple[idx] if outcome_tuple is not None else outcome
                    bn_evidence[genotype_node_name(person, gene)] = int(value)
            return bn_evidence

        for person, outcome in evidence.items():
            if isinstance(outcome, list):
                outcome = tuple(outcome)
            if isinstance(outcome, tuple) and len(outcome) == 1:
                outcome = outcome[0]
            bn_evidence[person] = int(outcome)
        return bn_evidence

    def _cpd_probability(node: str, value: int, bn_evidence: Dict[str, int]) -> float:
        if bn_model is None:
            return 1.0
        cpd = bn_model.get_cpds(node)
        if cpd is None:
            return 1.0
        get_evidence = getattr(cpd, "get_evidence", None)
        evidence_vars = list(get_evidence() or []) if callable(get_evidence) else list(getattr(cpd, "evidence", []) or [])

        if not evidence_vars:
            get_value = getattr(cpd, "get_value", None)
            if callable(get_value):
                try:
                    return float(get_value(**{node: int(value)}))
                except Exception:
                    pass
            values = cpd.values.reshape(cpd.variable_card, -1)
            return float(values[int(value), 0])

        if any(parent not in bn_evidence for parent in evidence_vars):
            return 1.0

        assign = {node: int(value)}
        for parent in evidence_vars:
            assign[parent] = int(bn_evidence[parent])

        get_value = getattr(cpd, "get_value", None)
        if callable(get_value):
            return float(get_value(**assign))

        values = cpd.values.reshape(cpd.variable_card, -1)
        col = 0
        for parent in evidence_vars:
            col = col * 3 + int(bn_evidence[parent])
        return float(values[int(value), col])

    def _is_state_feasible(state) -> bool:
        if not filter_infeasible_states:
            return True
        if bn_model is None:
            return True
        cached = feasible_cache.get(state)
        if cached is not None:
            return cached
        bn_evidence = _state_to_bn_evidence(state)
        feasible = True
        for node, value in bn_evidence.items():
            if _cpd_probability(node, value, bn_evidence) < feasibility_eps:
                feasible = False
                break
        feasible_cache[state] = feasible
        return feasible

    def _state_to_items(state) -> list[tuple]:
        return sorted(_evidence(state), key=lambda pair: pair[0])

    def _state_payload(state) -> dict:
        return {"state": _state_to_items(state)}

    def _maybe_push_top(heap: list[tuple[float, int, dict]], score: float, payload: dict) -> None:
        if top_k <= 0:
            return
        entry = (score, next(heap_counter), payload)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
            return
        if score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def _normalize_outcome(outcome):
        if not tuple_mode:
            if isinstance(outcome, tuple) and len(outcome) == 1:
                return outcome[0]
            return outcome
        if isinstance(outcome, list):
            outcome = tuple(outcome)
        if isinstance(outcome, tuple):
            if len(outcome) == len(gene_list):
                return tuple(outcome)
            if len(outcome) == 1:
                return tuple(outcome[0] for _ in gene_list)
            return tuple(outcome[: len(gene_list)])
        return tuple(outcome for _ in gene_list)

    def _merge_state(state, person, outcome):
        evidence = _evidence(state)
        state_dict = dict(evidence)
        state_dict[person] = _normalize_outcome(outcome)
        return frozenset(state_dict.items())

    def _ensure_state(state):
        if state in local_belief:
            posterior_entry, z_post = local_belief[state]
            if tuple_mode and isinstance(posterior_entry, InferenceResult) and posterior_entry.has_tuple_pmfs():
                tuple_pmfs_by_state[state] = posterior_entry.get_tuple_pmfs()
            return posterior_entry, z_post

        if state in belief:
            posterior_entry, z_post = belief[state]
            if tuple_mode:
                if isinstance(posterior_entry, InferenceResult) and posterior_entry.has_tuple_pmfs():
                    tuple_pmfs_by_state[state] = posterior_entry.get_tuple_pmfs()
                    return posterior_entry, z_post
                # Recompute to get tuple PMFs.
                evidence = dict(_evidence(state))
                result = propagate_multigene_marginals(
                    infer, individuals, gen_states, evidence, gene_list, aggregate_only=False
                )
                posterior_entry = result
                z_post = {j: {g: 1.0 if evidence.get(j) == g else 0.0 for g in gen_states} for j in individuals}
                if no_mutation:
                    local_belief[state] = (posterior_entry, z_post)
                else:
                    belief[state] = (posterior_entry, z_post)
                if posterior_entry.has_tuple_pmfs():
                    tuple_pmfs_by_state[state] = posterior_entry.get_tuple_pmfs()
                return posterior_entry, z_post
            return posterior_entry, z_post

        evidence = dict(_evidence(state))
        if tuple_mode:
            result = propagate_multigene_marginals(
                infer, individuals, gen_states, evidence, gene_list, aggregate_only=False
            )
            posterior_entry = result
        elif multi_gene:
            result = propagate_multigene_marginals(
                infer, individuals, gen_states, evidence, gene_list, aggregate_only=True
            )
            posterior_entry = result
        else:
            if len(evidence) == len(individuals):
                marginals = {
                    person: {g: 1.0 if g == evidence[person] else 0.0 for g in gen_states}
                    for person in evidence
                }
                posterior_entry = InferenceResult(
                    marginals=marginals,
                    gene_order=("gene",),
                    gen_states=gen_states,
                )
            else:
                posterior_entry = propagate_all_marginals_safe(
                    infer, individuals, gen_states, evidence
                )

        if not isinstance(posterior_entry, InferenceResult):
            posterior_entry = InferenceResult(
                posterior_entry,
                gene_order=gene_list if multi_gene else ("gene",),
                gen_states=gen_states,
            )
        z_post = {j: {g: 1.0 if evidence.get(j) == g else 0.0 for g in gen_states} for j in individuals}
        if no_mutation:
            local_belief[state] = (posterior_entry, z_post)
        else:
            belief[state] = (posterior_entry, z_post)

        if tuple_mode and posterior_entry.has_tuple_pmfs():
            tuple_pmfs_by_state[state] = posterior_entry.get_tuple_pmfs()
        return posterior_entry, z_post

    def _get_per_gene_probs(posterior_entry):
        if not multi_gene:
            return None
        if isinstance(posterior_entry, InferenceResult):
            per_gene = posterior_entry.get_per_gene_probs()
            if per_gene:
                return per_gene
        if isinstance(posterior_entry, Mapping):
            return lift_single_gene_posteriors_to_genes(posterior_entry, gene_list)
        return None

    def _per_gene_p12_map(per_gene_probs, person):
        if not per_gene_probs:
            return None
        carrier = {}
        for gene, probs in per_gene_probs.items():
            if person not in probs:
                continue
            carrier_prob = probs[person].get(1, 0.0) + probs[person].get(2, 0.0)
            carrier[gene] = carrier_prob
        return carrier or None

    def _value_hat(state: frozenset) -> float:
        cached = value_cache.get(state)
        if cached is not None:
            return cached
        # Terminal boundary condition: once everyone is tested there is no future
        # reward, so V(s)=0 for all fully-tested states (independent of outcomes).
        if len(_evidence(state)) >= len(individuals):
            value_cache[state] = 0.0
            return 0.0
        _ensure_state(state)
        if Phi_star is not None and state in Phi_star:
            val = Phi_star[state]
            if val is not None and math.isfinite(val):
                value_cache[state] = val
                return val
        posterior_entry, z_post = _ensure_state(state)
        val = phi_hat(
            state,
            theta_star=theta_star,
            W_star=W_star,
            belief={state: (posterior_entry, z_post)} if no_mutation else belief,
            gen_states=gen_states,
            individuals=individuals,
            theta_mode=theta_mode_resolved,
            tuple_pmfs=tuple_pmfs_by_state if tuple_mode else None,
            tuple_mode=tuple_mode,
            aaub_star=aaub_star,
            W_edge_star=W_edge_star,
            pedigree_edges=pedigree_edges,
            W_trio_star=W_trio_star,
            pedigree_trios=pedigree_trios,
            infer=infer,
            genes=list(genes) if genes else None,
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
        )
        value_cache[state] = val
        return val

    truncated = False

    if state_mode not in {"belief", "exhaustive"}:
        raise ValueError(f"Unknown state_mode '{state_mode}' (expected 'belief' or 'exhaustive').")

    def _iter_states() -> Iterable:
        nonlocal truncated
        if fixed_state_list is not None:
            if max_states and len(fixed_state_list) > max_states:
                if enforce_fixed_states:
                    raise RuntimeError(
                        "DFVR fixed state-set exceeds max_states while enforce_fixed_states=1: "
                        f"fixed={len(fixed_state_list)} max_states={max_states}"
                    )
                truncated = True
                for state in fixed_state_list[:max_states]:
                    yield state
            else:
                for state in fixed_state_list:
                    yield state
            return

        if state_mode == "belief":
            states = list(belief.keys())
            if max_states and len(states) > max_states:
                truncated = True
                for s in rng.sample(states, max_states):
                    yield s
            else:
                for s in states:
                    yield s
            return

        if tuple_mode:
            outcomes = list(itertools.product(gen_states, repeat=len(gene_list)))
        else:
            outcomes = list(gen_states)
        indexer = StateIndexer(individuals, outcomes)
        count = 0
        for indexed in indexer.iter_indexed_states():
            yield indexer.materialize(indexed)
            count += 1
            if max_states and count >= max_states:
                truncated = True
                break

    coverage = {
        "state_mode": state_mode,
        "state_source": "fixed" if fixed_state_list is not None else state_mode,
        "num_states": 0,
        "num_infeasible_states": 0,
        "num_actions": 0,
        "num_test_actions": 0,
        "num_stop_actions": 0,
        "max_states": max_states,
        "truncated": False,
        "filter_infeasible_states": bool(filter_infeasible_states),
        "no_mutation": bool(no_mutation),
        "belief_size_before": belief_size_before,
        "belief_size_after": belief_size_before,
        "belief_mutated": False,
        "fixed_states_requested": len(fixed_state_list) if fixed_state_list is not None else None,
        "fixed_states_enforced": bool(fixed_state_list is not None and enforce_fixed_states),
        "fixed_states_match": None,
        "state_signature": None,
    }

    residual_norm = 0.0
    beta_rho = 0.0
    argmax_residual: Optional[dict] = None
    argmax_beta: Optional[dict] = None
    evaluated_state_keys = set()
    signature = hashlib.sha256()
    collected_state_items = [] if collect_state_list else None
    collected_state_keys = set() if collect_state_list else None

    for state in _iter_states():
        state_key = _state_key(state)
        evaluated_state_keys.add(state_key)
        state_items = _state_to_items(state)
        signature.update(json.dumps(state_items, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
        signature.update(b"\n")
        if collect_state_list and state_key not in collected_state_keys:
            collected_state_keys.add(state_key)
            collected_state_items.append(state_items)

        if not _is_state_feasible(state):
            coverage["num_infeasible_states"] += 1
            continue
        posterior_entry, _ = _ensure_state(state)
        p_s = posterior_entry.marginals if isinstance(posterior_entry, InferenceResult) else posterior_entry
        per_gene_probs = _get_per_gene_probs(posterior_entry)

        phi_s = _value_hat(state)
        rho_s = rho_fn(state)

        tested = dict(_evidence(state))
        untested = [i for i in individuals if i not in tested]

        stop_value = 0.0
        for k in untested:
            stop_value += r_reward(
                k,
                p_s,
                config.a,
                config.b,
                config.c,
                config.delta,
                per_gene_probs=per_gene_probs,
                a_gene=a_gene,
                b_gene=b_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )
        # Bellman optimality RHS at this state (reward maximization):
        #   (T Φ̂)(s) = max_a { r(s,a) + E[Φ̂(s')] }
        # DFVR residual bound is applied to the *cost-form* ALP (J=-Φ), so the
        # relevant nonnegative residual is the *slack*:
        #   (T_J J - J)(s) = Φ̂(s) - (T Φ̂)(s).
        best_rhs = stop_value
        best_action_payload = {
            "action": {"kind": "stop"},
            "immediate_reward": stop_value,
            "expected_phi_succ": 0.0,
            "sum_prob": 0.0,
        }
        coverage["num_actions"] += 1
        coverage["num_stop_actions"] += 1

        for person in untested:
            if tuple_mode:
                tuple_pmfs_state = tuple_pmfs_by_state.get(state, {})
                dist = tuple_pmfs_state.get(person, {})
            else:
                dist = p_s.get(person, {})

            if not dist:
                continue

            probs: Dict = {}
            phi_succ: Dict = {}
            total_prob = 0.0
            expected_phi_succ = 0.0
            for outcome, prob in dist.items():
                if prob <= 0.0:
                    continue
                norm_outcome = _normalize_outcome(outcome)
                probs[norm_outcome] = probs.get(norm_outcome, 0.0) + prob
                total_prob += prob
                if norm_outcome not in phi_succ:
                    succ_state = _merge_state(state, person, norm_outcome)
                    phi_succ[norm_outcome] = _value_hat(succ_state)
                expected_phi_succ += prob * phi_succ[norm_outcome]

            if successor_selector is not None:
                probs = successor_selector(dict(probs))
                if not probs:
                    continue
                phi_succ = {outcome: phi_succ[outcome] for outcome in probs if outcome in phi_succ}
                total_prob = sum(probs.values())
                expected_phi_succ = sum(probs[outcome] * phi_succ[outcome] for outcome in probs)

            if total_prob <= 0.0:
                continue
            if renormalize and abs(total_prob - 1.0) > 1e-6:
                for key in list(probs.keys()):
                    probs[key] = probs[key] / total_prob
                expected_phi_succ = sum(probs[outcome] * phi_succ[outcome] for outcome in probs)

            p12 = p_s.get(person, {}).get(1, 0.0) + p_s.get(person, {}).get(2, 0.0)
            per_gene_p12 = _per_gene_p12_map(per_gene_probs, person)
            immediate_reward = r_reward_testp(
                person,
                p12,
                config.a,
                config.b,
                config.c,
                config.delta,
                config.fixed_cost,
                config.variable_cost,
                per_gene_p12=per_gene_p12,
                a_gene=a_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )

            violation, rhs = bellman_violation(
                phi_S=phi_s,
                phi_succ=phi_succ,
                probs=probs,
                r_immediate=immediate_reward,
            )
            if rhs > best_rhs:
                best_rhs = rhs
                best_action_payload = {
                    "action": {"kind": "test", "person": person},
                    "immediate_reward": immediate_reward,
                    "expected_phi_succ": expected_phi_succ,
                    "sum_prob": float(sum(probs.values())),
                }

            beta_candidate = sum(
                prob * rho_fn(_merge_state(state, person, outcome)) for outcome, prob in probs.items()
            ) / rho_s
            if beta_candidate > beta_rho:
                beta_rho = beta_candidate
                argmax_beta = {
                    "beta_candidate": beta_candidate,
                    "rho": rho_s,
                    **_state_payload(state),
                    "action": {"kind": "test", "person": person},
                    "sum_prob": float(sum(probs.values())),
                }
            _maybe_push_top(
                top_beta_heap,
                beta_candidate,
                {
                    "beta_candidate": beta_candidate,
                    "rho": rho_s,
                    **_state_payload(state),
                    "action": {"kind": "test", "person": person},
                    "sum_prob": float(sum(probs.values())),
                },
            )

            coverage["num_actions"] += 1
            coverage["num_test_actions"] += 1

        slack = phi_s - best_rhs
        if slack > 0:
            scaled = slack / rho_s
            payload = {
                "scaled": scaled,
                "residual": slack,
                "rho": rho_s,
                **_state_payload(state),
                "phi_s": phi_s,
                "bellman_rhs": best_rhs,
                **best_action_payload,
            }
            if scaled > residual_norm:
                residual_norm = scaled
                argmax_residual = payload

            if top_k > 0:
                # Enrich logged payload with a compact decomposition of the
                # expected successor value for the argmax RHS action.
                action = best_action_payload.get("action") or {}
                if (
                    action.get("kind") == "test"
                    and max_outcomes_per_action > 0
                    and isinstance(action.get("person"), str)
                ):
                    person = action["person"]
                    dist = None
                    if tuple_mode:
                        tuple_pmfs_state = tuple_pmfs_by_state.get(state, {})
                        dist = tuple_pmfs_state.get(person, {})
                    else:
                        dist = p_s.get(person, {})
                    if dist:
                        contrib_items = []
                        for out, pr in dist.items():
                            if pr <= 0.0:
                                continue
                            norm_out = _normalize_outcome(out)
                            succ_state = _merge_state(state, person, norm_out)
                            phi_val = _value_hat(succ_state)
                            contrib_items.append(
                                {
                                    "outcome": norm_out,
                                    "prob": pr,
                                    "phi_succ": phi_val,
                                    "prob_phi": pr * phi_val,
                                }
                            )
                        contrib_items.sort(key=lambda item: abs(item["prob_phi"]), reverse=True)
                        payload = dict(payload)
                        payload["outcome_contribs"] = contrib_items[:max_outcomes_per_action]
                _maybe_push_top(top_residuals_heap, scaled, payload)

        coverage["num_states"] += 1

    coverage["truncated"] = truncated
    coverage["belief_size_after"] = len(belief)
    coverage["belief_mutated"] = bool(len(belief) != belief_size_before)
    coverage["state_signature"] = signature.hexdigest()

    fixed_integrity = None
    if fixed_state_keys is not None:
        missing = fixed_state_keys - evaluated_state_keys
        extra = evaluated_state_keys - fixed_state_keys
        fixed_match = not missing and not extra
        fixed_integrity = {
            "enabled": True,
            "requested": len(fixed_state_keys),
            "evaluated": len(evaluated_state_keys),
            "missing": len(missing),
            "extra": len(extra),
            "match": fixed_match,
        }
        coverage["fixed_states_match"] = fixed_match
        if enforce_fixed_states and not fixed_match:
            raise RuntimeError(
                "DFVR fixed state-set integrity failure: "
                f"requested={len(fixed_state_keys)} evaluated={len(evaluated_state_keys)} "
                f"missing={len(missing)} extra={len(extra)}"
            )
    elif enforce_fixed_states and fixed_states is not None:
        coverage["fixed_states_match"] = False
    else:
        coverage["fixed_states_match"] = None

    if no_mutation and coverage["belief_mutated"]:
        raise RuntimeError(
            "DFVR no-mutation violation: belief map changed during evaluation "
            f"(before={belief_size_before}, after={len(belief)})."
        )

    if beta_rho >= 1.0:
        dfvr_bound = math.inf
    else:
        dfvr_bound = (2.0 / (1.0 - beta_rho)) * residual_norm

    top_residuals = None
    top_betas = None
    if top_k > 0:
        top_residuals = [item[2] for item in sorted(top_residuals_heap, key=lambda t: t[0], reverse=True)]
        top_betas = [item[2] for item in sorted(top_beta_heap, key=lambda t: t[0], reverse=True)]

    payload = {
        "dfvr_bound": dfvr_bound,
        "beta_rho": beta_rho,
        "residual_norm": residual_norm,
        "coverage": coverage,
        "argmax_residual": argmax_residual,
        "argmax_beta": argmax_beta,
        "top_residuals": top_residuals,
        "top_betas": top_betas,
        "fixed_state_integrity": fixed_integrity,
    }
    if collect_state_list:
        payload["state_list"] = collected_state_items
    return payload
