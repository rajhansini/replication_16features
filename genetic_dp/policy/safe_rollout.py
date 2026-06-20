from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evaluator import compare_action_against_incumbent, compare_policies_from_root
from .extractor import q_test, stop_value


@dataclass(frozen=True)
class RolloutCandidateSummary:
    action: Any
    q_estimate: float
    exact_value: float
    delta_vs_incumbent: float
    incumbent_safe: bool
    is_incumbent: bool


@dataclass(frozen=True)
class SafeRolloutSelection:
    state: frozenset
    selected_action: Any
    incumbent_action: Any
    selected_exact_value: float
    incumbent_exact_value: float
    epsilon: float
    top_k: int
    candidates: tuple[RolloutCandidateSummary, ...]
    incumbent_safe: bool


def _action_signature(action: Any) -> Any:
    if isinstance(action, tuple) and len(action) >= 2:
        return (action[0], action[1])
    return action


def _normalize_action(action: Any, *, fallback_value: float = 0.0) -> Any:
    if isinstance(action, tuple):
        if len(action) >= 3:
            return action
        if len(action) == 2:
            return (action[0], action[1], float(fallback_value))
    return action


def rank_top_k_rollout_candidates(
    candidate_action_scores: Mapping[Any, float],
    incumbent_action: Any,
    *,
    top_k: int,
) -> tuple[Any, ...]:
    """Return the incumbent action first, then the top-K unique alternatives."""

    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    incumbent_action = _normalize_action(incumbent_action)
    ranked = sorted(
        candidate_action_scores.items(),
        key=lambda item: (-float(item[1]), repr(item[0])),
    )
    selected = [incumbent_action]
    seen = {_action_signature(incumbent_action)}
    for action, score in ranked:
        normalized_action = _normalize_action(action, fallback_value=float(score))
        signature = _action_signature(normalized_action)
        if signature in seen:
            continue
        selected.append(normalized_action)
        seen.add(signature)
        if len(selected) - 1 >= top_k:
            break
    return tuple(selected)


def select_incumbent_safe_rollout_action(
    state: frozenset,
    *,
    candidate_action_scores: Mapping[Any, float],
    incumbent_policy: Mapping[frozenset, Any],
    exact_eval_kwargs: Mapping[str, Any],
    top_k: int = 3,
    epsilon: float = 0.0,
    incumbent_action: Any = None,
) -> SafeRolloutSelection:
    """Select the best exact rollout action among top-K scored candidates."""

    if state not in incumbent_policy and incumbent_action is None:
        raise KeyError(f"Incumbent policy does not define state {state!r}.")

    resolved_incumbent_action = incumbent_action
    if resolved_incumbent_action is None:
        resolved_incumbent_action = incumbent_policy[state]
    resolved_incumbent_action = _normalize_action(resolved_incumbent_action)

    ranked_actions = rank_top_k_rollout_candidates(
        candidate_action_scores,
        resolved_incumbent_action,
        top_k=top_k,
    )
    q_by_signature = {
        _action_signature(_normalize_action(action, fallback_value=float(score))): float(score)
        for action, score in candidate_action_scores.items()
    }

    evaluated: list[RolloutCandidateSummary] = []
    incumbent_value = None
    selected_action = resolved_incumbent_action
    selected_exact_value = None

    strict_eval_kwargs = dict(exact_eval_kwargs)
    strict_eval_kwargs.setdefault("strict_mode", True)

    for action in ranked_actions:
        is_incumbent = _action_signature(action) == _action_signature(resolved_incumbent_action)
        comparison = compare_action_against_incumbent(
            state,
            action,
            incumbent_policy,
            incumbent_action=resolved_incumbent_action,
            epsilon=epsilon,
            **strict_eval_kwargs,
        )
        if is_incumbent:
            incumbent_value = float(comparison.incumbent.value)

        summary = RolloutCandidateSummary(
            action=action,
            q_estimate=float(q_by_signature.get(_action_signature(action), float("nan"))),
            exact_value=float(comparison.candidate.value),
            delta_vs_incumbent=float(comparison.delta),
            incumbent_safe=bool(comparison.incumbent_safe),
            is_incumbent=is_incumbent,
        )
        evaluated.append(summary)

        if selected_exact_value is None or summary.exact_value > selected_exact_value + epsilon:
            selected_action = action
            selected_exact_value = summary.exact_value

    if incumbent_value is None:
        raise KeyError("Incumbent action was not evaluated; missing incumbent candidate summary.")
    if selected_exact_value is None:
        selected_exact_value = incumbent_value

    incumbent_safe = selected_exact_value >= incumbent_value - epsilon
    if not incumbent_safe:
        selected_action = resolved_incumbent_action
        selected_exact_value = incumbent_value
        incumbent_safe = True

    return SafeRolloutSelection(
        state=state,
        selected_action=selected_action,
        incumbent_action=resolved_incumbent_action,
        selected_exact_value=float(selected_exact_value),
        incumbent_exact_value=float(incumbent_value),
        epsilon=float(epsilon),
        top_k=int(top_k),
        candidates=tuple(evaluated),
        incumbent_safe=bool(incumbent_safe),
    )


def _build_candidate_action_scores(
    state: frozenset,
    *,
    belief,
    theta_star,
    W_star,
    theta_mode,
    pedigree,
    theta_model,
    theta_model_spec,
    individuals,
    gen_states,
    r_reward_testp,
    a,
    b,
    c,
    delta,
    infer,
    fixed_cost,
    variable_cost,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    tuple_pmfs=None,
    tuple_mode=False,
    phi_values=None,
    aaub_star=None,
    W_edge_star=None,
    pedigree_edges=None,
    W_trio_star=None,
    pedigree_trios=None,
    feature_cache=None,
) -> dict[Any, float]:
    scores: dict[Any, float] = {}
    stop_q = float(
        stop_value(
            state,
            belief=belief,
            a=a,
            b=b,
            c=c,
            delta=delta,
            individuals=individuals,
            belief_gene=belief_gene,
            genes=genes,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
        )
    )
    scores[("stop", None, stop_q)] = stop_q

    tested = dict(state)
    for person in individuals:
        if person in tested:
            continue
        q_value = float(
            q_test(
                state,
                person,
                theta_star=theta_star,
                W_star=W_star,
                belief=belief,
                theta_mode=theta_mode,
                pedigree=pedigree,
                theta_model=theta_model,
                theta_model_spec=theta_model_spec,
                gen_states=gen_states,
                r_reward_testp=r_reward_testp,
                a=a,
                b=b,
                c=c,
                delta=delta,
                individuals=individuals,
                infer=infer,
                fixed_cost=fixed_cost,
                variable_cost=variable_cost,
                belief_gene=belief_gene,
                genes=genes,
                a_gene=a_gene,
                b_gene=b_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
                tuple_pmfs=tuple_pmfs,
                tuple_mode=tuple_mode,
                phi_values=phi_values,
                aaub_star=aaub_star,
                W_edge_star=W_edge_star,
                pedigree_edges=pedigree_edges,
                W_trio_star=W_trio_star,
                pedigree_trios=pedigree_trios,
                feature_cache=feature_cache,
            )
        )
        scores[("test", person, q_value)] = q_value
    return scores


def evaluate_safe_rollout(context: Mapping[str, Any]) -> dict[str, Any]:
    """Build and evaluate a one-step incumbent-safe rollout policy."""

    incumbent_policy = context.get("incumbent_policy_map")
    if not isinstance(incumbent_policy, Mapping) or not incumbent_policy:
        return {
            "decision": "fallback_to_adp_policy",
            "reason": "missing_incumbent_policy",
            "production_policy_source": "adp_policy",
        }

    candidate_policy = context.get("candidate_policy_map")
    states: list[frozenset] = []
    for policy_map in (incumbent_policy, candidate_policy):
        if not isinstance(policy_map, Mapping):
            continue
        for state in policy_map.keys():
            if state not in states:
                states.append(state)

    config = context["config"]
    belief = context["belief"]
    belief_gene = context.get("belief_gene")
    tuple_pmfs = context.get("tuple_pmfs")
    tuple_mode = bool(tuple_pmfs)
    top_k = int(context.get("top_k", 1))
    epsilon = float(context.get("epsilon", 0.0))

    exact_eval_kwargs = {
        "belief": belief,
        "individuals": context["individuals"],
        "gen_states": context["gen_states"],
        "r_reward_test": context["r_reward_test"],
        "a": config.a,
        "b": config.b,
        "c": config.c,
        "delta": config.delta,
        "infer": context["infer"],
        "theta_star": context["theta_star"],
        "W_star": context["W_star"],
        "theta_mode": context.get("theta_mode"),
        "pedigree": context.get("pedigree"),
        "theta_model": context.get("theta_model"),
        "theta_model_spec": context.get("theta_model_spec"),
        "fixed_cost": config.fixed_cost,
        "variable_cost": config.variable_cost,
        "lookahead_depth": 0,
        "belief_gene": belief_gene,
        "genes": config.genes if getattr(config, "genes", None) else None,
        "a_gene": config.a_gene if getattr(config, "a_gene", None) else None,
        "b_gene": config.b_gene if getattr(config, "b_gene", None) else None,
        "c_gene": config.c_gene if getattr(config, "c_gene", None) else None,
        "delta_gene": config.delta_gene if getattr(config, "delta_gene", None) else None,
        "tuple_pmfs": tuple_pmfs if tuple_mode else None,
        "aaub_star": context.get("aaub_star"),
        "W_edge_star": context.get("W_edge_star"),
        "pedigree_edges": context.get("pedigree_edges"),
        "W_trio_star": context.get("W_trio_star"),
        "pedigree_trios": context.get("pedigree_trios"),
        "feature_cache": context.get("feature_cache"),
        "strict_mode": True,
    }

    rollout_policy: dict[frozenset, Any] = {}
    state_diagnostics: dict[str, dict[str, Any]] = {}
    disagreement_count = 0
    fallback_count = 0

    for state in states:
        incumbent_action = incumbent_policy.get(state)
        if incumbent_action is None:
            continue
        incumbent_action = _normalize_action(incumbent_action)
        try:
            candidate_scores = _build_candidate_action_scores(
                state,
                belief=belief,
                theta_star=context["theta_star"],
                W_star=context["W_star"],
                theta_mode=context.get("theta_mode"),
                pedigree=context.get("pedigree"),
                theta_model=context.get("theta_model"),
                theta_model_spec=context.get("theta_model_spec"),
                individuals=context["individuals"],
                gen_states=context["gen_states"],
                r_reward_testp=context["r_reward_testp"],
                a=config.a,
                b=config.b,
                c=config.c,
                delta=config.delta,
                infer=context["infer"],
                fixed_cost=config.fixed_cost,
                variable_cost=config.variable_cost,
                belief_gene=belief_gene,
                genes=config.genes if getattr(config, "genes", None) else None,
                a_gene=config.a_gene if getattr(config, "a_gene", None) else None,
                b_gene=config.b_gene if getattr(config, "b_gene", None) else None,
                c_gene=config.c_gene if getattr(config, "c_gene", None) else None,
                delta_gene=config.delta_gene if getattr(config, "delta_gene", None) else None,
                tuple_pmfs=tuple_pmfs if tuple_mode else None,
                tuple_mode=tuple_mode,
                phi_values=context.get("phi_values"),
                aaub_star=context.get("aaub_star"),
                W_edge_star=context.get("W_edge_star"),
                pedigree_edges=context.get("pedigree_edges"),
                W_trio_star=context.get("W_trio_star"),
                pedigree_trios=context.get("pedigree_trios"),
                feature_cache=context.get("feature_cache"),
            )
            selection = select_incumbent_safe_rollout_action(
                state,
                candidate_action_scores=candidate_scores,
                incumbent_policy=incumbent_policy,
                exact_eval_kwargs=exact_eval_kwargs,
                top_k=top_k,
                epsilon=epsilon,
                incumbent_action=incumbent_action,
            )
            rollout_policy[state] = selection.selected_action
            if _action_signature(selection.selected_action) != _action_signature(incumbent_action):
                disagreement_count += 1
            state_diagnostics[repr(state)] = {
                "incumbent_action": selection.incumbent_action,
                "selected_action": selection.selected_action,
                "selected_exact_value": selection.selected_exact_value,
                "incumbent_exact_value": selection.incumbent_exact_value,
                "candidate_count": len(selection.candidates),
            }
        except Exception as exc:
            rollout_policy[state] = incumbent_action
            fallback_count += 1
            state_diagnostics[repr(state)] = {
                "incumbent_action": incumbent_action,
                "selected_action": incumbent_action,
                "error": repr(exc),
            }

    comparison = compare_policies_from_root(
        rollout_policy,
        incumbent_policy,
        epsilon=epsilon,
        candidate_fallback_policy=incumbent_policy,
        incumbent_fallback_policy=incumbent_policy,
        **exact_eval_kwargs,
    )
    enforce_incumbent_safe = bool(context.get("incumbent_safe", True))
    use_rollout = comparison.incumbent_safe or not enforce_incumbent_safe
    production_policy = rollout_policy if use_rollout else dict(incumbent_policy)
    production_value = comparison.candidate.value if use_rollout else comparison.incumbent.value

    return {
        "decision": "accepted" if use_rollout else "fallback_to_incumbent",
        "reason": (
            "candidate_improves_or_matches_incumbent"
            if comparison.incumbent_safe
            else "candidate_worse_than_incumbent"
        ),
        "selected_candidate_id": context.get("selected_candidate_id"),
        "production_policy_source": "safe_rollout" if use_rollout else "incumbent_policy",
        "production_policy_value": float(production_value),
        "policy_value": float(production_value),
        "candidate_policy_value": float(comparison.candidate.value),
        "incumbent_policy_value": float(comparison.incumbent.value),
        "exact_root_value": context.get("exact_root_value"),
        "incumbent_safe": bool(comparison.incumbent_safe),
        "top_k": top_k,
        "state_count": len(states),
        "disagreement_count": disagreement_count,
        "fallback_count": fallback_count,
        "production_policy_map": production_policy,
        "state_diagnostics": state_diagnostics,
    }
