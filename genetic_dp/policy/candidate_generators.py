from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from genetic_dp.policy_phase_diagram import action_to_string, actions_equal, finite_float


VALUE_TOL = 1e-10
STOP_ACTION = "STOP"


@dataclass(frozen=True)
class CandidateGeneratorContext:
    legal_actions: Sequence[Any]
    myopic_q_scores: Mapping[str, float] | None = None
    person_gene_probs: Mapping[str, Mapping[str, Mapping[int, float]]] | None = None
    reward_weights: Mapping[str, float] | None = None
    bridge_depth_scores: Mapping[str, float] | None = None
    frontier_scores: Mapping[str, float] | None = None
    carrier_mass_mode: str = "reward_weighted"
    bridge_depth_mode: str = "proxy_honest"
    frontier_mode: str = "all_untested_carrier_variance"


def normalize_action(action: Any) -> str:
    token = action_to_string(action)
    if token is None:
        raise ValueError(f"Cannot normalize action: {action!r}")
    return token


def is_stop(action: Any) -> bool:
    return normalize_action(action) == STOP_ACTION


def is_test_action(action: Any) -> bool:
    return normalize_action(action).startswith("TEST(")


def action_person(action: Any) -> str | None:
    token = normalize_action(action)
    if token.startswith("TEST(") and token.endswith(")"):
        return token[5:-1]
    return None


def dedupe_action_strings(actions: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for action in actions:
        token = normalize_action(action)
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _context_from_mapping(context: Mapping[str, Any] | CandidateGeneratorContext) -> CandidateGeneratorContext:
    if isinstance(context, CandidateGeneratorContext):
        return context
    return CandidateGeneratorContext(
        legal_actions=list(context.get("legal_actions") or []),
        myopic_q_scores=context.get("myopic_q_scores"),
        person_gene_probs=context.get("person_gene_probs"),
        reward_weights=context.get("reward_weights"),
        bridge_depth_scores=context.get("bridge_depth_scores"),
        frontier_scores=context.get("frontier_scores"),
        carrier_mass_mode=str(context.get("carrier_mass_mode") or "reward_weighted"),
        bridge_depth_mode=str(context.get("bridge_depth_mode") or "proxy_honest"),
        frontier_mode=str(context.get("frontier_mode") or "all_untested_carrier_variance"),
    )


def _legal_action_strings(context: CandidateGeneratorContext) -> list[str]:
    return dedupe_action_strings(list(context.legal_actions))


def legal_test_actions(context: CandidateGeneratorContext) -> list[str]:
    return [action for action in _legal_action_strings(context) if is_test_action(action)]


def _ranked_by_score(scores: Mapping[str, float], legal_tests: Sequence[str], k: int) -> list[str]:
    indexed = {action: index for index, action in enumerate(legal_tests)}
    ranked = sorted(
        legal_tests,
        key=lambda action: (-(finite_float(scores.get(action)) or float("-inf")), indexed[action], action),
    )
    return ranked[: max(0, int(k))]


def _payload_actions(grr_payload: Mapping[str, Any] | None, generator_id: str, k: int) -> list[str]:
    payload = dict(grr_payload or {})
    prefix = "grr2d" if generator_id == "GRR2D_TOPK" else "grr2"
    score_keys = (
        f"{prefix}_root_action_scores",
        f"{generator_id}_root_action_scores",
        "root_action_scores",
    )
    for key in score_keys:
        records = payload.get(key)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        actions = []
        for record in records:
            if isinstance(record, Mapping) and record.get("action") is not None:
                actions.append(record.get("action"))
        if actions:
            return dedupe_action_strings(actions)[: max(0, int(k))]
    action_keys = (f"{prefix}_topk_actions", f"{generator_id}_topk_actions", "topk_actions")
    for key in action_keys:
        actions = payload.get(key)
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            return dedupe_action_strings(actions)[: max(0, int(k))]
    return []


def _carrier_probability(state_probs: Mapping[int, float]) -> float:
    p0 = finite_float(state_probs.get(0)) or 0.0
    return max(0.0, min(1.0, 1.0 - p0))


def _person_positive_probs(context: CandidateGeneratorContext, person: str) -> dict[str, float]:
    result: dict[str, float] = {}
    per_gene = context.person_gene_probs or {}
    for gene, per_person in per_gene.items():
        state_probs = per_person.get(person, {}) if isinstance(per_person, Mapping) else {}
        if isinstance(state_probs, Mapping):
            result[str(gene)] = _carrier_probability(state_probs)
    return result


def _reward_weight(context: CandidateGeneratorContext, gene: str) -> float:
    value = finite_float((context.reward_weights or {}).get(gene))
    return 1.0 if value is None or value <= 0.0 else value


def carrier_mass_score(context: CandidateGeneratorContext, person: str) -> float:
    probs = _person_positive_probs(context, person)
    if not probs:
        return 0.0
    if context.carrier_mass_mode == "any_positive":
        not_positive = 1.0
        for prob in probs.values():
            not_positive *= 1.0 - prob
        return float(1.0 - not_positive)
    return float(sum(_reward_weight(context, gene) * prob for gene, prob in probs.items()))


def uncertainty_score(context: CandidateGeneratorContext, person: str) -> float:
    probs = _person_positive_probs(context, person)
    if not probs:
        return 0.0
    return float(
        sum(_reward_weight(context, gene) * prob * (1.0 - prob) for gene, prob in probs.items())
    )


def _person_score_map(context: CandidateGeneratorContext, scorer: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for action in legal_test_actions(context):
        person = action_person(action)
        if person is None:
            continue
        if scorer == "carrier":
            scores[action] = carrier_mass_score(context, person)
        elif scorer == "uncertainty":
            scores[action] = uncertainty_score(context, person)
        elif scorer == "bridge_depth":
            topo = finite_float((context.bridge_depth_scores or {}).get(person))
            scores[action] = uncertainty_score(context, person) * (topo if topo is not None else 1.0)
        elif scorer == "frontier":
            direct = finite_float((context.frontier_scores or {}).get(person))
            scores[action] = direct if direct is not None else uncertainty_score(context, person)
    return scores


def generate_candidate_actions(
    generator_id: str,
    state: Any = None,
    model_context: Mapping[str, Any] | CandidateGeneratorContext | None = None,
    k: int | str = 2,
    *,
    seed: int | None = None,
    grr_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return generator-ranked legal action strings in deterministic order.

    `state` is accepted for API compatibility with callers that generate scores
    from evidence states; this function only needs the prepared `model_context`.
    """

    del state
    context = _context_from_mapping(model_context or {})
    legal_tests = legal_test_actions(context)
    generator_id = str(generator_id).upper()
    if generator_id in {"GRR2_TOPK", "GRR2D_TOPK"}:
        payload_actions = _payload_actions(grr_payload, generator_id, int(k))
        legal = set(_legal_action_strings(context))
        return [action for action in payload_actions if action in legal]
    if generator_id == "MYOPIC_Q_TOPK":
        return _ranked_by_score(context.myopic_q_scores or {}, legal_tests, int(k))
    if generator_id == "CARRIER_MASS_TOPK":
        return _ranked_by_score(_person_score_map(context, "carrier"), legal_tests, int(k))
    if generator_id == "UNCERTAINTY_TOPK":
        return _ranked_by_score(_person_score_map(context, "uncertainty"), legal_tests, int(k))
    if generator_id == "BRIDGE_DEPTH_TOPK":
        return _ranked_by_score(_person_score_map(context, "bridge_depth"), legal_tests, int(k))
    if generator_id == "FRONTIER_VARIANCE_OR_ENTROPY_TOPK":
        return _ranked_by_score(_person_score_map(context, "frontier"), legal_tests, int(k))
    if generator_id == "RANDOM_TOPK":
        rng = random.Random(seed)
        actions = list(legal_tests)
        rng.shuffle(actions)
        return actions[: int(k)]
    if generator_id == "ALL_ACTIONS":
        return _legal_action_strings(context)
    raise ValueError(f"Unknown candidate generator: {generator_id}")


def build_safe_candidate_set(
    raw_topk: Sequence[Any],
    *,
    stop_action: Any = STOP_ACTION,
    myopic_action: Any,
    incumbent_action: Any | None,
    legal_actions: Sequence[Any],
) -> list[str]:
    legal = set(_legal_action_strings(CandidateGeneratorContext(legal_actions=list(legal_actions))))
    ordered = [stop_action, myopic_action]
    if incumbent_action is not None:
        ordered.append(incumbent_action)
    ordered.extend(raw_topk)
    result: list[str] = []
    seen: set[str] = set()
    illegal: list[str] = []
    for action in ordered:
        token = normalize_action(action)
        if token not in legal:
            illegal.append(token)
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    if illegal:
        raise ValueError(f"Candidate set contains illegal actions: {illegal}")
    if STOP_ACTION not in result:
        raise ValueError("Safe candidate set does not contain STOP")
    if normalize_action(myopic_action) not in result:
        raise ValueError("Safe candidate set does not contain myopic action")
    if incumbent_action is not None and normalize_action(incumbent_action) not in result:
        raise ValueError("Safe candidate set does not contain incumbent action")
    return result


def action_rank(action: Any, actions: Sequence[Any]) -> int | None:
    for index, candidate in enumerate(actions, start=1):
        if actions_equal(action, candidate):
            return index
    return None


def jaccard_actions(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    left_set = {normalize_action(action) for action in left}
    right_set = {normalize_action(action) for action in right}
    if not left_set and not right_set:
        return None
    return float(len(left_set & right_set) / len(left_set | right_set))

