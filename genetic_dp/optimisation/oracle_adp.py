from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
import re


def state_key(state):
    return tuple(sorted(state))


def state_label(state) -> str:
    key = state_key(state)
    if not key:
        return "root"
    return "__".join(f"{_safe_token(person)}_{_safe_token(outcome)}" for person, outcome in key)


def _safe_token(value) -> str:
    if isinstance(value, tuple):
        raw = "_".join(_safe_token(part) for part in value)
    else:
        raw = str(value)
    token = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return token or "x"


def _value_lookup(values, state, default=0.0) -> float:
    if not isinstance(values, Mapping):
        return default if default is None else float(default)
    key = state_key(state)
    value = values.get(state, values.get(key, values.get(state_label(state), default)))
    if value is None:
        return default if default is None else float(default)
    value_f = float(value)
    if isfinite(value_f):
        return value_f
    return default if default is None else float(default)


def _policy_lookup(policy, state):
    if not isinstance(policy, Mapping):
        return None
    key = state_key(state)
    return policy.get(state, policy.get(key, policy.get(state_label(state))))


def _stage(state) -> int:
    return len(state_key(state))


def _feature_name_for_state(state) -> str:
    return f"oracle_state__{state_label(state)}"


def _select_states(*, exact_values, state_pool, top_k: int) -> list:
    states = list(state_pool or [])
    if not states and isinstance(exact_values, Mapping):
        states = [key for key in exact_values if isinstance(key, frozenset)]
    root = [state for state in states if _stage(state) == 0]
    nonroot = [state for state in states if _stage(state) > 0]
    nonroot.sort(key=lambda state: (_stage(state), -abs(_value_lookup(exact_values, state)), state_label(state)))
    selected = []
    selected.extend(sorted(root, key=state_label)[:1])
    selected.extend(nonroot)
    deduped = []
    seen = set()
    for state in selected:
        key = state_key(state)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(state)
        if len(deduped) >= max(1, int(top_k)):
            break
    return deduped


def _action_kind(action):
    if isinstance(action, (tuple, list)) and action:
        return action[0]
    if isinstance(action, Mapping):
        return action.get("kind") or action.get("action")
    return None


def _action_person(action):
    if isinstance(action, (tuple, list)) and len(action) > 1:
        return action[1]
    if isinstance(action, Mapping):
        return action.get("person") or action.get("who")
    return None


def _state_feature_payload(
    state,
    *,
    exact_values,
    policy_exact=None,
    oracle_policy=None,
    baseline_values=None,
    belief=None,
    individuals=None,
):
    policy = policy_exact if policy_exact is not None else oracle_policy
    exact_value = _value_lookup(exact_values, state, default=None)
    baseline_value = _value_lookup(baseline_values, state, default=None)
    action = _policy_lookup(policy, state)
    action_kind = _action_kind(action)
    action_person = _action_person(action)
    tested = {person for person, _ in state_key(state)}
    untested_count = None
    if individuals is not None:
        untested_count = max(0, len(list(individuals)) - len(tested))
    gap = None
    if exact_value is not None and baseline_value is not None:
        gap = float(exact_value) - float(baseline_value)
    return {
        "state_key": state_key(state),
        "state_label": state_label(state),
        "stage": _stage(state),
        "tested_count": len(tested),
        "untested_count": untested_count,
        "oracle_value": exact_value,
        "baseline_value": baseline_value,
        "oracle_gap_to_baseline": gap,
        "oracle_action_kind": action_kind,
        "oracle_action_person": action_person,
        "bias": 1.0,
    }


def build_oracle_feature_payload(
    state=None,
    *,
    exact_values,
    policy_exact=None,
    oracle_policy=None,
    baseline_values=None,
    belief=None,
    individuals=None,
    state_pool=None,
    mode: str = "exact_value_fixed",
    top_k: int = 32,
):
    """Build a deliberately non-production exact-backed feature payload."""

    if state is not None:
        return _state_feature_payload(
            state,
            exact_values=exact_values,
            policy_exact=policy_exact,
            oracle_policy=oracle_policy,
            baseline_values=baseline_values,
            belief=belief,
            individuals=individuals,
        )

    mode = (mode or "exact_value_fixed").strip().lower()
    if mode not in {"exact_value_fixed", "sparse_exact_value", "exact_advantage", "oracle_residual"}:
        raise ValueError(
            f"Unknown ORACLE_ADP_MODE={mode!r} "
            "(expected exact_value_fixed, sparse_exact_value, exact_advantage, or oracle_residual)."
        )
    policy = policy_exact if policy_exact is not None else oracle_policy
    selected_states = _select_states(exact_values=exact_values, state_pool=state_pool, top_k=top_k)
    selected_features: list[str] = []
    if mode == "sparse_exact_value":
        selected_features = [_feature_name_for_state(state) for state in selected_states]
    elif mode in {"exact_advantage", "oracle_residual"}:
        selected_features = [
            "bias",
            "oracle_exact_value",
            "oracle_value",
            "oracle_gap_to_baseline",
            "oracle_exact_policy_tests",
            "oracle_exact_policy_stops",
            "oracle_exact_value_stage_scaled",
        ]
        selected_features.extend(_feature_name_for_state(state) for state in selected_states[: max(0, int(top_k) - 4)])

    root_value = _value_lookup(exact_values, frozenset(), default=None) if isinstance(exact_values, Mapping) else None
    return {
        "enabled": True,
        "mode": mode,
        "top_k": int(top_k),
        "exact_values": exact_values,
        "policy_exact": policy or {},
        "oracle_policy": policy or {},
        "baseline_values": baseline_values or {},
        "selected_states": selected_states,
        "selected_state_labels": [state_label(state) for state in selected_states],
        "feature_names": list(selected_features),
        "selected_features": list(selected_features),
        "root_exact_value": root_value,
        "root_oracle_value": root_value,
        "diagnostics": {
            "exact_value_count": len(exact_values or {}),
            "policy_state_count": len(policy or {}),
            "selected_state_count": len(selected_states),
        },
    }


def oracle_feature_values(state, star) -> dict[str, float]:
    if not isinstance(star, Mapping) or not star.get("enabled"):
        return {}
    exact_values = star.get("exact_values", {})
    policy = star.get("policy_exact", star.get("oracle_policy", {}))
    baseline_values = star.get("baseline_values", {})
    mode = star.get("mode")
    exact_value = _value_lookup(exact_values, state, default=0.0)
    baseline_value = _value_lookup(baseline_values, state, default=0.0)
    action = _policy_lookup(policy, state)
    values: dict[str, float] = {
        "bias": 1.0,
        "oracle_value": exact_value,
        "oracle_exact_value": exact_value,
        "baseline_value": baseline_value,
        "oracle_gap_to_baseline": exact_value - baseline_value,
        "oracle_exact_value_stage_scaled": exact_value / float(1 + _stage(state)),
        "oracle_exact_policy_tests": 1.0 if action and _action_kind(action) == "test" else 0.0,
        "oracle_exact_policy_stops": 1.0 if action and _action_kind(action) == "stop" else 0.0,
    }
    if mode in {"sparse_exact_value", "exact_advantage", "oracle_residual"}:
        selected = star.get("selected_states", ())
        selected_keys = {state_key(selected_state): _feature_name_for_state(selected_state) for selected_state in selected}
        feature_name = selected_keys.get(state_key(state))
        if feature_name:
            values[feature_name] = 1.0
    return values


def oracle_adp_term_value(state, star, **_kwargs) -> float:
    if not isinstance(star, Mapping) or not star.get("enabled"):
        return 0.0
    term = 0.0
    mode = star.get("mode")
    if mode == "exact_value_fixed":
        term += _value_lookup(star.get("exact_values", {}), state, default=0.0)
    coeffs = star.get("coefficients", {})
    if isinstance(coeffs, Mapping) and coeffs:
        features = oracle_feature_values(state, star)
        for name, coef in coeffs.items():
            term += float(coef or 0.0) * float(features.get(name, 0.0))
    return float(term)


def serializable_summary(star):
    if not isinstance(star, Mapping):
        return None
    coeffs = star.get("coefficients", {}) if isinstance(star.get("coefficients"), Mapping) else {}
    nonzero = sum(1 for value in coeffs.values() if abs(float(value or 0.0)) > 1e-9)
    policy = star.get("policy_exact", star.get("oracle_policy", {}))
    root_value = star.get("root_oracle_value", star.get("root_exact_value"))
    return {
        "enabled": bool(star.get("enabled")),
        "mode": star.get("mode"),
        "top_k": star.get("top_k"),
        "feature_names": list(star.get("feature_names", ())),
        "selected_features": list(star.get("selected_features", ())),
        "selected_state_labels": list(star.get("selected_state_labels", ())),
        "coefficient_count": len(coeffs),
        "coefficient_nonzero": int(nonzero),
        "exact_value_count": len(star.get("exact_values", {}) or {}),
        "policy_state_count": len(policy or {}),
        "oracle_state_count": len(star.get("exact_values", {}) or {}),
        "oracle_policy_state_count": len(policy or {}),
        "root_exact_value": root_value,
        "root_oracle_value": root_value,
        "diagnostics": dict(star.get("diagnostics", {}) or {}),
    }
