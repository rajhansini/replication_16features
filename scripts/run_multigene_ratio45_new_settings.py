#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from genetic_dp.experiments.core import run_and_compare_solvers
from genetic_dp.models.belief import InferenceResult
from genetic_dp.models.reward import r_reward, r_reward_test, r_reward_testp
from genetic_dp.optimisation.myopic_adp import (
    build_state_features,
    feature_semantics_for_bank,
    regime_parameter_gates,
    regime_residual_v2_candidate_features,
    resolve_feature_bank,
    resolve_feature_semantics,
    select_weighted_signature_features,
)
from genetic_dp.policy.baselines import myopic_greedy
from genetic_dp.policy.evaluator import exact_value_under_policy
from genetic_dp.utils.artifact_paths import safe_artifact_path
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from scripts.search_multigene_myopic_vs_stop import COEF_PRESETS, FAMILY_CASES, _build_config


FIXED_ENV = {
    "EXACT_DP_SOLVER": "dual",
    "ENABLE_TUPLE_ROWGEN": "1",
    "ENABLE_PER_GENE_PHI": "1",
    "EXHAUSTIVE_BELLMAN": "1",
    "EXHAUSTIVE_STRICT": "1",
    "MAX_STATES_PER_ITER": "110000",
    "MAX_CUTS_PER_ITER": "1500000",
    "EXHAUSTIVE_WALLTIME_LIMIT_SEC": "7200",
    "EXHAUSTIVE_NO_PROGRESS_LIMIT_SEC": "600",
    "EXHAUSTIVE_HEARTBEAT_EVERY_SEC": "30",
    "EXHAUSTIVE_HEARTBEAT_EVERY_ITERS": "1",
    "DISABLE_TRUNCATED_TUPLE_STRENGTHENING": "1",
    "GUROBI_SEED": "0",
    "PULP_SOLVER": "gurobi",
}

DEFAULT_INCUMBENT_ENV = {
    "THETA_MODE": "stage",
    "THETA_MODEL": "",
    "THETA_MODEL_SPEC_PATH": "",
    "ENABLE_EDGE_FEATURES": "0",
}


@dataclass(frozen=True)
class RunnerSpec:
    label: str
    env: Dict[str, str]
    feature_bank: Optional[str] = None
    selector_mode: Optional[str] = None


@dataclass(frozen=True)
class Setting:
    name: str
    family: str
    preset: str
    allele_freqs: Dict[str, float]
    a_scale: float
    b_scale: float
    delta_shift: float
    fixed_cost: float
    variable_cost: float

    @property
    def pedigree(self):
        return generate_deterministic_pedigree(FAMILY_CASES[self.family])

    def build_config(self):
        return _build_config(
            self.pedigree,
            allele_freqs=self.allele_freqs,
            preset_label=self.preset,
            a_scale=self.a_scale,
            b_scale=self.b_scale,
            delta_shift=self.delta_shift,
            fixed_cost=self.fixed_cost,
            variable_cost=self.variable_cost,
        )


def setting_to_manifest_entry(setting: Setting) -> Dict[str, object]:
    return {
        "name": setting.name,
        "family": setting.family,
        "preset": setting.preset,
        "allele_freqs": dict(setting.allele_freqs),
        "a_scale": setting.a_scale,
        "b_scale": setting.b_scale,
        "delta_shift": setting.delta_shift,
        "fixed_cost": setting.fixed_cost,
        "variable_cost": setting.variable_cost,
    }


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_env(raw_env: Optional[Mapping[str, object]]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not raw_env:
        return env
    for key, value in raw_env.items():
        env[str(key)] = "" if value is None else str(value)
    return env


def _env_enabled(env: Mapping[str, str], name: str) -> bool:
    value = str(env.get(name, "")).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _spec_uses_grr(env: Mapping[str, str]) -> bool:
    return _env_enabled(env, "GAUGED_REGIME_RESIDUAL_ADP_ENABLED") or _env_enabled(
        env,
        "GAUGED_REGIME_RESIDUAL_V2_ENABLED",
    )


def _canonical_selector_mode(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    normalized = token.lower().replace("-", "_")
    aliases = {
        "root_test": "ROOTSIG",
        "rootsig": "ROOTSIG",
        "root_signature": "ROOTSIG",
        "state_pool_weighted": "STATEFALLBACK",
        "statefallback": "STATEFALLBACK",
        "state_fallback": "STATEFALLBACK",
        "statedual": "STATEDUAL",
        "state_dual": "STATEDUAL",
        "state_pool_dual": "STATEDUAL",
        "hotspot2": "HOTSPOT2",
    }
    return aliases.get(normalized, token.upper())


def _load_spec(path: Optional[str], *, default_label: str, default_env: Optional[Mapping[str, str]] = None) -> RunnerSpec:
    if path is None:
        return RunnerSpec(label=default_label, env=dict(default_env or {}), feature_bank=None, selector_mode=None)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Spec must be a JSON object: {path}")
    label = str(payload.get("label") or payload.get("run_name") or default_label)
    env = dict(default_env or {})
    env.update(_normalize_env(payload.get("env")))
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else {}
    selector_mode = _canonical_selector_mode(
        payload.get("selector_mode")
        or candidate.get("selector")
        or env.get("GAUGED_REGIME_RESIDUAL_SELECTOR")
    )
    if selector_mode and _spec_uses_grr(env) and "GAUGED_REGIME_RESIDUAL_SELECTOR" not in env:
        env["GAUGED_REGIME_RESIDUAL_SELECTOR"] = selector_mode
    feature_bank_raw = (
        payload.get("feature_bank")
        or candidate.get("feature_bank")
        or env.get("GAUGED_REGIME_FEATURE_BANK")
    )
    feature_bank = None
    if feature_bank_raw is None or str(feature_bank_raw).strip() == "":
        if _spec_uses_grr(env):
            raise ValueError(
                f"Benchmark-facing GRR spec {path} must declare feature_bank "
                "(FB0_PROXY, FB1_STRICT, FB1R_CALIB, FB2_HYBRID, or ABCD_HAND)."
            )
    else:
        feature_bank = resolve_feature_bank(feature_bank_raw, require=True)
        env["GAUGED_REGIME_FEATURE_BANK"] = feature_bank
        expected_semantics = feature_semantics_for_bank(feature_bank)
        legacy_semantics_raw = (
            candidate.get("feature_semantics")
            or env.get("GAUGED_REGIME_FEATURE_SEMANTICS")
        )
        if legacy_semantics_raw:
            legacy_semantics = resolve_feature_semantics(legacy_semantics_raw)
            if legacy_semantics != expected_semantics:
                raise ValueError(
                    f"Spec {path} feature_bank={feature_bank!r} implies "
                    f"feature_semantics={expected_semantics!r}, got {legacy_semantics!r}."
                )
        env["GAUGED_REGIME_FEATURE_SEMANTICS"] = expected_semantics
    return RunnerSpec(label=label, env=env, feature_bank=feature_bank, selector_mode=selector_mode)


def _load_settings(path: str) -> List[Setting]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_settings = payload.get("settings")
    if not isinstance(raw_settings, list) or not raw_settings:
        raise ValueError(f"Manifest must contain a non-empty settings list: {path}")
    settings: List[Setting] = []
    for raw in raw_settings:
        settings.append(
            Setting(
                name=str(raw["name"]),
                family=str(raw["family"]),
                preset=str(raw["preset"]),
                allele_freqs={str(k): float(v) for k, v in dict(raw["allele_freqs"]).items()},
                a_scale=float(raw.get("a_scale", 1.0)),
                b_scale=float(raw.get("b_scale", 1.0)),
                delta_shift=float(raw.get("delta_shift", 0.0)),
                fixed_cost=float(raw["fixed_cost"]),
                variable_cost=float(raw["variable_cost"]),
            )
        )
    return settings


@contextlib.contextmanager
def _temporary_env(updates: Mapping[str, str]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-9) -> Optional[float]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= eps:
        return None
    return numerator / denominator


def _clean_near_zero(value: Optional[float], *, eps: float = 1e-7) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if abs(value_f) <= eps:
        return 0.0
    return value_f


def _exact_solver_meta(env: Mapping[str, str]) -> Dict[str, str]:
    exact_solver_mode = str(env.get("EXACT_DP_SOLVER") or os.getenv("EXACT_DP_SOLVER", "dual"))
    exact_dual_lp_backend = str(
        env.get("EXACT_DUAL_LP_SOLVER") or os.getenv("EXACT_DUAL_LP_SOLVER", "gurobi")
    )
    return {
        "exact_solver_mode": exact_solver_mode,
        "exact_dual_lp_backend": exact_dual_lp_backend,
    }


def _action_label(action: object) -> Optional[str]:
    if action is None:
        return None
    if isinstance(action, (list, tuple)):
        kind = str(action[0]) if len(action) >= 1 and action[0] is not None else ""
        person = action[1] if len(action) >= 2 else None
        if kind.lower() == "stop":
            return "STOP"
        if person is not None:
            return str(person)
        return kind or None
    if isinstance(action, Mapping):
        kind = str(action.get("kind") or action.get("action") or "")
        person = action.get("person")
        if kind.lower() == "stop":
            return "STOP"
        if person is not None:
            return str(person)
        return kind or None
    return str(action)


def _action_payload(action: object) -> Optional[Dict[str, object]]:
    if action is None:
        return None
    if isinstance(action, Mapping):
        payload = dict(action)
        payload.setdefault("label", _action_label(action))
        return payload
    if isinstance(action, (list, tuple)):
        payload: Dict[str, object] = {
            "kind": action[0] if len(action) >= 1 else None,
            "person": action[1] if len(action) >= 2 else None,
            "value": action[2] if len(action) >= 3 else None,
        }
        payload["label"] = _action_label(action)
        return payload
    return {"kind": str(action), "person": None, "value": None, "label": _action_label(action)}


def _extract_exact_solver_status(message: str) -> Optional[str]:
    match = re.search(r"status=([^)]+)", message)
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def _extract_metrics(results: Mapping[str, object]) -> Dict[str, object]:
    exact = float(results["Exact_DP_root_value"])
    adp_phi = float(results["ADP_root_value_phi"])
    stop_value = float(results["ADP_root_value_R_stop"])
    production_policy_value = results.get("production_policy_value")
    if production_policy_value is None:
        production_policy_value = results["ADP_policy_value"]
    policy_value = float(production_policy_value)
    denom = exact - stop_value
    gap1 = _clean_near_zero(adp_phi - exact)
    gap2 = _clean_near_zero(exact - policy_value)
    ratio2 = _safe_ratio(gap2, denom)
    ratio3 = _clean_near_zero(_safe_ratio(gap1, denom))
    root_diagnostics = results.get("root_diagnostics")
    if not isinstance(root_diagnostics, Mapping):
        root_diagnostics = {}
    runtime_tracking = results.get("exact_dp_runtime_tracking")
    if not isinstance(runtime_tracking, Mapping):
        runtime_tracking = {}
    belief_tracking = runtime_tracking.get("belief_map")
    if not isinstance(belief_tracking, Mapping):
        belief_tracking = {}
    exact_dual_tracking = runtime_tracking.get("exact_dual")
    if not isinstance(exact_dual_tracking, Mapping):
        exact_dual_tracking = {}
    exact_policy = results.get("policy_exact") if isinstance(results.get("policy_exact"), Mapping) else {}
    exact_root_action = exact_policy.get(frozenset()) if isinstance(exact_policy, Mapping) else None
    return {
        "exact_root_value": exact,
        "exact_root_action": _action_payload(exact_root_action),
        "exact_root_action_label": _action_label(exact_root_action),
        "adp_phi": adp_phi,
        "stop_value": stop_value,
        "production_policy_value": policy_value,
        "raw_adp_policy_value": float(results.get("ADP_policy_value", policy_value)),
        "guarded_production_policy_value": policy_value,
        "guardrail_decision": results.get(
            "myopic_safe_guardrail_decision",
            results.get("safe_rollout_decision"),
        ),
        "gap1": gap1,
        "gap2": gap2,
        "denom_stop": denom,
        "ratio2": ratio2,
        "ratio3": ratio3,
        "selected_candidate_id": results.get("selected_candidate_id"),
        "production_policy_source": results.get("production_policy_source"),
        "myopic_policy_value_guardrail": results.get("myopic_policy_value"),
        "myopic_safe_guardrail_enabled": results.get("myopic_safe_guardrail_enabled"),
        "myopic_safe_guardrail_decision": results.get("myopic_safe_guardrail_decision"),
        "myopic_safe_guardrail_reason": results.get("myopic_safe_guardrail_reason"),
        "safe_rollout_decision": results.get("safe_rollout_decision"),
        "safe_rollout_reason": results.get("safe_rollout_reason"),
        "oracle_adp": results.get("oracle_adp", root_diagnostics.get("oracle_adp")),
        "oracle_adp_enabled": results.get("oracle_adp_enabled", root_diagnostics.get("oracle_adp_enabled")),
        "oracle_plumbing_mode": root_diagnostics.get("oracle_plumbing_mode"),
        "oracle_root_term": root_diagnostics.get("oracle_root_term"),
        "legacy_residual_root_term": root_diagnostics.get("legacy_residual_root_term"),
        "phi_root_lp": root_diagnostics.get("phi_root_lp"),
        "phi_root_reconstructed": root_diagnostics.get("phi_root_reconstructed"),
        "phi_root_lp_reconstruction_diff": root_diagnostics.get("phi_root_lp_reconstruction_diff"),
        "dual_component_available": root_diagnostics.get("dual_component_available"),
        "nonzero_dual_row_count": root_diagnostics.get("nonzero_dual_row_count"),
        "aggregated_dual_row_count": root_diagnostics.get("aggregated_dual_row_count"),
        "truncated_nonzero_dual_row_count": root_diagnostics.get("truncated_nonzero_dual_row_count"),
        "max_dual_complementarity_abs": root_diagnostics.get("max_dual_complementarity_abs"),
        "bellman_row_dual_validation": root_diagnostics.get("bellman_row_dual_validation"),
        "oracle_payload_coverage_count": root_diagnostics.get("oracle_payload_coverage_count"),
        "oracle_payload_missing_count": root_diagnostics.get("oracle_payload_missing_count"),
        "oracle_active_in_lp": root_diagnostics.get("oracle_active_in_lp"),
        "oracle_active_in_reconstruction": root_diagnostics.get("oracle_active_in_reconstruction"),
        "gauge_constraints_added": root_diagnostics.get("gauge_constraints_added"),
        "gauge_constraint_count": root_diagnostics.get("gauge_constraint_count"),
        "rowgen_oracle_only_truncated_tuple_cuts_suppressed": root_diagnostics.get(
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed"
        ),
        "bellman_signature_diagnostic": root_diagnostics.get("bellman_signature_diagnostic"),
        "gauged_regime_residual_enabled": root_diagnostics.get("gauged_regime_residual_enabled"),
        "gauged_regime_residual": root_diagnostics.get("gauged_regime_residual"),
        "selected_regime_features": root_diagnostics.get("selected_regime_features"),
        "selected_v1_base_features": root_diagnostics.get("selected_v1_base_features"),
        "selected_v2_features": root_diagnostics.get("selected_v2_features"),
        "regime_residual_selector": root_diagnostics.get("regime_residual_selector"),
        "regime_residual_anchor": root_diagnostics.get("regime_residual_anchor"),
        "regime_feature_scales": root_diagnostics.get("regime_feature_scales"),
        "regime_feature_root_values": root_diagnostics.get("regime_feature_root_values"),
        "regime_signature_by_root_action": root_diagnostics.get("regime_signature_by_root_action"),
        "regime_signature_residual_norms": root_diagnostics.get("regime_signature_residual_norms"),
        "regime_signature_incremental_norms": root_diagnostics.get("regime_signature_incremental_norms"),
        "regime_weighted_signature_diagnostics": root_diagnostics.get("regime_weighted_signature_diagnostics"),
        "legacy_signature_rank": root_diagnostics.get("legacy_signature_rank"),
        "selected_signature_rank": root_diagnostics.get("selected_signature_rank"),
        "regime_residual_root_term": root_diagnostics.get("regime_residual_root_term"),
        "regime_feature_bank": root_diagnostics.get("regime_feature_bank"),
        "regime_feature_semantics": root_diagnostics.get("regime_feature_semantics"),
        "truncated_tuple_cuts_suppressed": root_diagnostics.get(
            "truncated_tuple_cuts_suppressed",
            root_diagnostics.get("rowgen_oracle_only_truncated_tuple_cuts_suppressed"),
        ),
        "root_binding_action": root_diagnostics.get("candidate_root_binding_action"),
        "root_action_margin": root_diagnostics.get("candidate_root_action_margin"),
        "oracle_policy_enabled": results.get("oracle_policy_enabled", root_diagnostics.get("oracle_policy_enabled")),
        "oracle_policy_decision": results.get("oracle_policy_decision", root_diagnostics.get("oracle_policy_decision")),
        "oracle_policy_reason": results.get("oracle_policy_reason", root_diagnostics.get("oracle_policy_reason")),
        "exact_dp_runtime_tracking": dict(runtime_tracking),
        "exact_dp_total_time_sec": results.get("exact_dp_total_time"),
        "exact_dp_solve_phase_sec": results.get("exact_dp_solve_time"),
        "belief_map_build_time_sec": results.get(
            "belief_map_construction_time_sec",
            belief_tracking.get("elapsed_sec"),
        ),
        "belief_map_progress_status": results.get(
            "belief_map_progress_status",
            belief_tracking.get("progress_status"),
        ),
        "belief_map_build_mode": results.get("belief_map_build_mode", belief_tracking.get("mode")),
        "belief_map_state_count": results.get("belief_map_state_count", belief_tracking.get("state_count")),
        "belief_map_total_indexed_state_count": results.get(
            "belief_map_total_indexed_state_count",
            belief_tracking.get("total_indexed_state_count"),
        ),
        "belief_map_processed_state_count": results.get(
            "belief_map_processed_state_count",
            belief_tracking.get("processed_state_count"),
        ),
        "belief_map_generated_successor_count": results.get(
            "belief_map_generated_successor_count",
            belief_tracking.get("generated_successor_count"),
        ),
        "exact_dual_progress_status": results.get(
            "exact_dual_progress_status",
            exact_dual_tracking.get("progress_status"),
        ),
        "exact_dual_lp_backend": results.get(
            "exact_dual_lp_backend",
            exact_dual_tracking.get("backend"),
        ),
        "exact_dual_lp_status": results.get(
            "exact_dual_lp_status",
            exact_dual_tracking.get("status"),
        ),
        "exact_dual_lp_variable_count": results.get(
            "exact_dual_lp_variable_count",
            exact_dual_tracking.get("lp_variable_count"),
        ),
        "exact_dual_lp_constraint_count": results.get(
            "exact_dual_lp_constraint_count",
            exact_dual_tracking.get("lp_constraint_count"),
        ),
        "exact_dual_lp_build_time_sec": results.get(
            "exact_dual_lp_build_time_sec",
            exact_dual_tracking.get("lp_build_elapsed_sec"),
        ),
        "exact_dual_lp_solve_time_sec": results.get(
            "exact_dual_lp_solve_time_sec",
            exact_dual_tracking.get("lp_solve_elapsed_sec"),
        ),
        "exact_dual_lp_total_time_sec": results.get(
            "exact_dual_lp_total_time_sec",
            exact_dual_tracking.get("total_elapsed_sec"),
        ),
        "exact_dual_lp_log_path": results.get(
            "exact_dual_lp_log_path",
            exact_dual_tracking.get("log_path"),
        ),
    }


def _entry_payload(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _entry_tuple_pmfs(entry) -> Optional[Mapping[object, Mapping[object, float]]]:
    payload = _entry_payload(entry)
    if isinstance(payload, InferenceResult) and payload.has_tuple_pmfs():
        return payload.get_tuple_pmfs()
    return None


def _entry_per_gene(entry):
    payload = _entry_payload(entry)
    if isinstance(payload, InferenceResult):
        return payload.get_per_gene_probs()
    return None


def _wrap_belief_entry(entry, *, state: frozenset, individuals, gen_states):
    if isinstance(entry, tuple):
        return entry
    evidence = dict(state)
    z_post = {
        person: {g: 1.0 if evidence.get(person) == g else 0.0 for g in gen_states}
        for person in individuals
    }
    return (entry, z_post)


def _state_outcomes(
    *,
    state: frozenset,
    person: object,
    belief: Mapping[frozenset, object],
    gen_states: List[object],
) -> Iterable[tuple[object, float]]:
    entry = belief[state]
    tuple_pmfs = _entry_tuple_pmfs(entry)
    if tuple_pmfs is not None:
        person_dist = tuple_pmfs.get(person, {})
        for outcome, prob in person_dist.items():
            prob_f = float(prob)
            if prob_f > 0.0:
                yield outcome, prob_f
        return

    payload = _entry_payload(entry)
    marginals = payload.marginals if isinstance(payload, InferenceResult) else payload
    dist = marginals.get(person, {}) if isinstance(marginals, Mapping) else {}
    for outcome in gen_states:
        prob_f = float(dist.get(outcome, 0.0))
        if prob_f > 0.0:
            yield outcome, prob_f


def _merge_state(state: frozenset, person: object, outcome: object) -> frozenset:
    evidence = dict(state)
    evidence[person] = outcome
    return frozenset(evidence.items())


def _build_myopic_policy_map(
    *,
    belief,
    individuals,
    gen_states,
    infer,
    config,
) -> tuple[Dict[frozenset, object], Dict[frozenset, object], Dict[frozenset, object]]:
    policy: Dict[frozenset, object] = {}
    belief_gene: Dict[frozenset, object] = {}
    tuple_pmfs: Dict[frozenset, object] = {}
    tuple_mode = bool(getattr(config, "genes", None))

    for state, entry in belief.items():
        per_gene = _entry_per_gene(entry)
        if per_gene is not None:
            belief_gene[state] = per_gene
        entry_tuple_pmfs = _entry_tuple_pmfs(entry)
        if entry_tuple_pmfs is not None:
            tuple_pmfs[state] = entry_tuple_pmfs

    stack = [frozenset()]
    seen = {frozenset()}
    while stack:
        state = stack.pop()
        action = myopic_greedy(
            state,
            belief=belief,
            individuals=individuals,
            gen_states=gen_states,
            infer=infer,
            a=config.a,
            b=config.b,
            c=config.c,
            delta=config.delta,
            fixed_cost=config.fixed_cost,
            variable_cost=config.variable_cost,
            belief_gene=belief_gene,
            genes=config.genes if getattr(config, "genes", None) else None,
            a_gene=config.a_gene if getattr(config, "a_gene", None) else None,
            b_gene=config.b_gene if getattr(config, "b_gene", None) else None,
            c_gene=config.c_gene if getattr(config, "c_gene", None) else None,
            delta_gene=config.delta_gene if getattr(config, "delta_gene", None) else None,
            tuple_mode=tuple_mode,
        )
        policy[state] = action
        if action[0] != "test":
            continue
        person = action[1]
        for outcome, _prob in _state_outcomes(
            state=state,
            person=person,
            belief=belief,
            gen_states=gen_states,
        ):
            succ = _merge_state(state, person, outcome)
            if len(succ) >= len(individuals):
                continue
            if succ in seen:
                continue
            seen.add(succ)
            stack.append(succ)

    return policy, belief_gene, tuple_pmfs


def _compute_myopic_value(setting: Setting, results: Mapping[str, object]) -> Dict[str, object]:
    belief_exact = results.get("belief_exact")
    infer = results.get("infer")
    if not isinstance(belief_exact, Mapping):
        raise ValueError("run_and_compare_solvers did not return belief_exact; cannot evaluate myopic policy.")
    if infer is None:
        raise ValueError("run_and_compare_solvers did not return infer; cannot evaluate myopic policy.")

    config = setting.build_config()
    individuals = setting.pedigree.to_list()
    gen_states = [0, 1, 2]
    belief = {
        state: _wrap_belief_entry(entry, state=state, individuals=individuals, gen_states=gen_states)
        for state, entry in belief_exact.items()
    }
    policy, belief_gene, tuple_pmfs = _build_myopic_policy_map(
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        infer=infer,
        config=config,
    )
    value_map = exact_value_under_policy(
        policy=dict(policy),
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        r_reward_test=r_reward_test,
        a=config.a,
        b=config.b,
        c=config.c,
        delta=config.delta,
        infer=infer,
        theta_star=results.get("theta_star"),
        W_star=results.get("W_star"),
        theta_mode=os.getenv("THETA_MODE", "stage"),
        fixed_cost=config.fixed_cost,
        variable_cost=config.variable_cost,
        lookahead_depth=0,
        belief_gene=belief_gene,
        genes=config.genes if getattr(config, "genes", None) else None,
        a_gene=config.a_gene if getattr(config, "a_gene", None) else None,
        b_gene=config.b_gene if getattr(config, "b_gene", None) else None,
        c_gene=config.c_gene if getattr(config, "c_gene", None) else None,
        delta_gene=config.delta_gene if getattr(config, "delta_gene", None) else None,
        tuple_pmfs=tuple_pmfs if tuple_pmfs else None,
        aaub_star=results.get("aaub"),
        strict_mode=True,
    )
    root_value = value_map.get(frozenset())
    if root_value is None:
        raise ValueError("Exact myopic policy evaluation did not return a root value.")
    return {
        "myopic_value": float(root_value),
        "myopic_state_count": len(policy),
        "myopic_root_action": _action_payload(policy.get(frozenset())),
        "myopic_root_action_label": _action_label(policy.get(frozenset())),
    }


def _run_setting(
    setting: Setting,
    spec: RunnerSpec,
    *,
    benchmark_tier: str = "multigene_ratio45",
    progress_prefix: str = "multigene-ratio45",
) -> tuple[Dict[str, object], Dict[str, object], Dict[str, str]]:
    config = setting.build_config()
    env = dict(FIXED_ENV)
    env.update(spec.env)
    env["BENCHMARK_CASE"] = setting.name
    env["BENCHMARK_TIER"] = benchmark_tier
    env["BENCHMARK_RUN_ID"] = f"{benchmark_tier}::{spec.label}::{setting.name}"
    solver_meta = _exact_solver_meta(env)

    with _temporary_env(env):
        raw_results = run_and_compare_solvers(
            setting.pedigree,
            config,
            verbose=False,
            lookahead_depths=(0, 1),
            print_policies=False,
            progress_label=f"{progress_prefix}::{spec.label}::{setting.name}",
            belief_parallelism=1,
            dfvr_bound=False,
            return_infer=True,
        )
    return _extract_metrics(raw_results), raw_results, solver_meta


def _state_sort_key(state: frozenset) -> tuple:
    return tuple(sorted((str(person), repr(outcome)) for person, outcome in state))


def _posterior_payload(entry):
    return entry[0] if isinstance(entry, tuple) else entry


def _posterior_marginals(entry) -> Mapping:
    payload = _posterior_payload(entry)
    return payload.marginals if hasattr(payload, "marginals") else payload


def _per_gene_probs(entry) -> Mapping:
    payload = _posterior_payload(entry)
    if hasattr(payload, "get_per_gene_probs"):
        return payload.get_per_gene_probs() or {}
    return {}


def _tuple_pmfs(entry) -> Mapping:
    payload = _posterior_payload(entry)
    if hasattr(payload, "get_tuple_pmfs"):
        return payload.get_tuple_pmfs() or {}
    return {}


def _merge_state(state: frozenset, person: str, outcome) -> frozenset:
    return frozenset({*state, (person, outcome)})


def _gene_outcome(outcome, gene_idx: int):
    if isinstance(outcome, list):
        outcome = tuple(outcome)
    if isinstance(outcome, tuple):
        if gene_idx < len(outcome):
            return outcome[gene_idx]
        return outcome[0]
    return outcome


def _carrier_prob(dist: Mapping) -> float:
    return float(dist.get(1, 0.0) + dist.get(2, 0.0)) if isinstance(dist, Mapping) else 0.0


def _feature_value(raw_by_state: Mapping[frozenset, Mapping[str, float]], state: frozenset, name: str, root_values, scales) -> float:
    scale = float(scales.get(name, 0.0) or 0.0)
    if abs(scale) < 1e-12:
        return 0.0
    raw = raw_by_state.get(state, {})
    return (float(raw.get(name, 0.0) or 0.0) - float(root_values.get(name, 0.0) or 0.0)) / scale


def _normalise_components(rows: list[dict], component: str) -> dict[int, float]:
    values = {idx: max(0.0, float(row.get(component, 0.0) or 0.0)) for idx, row in enumerate(rows)}
    max_value = max(values.values(), default=0.0)
    if max_value <= 0.0:
        return {idx: 0.0 for idx in values}
    return {idx: value / max_value for idx, value in values.items()}


def _normalise_sum_components(rows: list[dict], component: str) -> dict[int, float]:
    values = {idx: max(0.0, float(row.get(component, 0.0) or 0.0)) for idx, row in enumerate(rows)}
    total = sum(values.values())
    if total <= 0.0:
        return {idx: 0.0 for idx in values}
    return {idx: value / total for idx, value in values.items()}


def _state_label_for_dual_key(state: frozenset) -> str:
    if not state:
        return "root"
    labels = []
    for person, outcome in state:
        if isinstance(outcome, list):
            outcome = tuple(outcome)
        if isinstance(outcome, tuple):
            outcome_label = "-".join(str(value) for value in outcome)
        else:
            outcome_label = str(outcome)
        labels.append(f"{person}_{outcome_label}")
    return "_".join(sorted(labels))


def _canonical_bellman_key(state: frozenset, row_type: str, person: Optional[str]) -> str:
    action = "STOP" if row_type == "stop" else str(person)
    return f"{_state_label_for_dual_key(state)}|{row_type}|{action}"


def _state_key_json(state: frozenset) -> list:
    entries = []
    for person, outcome in sorted(state, key=lambda item: (str(item[0]), repr(item[1]))):
        if isinstance(outcome, tuple):
            outcome_value = list(outcome)
        else:
            outcome_value = outcome
        entries.append([str(person), outcome_value])
    return entries


def _component_weights_for_mode(mode: str) -> Dict[str, float]:
    normalized = str(mode or "full_dual_mix").strip().lower()
    modes = {
        "dual_only": {"dual": 1.0, "tightness": 0.0, "margin": 0.0, "stage_reach": 0.0},
        "tightness_only": {"dual": 0.0, "tightness": 1.0, "margin": 0.0, "stage_reach": 0.0},
        "margin_only": {"dual": 0.0, "tightness": 0.0, "margin": 1.0, "stage_reach": 0.0},
        "reach_only": {"dual": 0.0, "tightness": 0.0, "margin": 0.0, "stage_reach": 1.0},
        "dual_plus_margin": {"dual": 0.7, "tightness": 0.0, "margin": 0.3, "stage_reach": 0.0},
        "dual_plus_tightness": {"dual": 0.7, "tightness": 0.3, "margin": 0.0, "stage_reach": 0.0},
        "fallback_current": {"dual": 0.0, "tightness": 0.25 / 0.55, "margin": 0.20 / 0.55, "stage_reach": 0.10 / 0.55},
        "full_dual_mix": {"dual": 0.45, "tightness": 0.25, "margin": 0.20, "stage_reach": 0.10},
    }
    if normalized not in modes:
        raise ValueError(f"Unknown GRR2D state-pool weight_mode={mode!r}.")
    return dict(modes[normalized])


def _row_summary_for_payload(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "canonical_row_key": row.get("canonical_row_key"),
        "row_type": row.get("kind") or row.get("row_type"),
        "state_key": _state_key_json(row["state"]) if isinstance(row.get("state"), frozenset) else row.get("state_key"),
        "stage": len(row["state"]) if isinstance(row.get("state"), frozenset) else row.get("stage"),
        "person_tested": row.get("person"),
        "slack": row.get("slack"),
        "dual_raw": row.get("dual_raw"),
        "tightness_raw": row.get("tightness_raw"),
        "margin_raw": row.get("margin_raw"),
        "stage_reach_raw": row.get("stage_reach_raw"),
        "weight": row.get("weight"),
        "source": row.get("source"),
    }


def _make_v2_selection_payload(
    *,
    setting: Setting,
    anchor_results: Mapping[str, object],
    anchor_mode: str,
    payload_path: Path,
    top_k: int,
    min_ratio: float,
    incremental_tol: float,
    feature_bank: str,
    feature_semantics: str,
    selector_mode: str = "STATEFALLBACK",
    weight_mode: Optional[str] = None,
    dual_component_required: bool = False,
    dual_positive_rows_only: bool = False,
    row_cap: int = 5000,
) -> dict:
    root_state = frozenset()
    root_diag = anchor_results.get("root_diagnostics") or {}
    if not isinstance(root_diag, Mapping):
        root_diag = {}
    selector_mode = _canonical_selector_mode(selector_mode) or "STATEFALLBACK"
    state_dual_selector = selector_mode == "STATEDUAL"
    if weight_mode is None:
        weight_mode = "full_dual_mix" if state_dual_selector else "fallback_current"
    component_weights = _component_weights_for_mode(weight_mode)
    dual_export = root_diag.get("bellman_row_dual_export") or {}
    if not isinstance(dual_export, Mapping):
        dual_export = {}
    dual_validation = dual_export.get("validation") or {}
    if not isinstance(dual_validation, Mapping):
        dual_validation = {}
    dual_required = bool(dual_component_required or state_dual_selector or component_weights.get("dual", 0.0) > 0.0)
    if dual_required:
        if feature_bank not in {"FB2_HYBRID", "ABCD_HAND"}:
            raise ValueError(
                "Dual-weighted state-pool selection requires feature_bank='FB2_HYBRID', "
                f"or 'ABCD_HAND', got {feature_bank!r}."
            )
        if selector_mode != "STATEDUAL":
            raise ValueError(f"Dual-weighted state-pool selection requires selector_mode='STATEDUAL', got {selector_mode!r}.")
        if not dual_export.get("available"):
            raise ValueError("GRR2D STATEDUAL requires an available bellman_row_dual_export from the anchor LP.")
        if not dual_validation.get("dual_component_available"):
            raise ValueError("GRR2D STATEDUAL requires nonzero LP dual mass; silent STATEFALLBACK is disallowed.")
        if not dual_validation.get("nonzero_dual_rows_not_truncated"):
            raise ValueError("GRR2D STATEDUAL validation failed: nonzero dual rows were truncated.")
        if not dual_validation.get("duplicate_canonical_rows_aggregated"):
            raise ValueError("GRR2D STATEDUAL validation failed: duplicate canonical rows were not aggregated.")
        if not dual_validation.get("complementarity_check_pass"):
            raise ValueError("GRR2D STATEDUAL validation failed: dual/slack complementarity exceeded tolerance.")
    dual_by_key: Dict[str, Mapping[str, object]] = {}
    raw_dual_rows = dual_export.get("aggregated_rows")
    if not isinstance(raw_dual_rows, list):
        raw_dual_rows = []
    for row in raw_dual_rows:
        if not isinstance(row, Mapping):
            continue
        key = row.get("canonical_row_key")
        if key:
            dual_by_key[str(key)] = row
    config = setting.build_config()
    pedigree = setting.pedigree
    individuals = pedigree.to_list()
    genes = tuple(config.genes or ())
    gene_list = genes
    belief = anchor_results.get("belief_post_approx")
    phi = anchor_results.get("Phi_star_approx")
    if not isinstance(belief, Mapping) or not isinstance(phi, Mapping):
        raise ValueError("V2 selector requires anchor belief_post_approx and Phi_star_approx.")
    tuple_posteriors_external = anchor_results.get("tuple_posteriors")
    if not isinstance(tuple_posteriors_external, Mapping):
        tuple_posteriors_external = {}
    anchor_star = root_diag.get("gauged_regime_residual") or {}
    selected_v1 = []
    if anchor_mode == "v1" and isinstance(anchor_star, Mapping):
        selected_v1 = [str(name) for name in anchor_star.get("selected_features", ()) or ()]

    regime_gates = regime_parameter_gates(
        genes=gene_list or None,
        a_gene=config.a_gene if gene_list else None,
        b_gene=config.b_gene if gene_list else None,
        delta_gene=config.delta_gene if gene_list else None,
        fixed_cost=config.fixed_cost,
        variable_cost=config.variable_cost,
    )
    feature_names = list(
        regime_residual_v2_candidate_features(
            gene_list or None,
            feature_bank=feature_bank,
        )
    )
    states = sorted(belief.keys(), key=_state_sort_key)
    raw_by_state = {
        state: build_state_features(
            state,
            belief=belief,
            individuals=individuals,
            pedigree=pedigree,
            genes=gene_list or None,
            regime_gates=regime_gates,
            feature_semantics=feature_semantics,
        )
        for state in states
    }
    root_raw = raw_by_state.get(root_state, {})
    root_values = {name: float(root_raw.get(name, 0.0) or 0.0) for name in feature_names}
    scales = {}
    skipped_low_scale = []
    for name in feature_names:
        root_value = root_values[name]
        scale = max(abs(float(raw_by_state[state].get(name, 0.0) or 0.0) - root_value) for state in states)
        if scale < 1e-12:
            skipped_low_scale.append(name)
            continue
        scales[name] = float(scale)

    def htilde(state, name):
        return _feature_value(raw_by_state, state, name, root_values, scales)

    def _state_entry(state):
        return belief[state]

    def _successors(state, person):
        entry = _state_entry(state)
        if gene_list:
            dist = _tuple_pmfs(entry).get(person, {})
            if not dist:
                dist = (tuple_posteriors_external.get(state, {}) or {}).get(person, {})
            if not dist:
                return []
        else:
            dist = _posterior_marginals(entry).get(person, {})
        succs = []
        for outcome, prob in dist.items():
            prob_f = float(prob)
            if prob_f <= 0.0:
                continue
            succ = _merge_state(state, person, outcome)
            if succ not in belief or succ not in phi:
                continue
            succs.append((succ, prob_f))
        return succs

    def _stop_rhs(state):
        entry = _state_entry(state)
        p_s = _posterior_marginals(entry)
        per_gene = _per_gene_probs(entry) if gene_list else None
        tested = {person for person, _ in state}
        return sum(
            r_reward(
                person,
                p_s,
                config.a,
                config.b,
                config.c,
                config.delta,
                per_gene_probs=per_gene,
                a_gene=config.a_gene if gene_list else None,
                b_gene=config.b_gene if gene_list else None,
                c_gene=config.c_gene if gene_list else None,
                delta_gene=config.delta_gene if gene_list else None,
            )
            for person in individuals
            if person not in tested
        )

    def _test_reward(state, person):
        entry = _state_entry(state)
        p_s = _posterior_marginals(entry)
        per_gene = _per_gene_probs(entry) if gene_list else {}
        p12 = _carrier_prob(p_s.get(person, {}))
        per_gene_p12 = {
            gene: _carrier_prob((per_gene.get(gene, {}) or {}).get(person, {}))
            for gene in gene_list
        } if gene_list else None
        return r_reward_testp(
            person,
            p12,
            config.a,
            config.b,
            config.c,
            config.delta,
            config.fixed_cost,
            config.variable_cost,
            per_gene_p12=per_gene_p12,
            a_gene=config.a_gene if gene_list else None,
            c_gene=config.c_gene if gene_list else None,
            delta_gene=config.delta_gene if gene_list else None,
        )

    rows_all = []
    for state in states:
        if state not in phi:
            continue
        phi_s = float(phi[state])
        tested = {person for person, _ in state}
        row_candidates = []
        stop_rhs = float(_stop_rhs(state))
        row_candidates.append(
            {
                "kind": "stop",
                "state": state,
                "person": None,
                "succs": [],
                "rhs": stop_rhs,
                "slack": phi_s - stop_rhs,
            }
        )
        for person in individuals:
            if person in tested:
                continue
            succs = _successors(state, person)
            if not succs:
                continue
            rhs = float(_test_reward(state, person)) + sum(prob * float(phi[succ]) for succ, prob in succs)
            row_candidates.append(
                {
                    "kind": "test",
                    "state": state,
                    "person": str(person),
                    "succs": succs,
                    "rhs": rhs,
                    "slack": phi_s - rhs,
                }
            )
        if not row_candidates:
            continue
        best_rhs = max(float(row["rhs"]) for row in row_candidates)
        for row in row_candidates:
            margin = max(0.0, best_rhs - float(row["rhs"]))
            row_type = str(row["kind"])
            person = row.get("person")
            canonical_row_key = _canonical_bellman_key(row["state"], row_type, person)
            dual_payload = dual_by_key.get(canonical_row_key)
            dual_raw = float(dual_payload.get("dual_abs", 0.0) or 0.0) if dual_payload else 0.0
            row["action_margin"] = margin
            row["canonical_row_key"] = canonical_row_key
            row["dual_row_count"] = int(dual_payload.get("raw_constraint_count", 0) or 0) if dual_payload else 0
            row["nonzero_dual_row_count"] = (
                int(dual_payload.get("nonzero_dual_row_count", 0) or 0) if dual_payload else 0
            )
            row["stage_reach_raw"] = 1.0 / (1.0 + len(state))
            row["tightness_raw"] = 1.0 / (1e-7 + abs(float(row["slack"])))
            row["margin_raw"] = 1.0 / (1e-7 + margin)
            row["dual_raw"] = dual_raw
            if state == root_state:
                row["source"] = "root"
            elif dual_raw > 0.0:
                row["source"] = "dual_support"
            else:
                row["source"] = "near_tight"
            rows_all.append(row)

    if not rows_all:
        raise ValueError("V2 selector produced no state-action rows.")
    row_keys = {str(row.get("canonical_row_key")) for row in rows_all}
    unmatched_nonzero_dual_rows = [
        {
            "canonical_row_key": key,
            "dual_abs": float(payload.get("dual_abs", 0.0) or 0.0),
            "raw_constraint_count": int(payload.get("raw_constraint_count", 0) or 0),
        }
        for key, payload in dual_by_key.items()
        if float(payload.get("dual_abs", 0.0) or 0.0) > 1e-12 and key not in row_keys
    ]
    if dual_required and unmatched_nonzero_dual_rows:
        sample = unmatched_nonzero_dual_rows[:5]
        raise ValueError(
            "GRR2D STATEDUAL validation failed: nonzero dual canonical rows were not "
            f"available to the selector. Sample={sample!r}"
        )
    dual_component = _normalise_sum_components(rows_all, "dual_raw")
    tight_component = _normalise_components(rows_all, "tightness_raw")
    margin_component = _normalise_components(rows_all, "margin_raw")
    reach_component = _normalise_components(rows_all, "stage_reach_raw")
    if dual_required and max(dual_component.values(), default=0.0) <= 0.0:
        raise ValueError("GRR2D STATEDUAL validation failed: no selector row received positive dual mass.")
    if not dual_required and max(dual_component.values(), default=0.0) <= 0.0 and component_weights.get("dual", 0.0) > 0.0:
        renorm = component_weights["tightness"] + component_weights["margin"] + component_weights["stage_reach"]
        component_weights = {
            "dual": 0.0,
            "tightness": component_weights["tightness"] / renorm,
            "margin": component_weights["margin"] / renorm,
            "stage_reach": component_weights["stage_reach"] / renorm,
        }
    for idx, row in enumerate(rows_all):
        row["weight"] = (
            component_weights["dual"] * dual_component[idx]
            + component_weights["tightness"] * tight_component[idx]
            + component_weights["margin"] * margin_component[idx]
            + component_weights["stage_reach"] * reach_component[idx]
        )
    pool_rows = rows_all
    if dual_positive_rows_only:
        if not dual_required:
            raise ValueError("dual_positive_rows_only requires dual_component_required or STATEDUAL selection.")
        pool_rows = [row for row in rows_all if float(row.get("dual_raw", 0.0) or 0.0) > 1e-12]
        if not pool_rows:
            raise ValueError("Dual-positive state-pool selection found no positive-dual selector rows.")
    root_rows = [row for row in pool_rows if row["state"] == root_state]
    non_root = [row for row in pool_rows if row["state"] != root_state]
    non_root.sort(key=lambda row: (-float(row["weight"]), len(row["state"]), str(row["kind"]), str(row.get("person"))))
    rows = root_rows + non_root[: max(0, int(row_cap) - len(root_rows))]
    weights = [max(1e-12, float(row.get("weight", 0.0) or 0.0)) for row in rows]

    def _signature(name, row):
        state = row["state"]
        current = htilde(state, name)
        if row["kind"] == "stop":
            return current
        return current - sum(prob * htilde(succ, name) for succ, prob in row["succs"])

    candidate_vectors = {
        name: [_signature(name, row) for row in rows]
        for name in feature_names
        if name in scales
    }

    legacy_vectors = []
    max_stage = max(len(state) for state in states)
    for stage in range(max_stage + 1):
        stage_vector = []
        for row in rows:
            current = 1.0 if len(row["state"]) == stage else 0.0
            if row["kind"] == "stop":
                stage_vector.append(current)
            else:
                stage_vector.append(
                    current - sum(prob * (1.0 if len(succ) == stage else 0.0) for succ, prob in row["succs"])
                )
        legacy_vectors.append(stage_vector)

    def _w_coeff(state, gene, person, genotype):
        entry = _state_entry(state)
        tested = dict(state)
        if person in tested:
            outcome = tested[person]
            if gene_list:
                gene_idx = gene_list.index(gene)
                obs = _gene_outcome(outcome, gene_idx)
            else:
                obs = outcome
            return 1.0 if int(obs) == int(genotype) else 0.0
        if gene_list:
            per_gene = _per_gene_probs(entry)
            dist = (per_gene.get(gene, {}) or {}).get(person, {})
        else:
            dist = _posterior_marginals(entry).get(person, {})
        return float(dist.get(genotype, 0.0) or 0.0)

    for gene in (gene_list or ("gene",)):
        for person in individuals:
            for genotype in (0, 1, 2):
                vector = []
                for row in rows:
                    current = _w_coeff(row["state"], gene, person, genotype)
                    if row["kind"] == "stop":
                        vector.append(current)
                    else:
                        vector.append(
                            current
                            - sum(prob * _w_coeff(succ, gene, person, genotype) for succ, prob in row["succs"])
                        )
                legacy_vectors.append(vector)

    seed_vectors = [candidate_vectors[name] for name in selected_v1 if name in candidate_vectors]
    selected_v2, _selected_vectors, diagnostics = select_weighted_signature_features(
        candidate_vectors,
        weights=weights,
        legacy_vectors=legacy_vectors,
        selected_seed_vectors=seed_vectors,
        exclude_features=selected_v1,
        top_k=top_k,
        min_ratio=min_ratio,
        incremental_tol=incremental_tol,
    )
    fallback_low_ratio_selection = False
    if not selected_v1 and not selected_v2:
        fallback_low_ratio_selection = True
        selected_v2, _selected_vectors, diagnostics = select_weighted_signature_features(
            candidate_vectors,
            weights=weights,
            legacy_vectors=legacy_vectors,
            selected_seed_vectors=seed_vectors,
            exclude_features=selected_v1,
            top_k=top_k,
            min_ratio=0.0,
            incremental_tol=incremental_tol,
        )
    if not selected_v1 and not selected_v2 and candidate_vectors:
        fallback_low_ratio_selection = True
        scored = []
        for name, vector in candidate_vectors.items():
            norm = math.sqrt(sum(float(w) * float(v) * float(v) for w, v in zip(weights, vector)))
            if norm > incremental_tol:
                scored.append((norm, name))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected_v2 = [scored[0][1]]
            diagnostics = dict(diagnostics)
            diagnostics.setdefault("selection_order", []).append(
                {
                    "feature": selected_v2[0],
                    "residual_ratio": 0.0,
                    "incremental_norm": float(scored[0][0]),
                    "fallback": "raw_weighted_norm",
                }
            )
    selected_features = list(selected_v1) + list(selected_v2)
    dual_positive_selector_rows = int(sum(1 for row in rows_all if float(row.get("dual_raw", 0.0) or 0.0) > 1e-12))
    fallback_scores = {
        idx: (
            (0.25 / 0.55) * float(tight_component.get(idx, 0.0))
            + (0.20 / 0.55) * float(margin_component.get(idx, 0.0))
            + (0.10 / 0.55) * float(reach_component.get(idx, 0.0))
        )
        for idx in range(len(rows_all))
    }
    top_dual_rows = sorted(
        rows_all,
        key=lambda row: (-float(row.get("dual_raw", 0.0) or 0.0), str(row.get("canonical_row_key"))),
    )[:20]
    top_fallback_indices = sorted(
        range(len(rows_all)),
        key=lambda idx: (-float(fallback_scores.get(idx, 0.0)), str(rows_all[idx].get("canonical_row_key"))),
    )[:20]
    top_fallback_rows = [rows_all[idx] for idx in top_fallback_indices]
    payload = {
        "enabled": True,
        "mode": "gauged_regime_residual_v2_state_pool",
        "feature_bank": feature_bank,
        "feature_semantics": feature_semantics,
        "selector": selector_mode,
        "weight_mode": str(weight_mode),
        "anchor": anchor_mode,
        "setting_name": setting.name,
        "selected_v1_base_features": list(selected_v1),
        "selected_v2_features": list(selected_v2),
        "selected_features": selected_features,
        "feature_names": feature_names,
        "feature_root_values": {name: float(root_values.get(name, 0.0) or 0.0) for name in feature_names},
        "feature_scales": {name: float(scales[name]) for name in scales},
        "regime_gates": dict(regime_gates),
        "diagnostics": {
            "anchor_mode": anchor_mode,
            "feature_bank": feature_bank,
            "feature_semantics": feature_semantics,
            "state_pool_count": len(states),
            "candidate_row_count": len(rows_all),
            "weighted_row_count": len(rows),
            "row_cap": int(row_cap),
            "root_row_count": len(root_rows),
            "selector_mode": selector_mode,
            "weight_mode": str(weight_mode),
            "dual_component_required": bool(dual_required),
            "dual_positive_rows_only": bool(dual_positive_rows_only),
            "dual_component_available": bool(dual_positive_selector_rows > 0),
            "nonzero_dual_row_count": int(dual_export.get("nonzero_dual_row_count") or 0),
            "dual_positive_selector_row_count": int(dual_positive_selector_rows),
            "aggregated_dual_row_count": int(dual_export.get("aggregated_row_count") or 0),
            "duplicate_canonical_row_count": int(dual_export.get("duplicate_canonical_row_count") or 0),
            "truncated_nonzero_dual_row_count": int(dual_export.get("truncated_nonzero_dual_row_count") or 0),
            "max_dual_complementarity_abs": dual_export.get("max_complementarity_abs"),
            "bellman_row_dual_validation": dict(dual_validation),
            "unmatched_nonzero_dual_row_count": int(len(unmatched_nonzero_dual_rows)),
            "unmatched_nonzero_dual_rows_sample": unmatched_nonzero_dual_rows[:10],
            "top_20_dual_weighted_rows": [_row_summary_for_payload(row) for row in top_dual_rows],
            "top_20_fallback_weighted_rows": [_row_summary_for_payload(row) for row in top_fallback_rows],
            "component_weights": component_weights,
            "skipped_low_scale_features": skipped_low_scale,
            "selected_v1_base_features": list(selected_v1),
            "selected_v2_features": list(selected_v2),
            "fallback_low_ratio_selection": bool(fallback_low_ratio_selection),
            "feature_definition": {
                "raw": "h_j(s)",
                "centered_scaled": "htilde_j(s)=(h_j(s)-h_j(s0))/scale_j",
                "state_pool_signature": "d_j(s,a)=htilde_j(s)-E[htilde_j(s')|s,a]",
            },
            "forbidden_inputs_used": False,
            **diagnostics,
        },
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _enrich_with_myopic(metrics: Mapping[str, object], *, myopic_value: float) -> Dict[str, object]:
    enriched = dict(metrics)
    exact = float(metrics["exact_root_value"])
    gap1 = float(metrics["gap1"])
    policy_value = float(metrics["production_policy_value"])
    denom_myopic = exact - myopic_value
    ratio4 = _safe_ratio(exact - policy_value, denom_myopic)
    ratio5 = _safe_ratio(gap1, denom_myopic)
    ratio3 = metrics.get("ratio3")
    ratio5_better_than_ratio3 = None
    if ratio5 is not None and ratio3 is not None:
        ratio5_better_than_ratio3 = ratio5 < float(ratio3)
    enriched.update(
        {
            "myopic_value": float(myopic_value),
            "denom_myopic": float(denom_myopic),
            "ratio4": ratio4,
            "ratio5": ratio5,
            "ratio5_better_than_ratio3": ratio5_better_than_ratio3,
        }
    )
    return enriched


def _build_row(
    *,
    setting: Setting,
    incumbent_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    myopic: Mapping[str, object],
    solver_meta: Mapping[str, str],
    r2_threshold: float,
    r3_threshold: float,
    anchor_metrics: Optional[Mapping[str, object]] = None,
    selection_payload_path: Optional[Path] = None,
    selection_payload: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    myopic_value = float(myopic["myopic_value"])
    incumbent = _enrich_with_myopic(incumbent_metrics, myopic_value=myopic_value)
    candidate = _enrich_with_myopic(candidate_metrics, myopic_value=myopic_value)
    exact = float(candidate["exact_root_value"])
    stop_value = float(candidate["stop_value"])
    candidate_ratio2 = candidate.get("ratio2")
    r3 = candidate.get("ratio3")
    exact_minus_stop = exact - stop_value
    stop_minus_myopic = stop_value - myopic_value
    policy_value = float(candidate["production_policy_value"])
    v_best_simple = max(float(stop_value), float(myopic_value))
    policy_gap = exact - policy_value
    myopic_regret = exact - myopic_value
    best_simple_regret = exact - v_best_simple
    bound_gap = float(candidate["adp_phi"]) - policy_value
    bound_ratio_exact = _safe_ratio(bound_gap, exact_minus_stop)
    ratio_sum = None
    ratio_identity_abs_error = None
    ratio3 = candidate.get("ratio3")
    if bound_ratio_exact is not None and candidate_ratio2 is not None and ratio3 is not None:
        ratio_sum = float(candidate_ratio2) + float(ratio3)
        ratio_identity_abs_error = abs(float(bound_ratio_exact) - ratio_sum)
    ratio_identity_ok = ratio_identity_abs_error is not None and ratio_identity_abs_error <= 1e-7
    exact_first_action = candidate.get("exact_root_action_label")
    myopic_first_action = myopic.get("myopic_root_action_label")
    if candidate.get("production_policy_source") == "myopic_policy":
        grr_policy_action = myopic_first_action
    else:
        grr_policy_action = candidate.get("root_binding_action") or candidate.get("selected_candidate_id")
    gate_exact_optimal = True
    gate_myopic_lt_stop_lt_exact = myopic_value < stop_value < exact
    gate_candidate_r2_le_threshold = (
        candidate_ratio2 is not None and float(candidate_ratio2) <= float(r2_threshold)
    )
    gate_r3_le_threshold = r3 is not None and float(r3) <= float(r3_threshold)
    gate_all_target_conditions = (
        gate_exact_optimal
        and gate_myopic_lt_stop_lt_exact
        and gate_candidate_r2_le_threshold
        and gate_r3_le_threshold
    )
    anchor = _enrich_with_myopic(anchor_metrics, myopic_value=myopic_value) if anchor_metrics else None
    delta_gap1_vs_anchor = None
    delta_ratio3_vs_anchor = None
    delta_ratio2_vs_anchor = None
    if anchor:
        delta_gap1_vs_anchor = candidate.get("gap1") - anchor.get("gap1")
        delta_ratio3_vs_anchor = candidate.get("ratio3") - anchor.get("ratio3")
        delta_ratio2_vs_anchor = (
            None
            if candidate.get("ratio2") is None or anchor.get("ratio2") is None
            else candidate.get("ratio2") - anchor.get("ratio2")
        )
    return {
        "setting_name": setting.name,
        "family": setting.family,
        "preset": setting.preset,
        "allele_freqs": dict(setting.allele_freqs),
        "a_scale": setting.a_scale,
        "b_scale": setting.b_scale,
        "fixed_cost": setting.fixed_cost,
        "variable_cost": setting.variable_cost,
        "delta_shift": setting.delta_shift,
        "exact_solver_mode": solver_meta["exact_solver_mode"],
        "exact_dual_lp_backend": solver_meta["exact_dual_lp_backend"],
        "incumbent_exact_solver_status": "Optimal",
        "candidate_exact_solver_status": "Optimal",
        "exact_solver_status": "Optimal",
        "exact_dp_runtime_tracking": candidate.get("exact_dp_runtime_tracking"),
        "exact_dp_total_time_sec": candidate.get("exact_dp_total_time_sec"),
        "exact_dp_solve_phase_sec": candidate.get("exact_dp_solve_phase_sec"),
        "belief_map_build_time_sec": candidate.get("belief_map_build_time_sec"),
        "belief_map_progress_status": candidate.get("belief_map_progress_status"),
        "belief_map_build_mode": candidate.get("belief_map_build_mode"),
        "belief_map_state_count": candidate.get("belief_map_state_count"),
        "belief_map_total_indexed_state_count": candidate.get("belief_map_total_indexed_state_count"),
        "belief_map_processed_state_count": candidate.get("belief_map_processed_state_count"),
        "belief_map_generated_successor_count": candidate.get("belief_map_generated_successor_count"),
        "exact_dual_progress_status": candidate.get("exact_dual_progress_status"),
        "exact_dual_lp_status": candidate.get("exact_dual_lp_status"),
        "exact_dual_lp_variable_count": candidate.get("exact_dual_lp_variable_count"),
        "exact_dual_lp_constraint_count": candidate.get("exact_dual_lp_constraint_count"),
        "exact_dual_lp_build_time_sec": candidate.get("exact_dual_lp_build_time_sec"),
        "exact_dual_lp_solve_time_sec": candidate.get("exact_dual_lp_solve_time_sec"),
        "exact_dual_lp_total_time_sec": candidate.get("exact_dual_lp_total_time_sec"),
        "exact_dual_lp_log_path": candidate.get("exact_dual_lp_log_path"),
        "incumbent_exact_dp_runtime_tracking": incumbent.get("exact_dp_runtime_tracking"),
        "incumbent_belief_map_build_time_sec": incumbent.get("belief_map_build_time_sec"),
        "incumbent_exact_dual_lp_solve_time_sec": incumbent.get("exact_dual_lp_solve_time_sec"),
        "candidate_exact_dp_runtime_tracking": candidate.get("exact_dp_runtime_tracking"),
        "candidate_belief_map_build_time_sec": candidate.get("belief_map_build_time_sec"),
        "candidate_exact_dual_lp_solve_time_sec": candidate.get("exact_dual_lp_solve_time_sec"),
        "exact_root_value": exact,
        "V_star": exact,
        "stop_value": stop_value,
        "V_stop": stop_value,
        "myopic_value": myopic_value,
        "V_myopic": myopic_value,
        "V_best_simple": v_best_simple,
        "myopic_policy_value": myopic_value,
        "myopic_state_count": myopic["myopic_state_count"],
        "exact_root_action": candidate.get("exact_root_action"),
        "exact_first_action": exact_first_action,
        "myopic_root_action": myopic.get("myopic_root_action"),
        "myopic_first_action": myopic_first_action,
        "ADP_root_value_phi": candidate.get("adp_phi"),
        "U_reference": candidate.get("adp_phi"),
        "gap1": candidate.get("gap1"),
        "ratio3": candidate.get("ratio3"),
        "production_policy_value": candidate.get("production_policy_value"),
        "V_policy": policy_value,
        "L_policy": policy_value,
        "policy_gap": policy_gap,
        "myopic_regret": myopic_regret,
        "best_simple_regret": best_simple_regret,
        "production_action": grr_policy_action,
        "GRR_policy_action": grr_policy_action,
        "raw_adp_policy_value": candidate.get("raw_adp_policy_value"),
        "guarded_production_policy_value": candidate.get("guarded_production_policy_value"),
        "guardrail_decision": candidate.get("guardrail_decision"),
        "incumbent_policy_value": incumbent.get("production_policy_value"),
        "ratio2": candidate.get("ratio2"),
        "candidate_ratio2": candidate_ratio2,
        "r3": r3,
        "bound_gap": bound_gap,
        "bound_ratio_exact": bound_ratio_exact,
        "ratio3_plus_ratio2": ratio_sum,
        "ratio3_plus_ratio2_check": ratio_identity_ok,
        "ratio_identity_abs_error": ratio_identity_abs_error,
        "upper_bound_status": "certified" if candidate.get("gap1") is not None and float(candidate.get("gap1")) >= -1e-7 else "invalid",
        "root_reconstruction_diff": candidate.get("phi_root_lp_reconstruction_diff"),
        "oracle_plumbing_mode": candidate.get("oracle_plumbing_mode"),
        "oracle_root_term": candidate.get("oracle_root_term"),
        "legacy_residual_root_term": candidate.get("legacy_residual_root_term"),
        "phi_root_lp": candidate.get("phi_root_lp"),
        "phi_root_reconstructed": candidate.get("phi_root_reconstructed"),
        "phi_root_lp_reconstruction_diff": candidate.get("phi_root_lp_reconstruction_diff"),
        "dual_component_available": candidate.get("dual_component_available"),
        "nonzero_dual_row_count": candidate.get("nonzero_dual_row_count"),
        "aggregated_dual_row_count": candidate.get("aggregated_dual_row_count"),
        "truncated_nonzero_dual_row_count": candidate.get("truncated_nonzero_dual_row_count"),
        "max_dual_complementarity_abs": candidate.get("max_dual_complementarity_abs"),
        "bellman_row_dual_validation": candidate.get("bellman_row_dual_validation"),
        "oracle_payload_coverage_count": candidate.get("oracle_payload_coverage_count"),
        "oracle_payload_missing_count": candidate.get("oracle_payload_missing_count"),
        "oracle_active_in_lp": candidate.get("oracle_active_in_lp"),
        "oracle_active_in_reconstruction": candidate.get("oracle_active_in_reconstruction"),
        "gauge_constraints_added": candidate.get("gauge_constraints_added"),
        "gauge_constraint_count": candidate.get("gauge_constraint_count"),
        "rowgen_oracle_only_truncated_tuple_cuts_suppressed": candidate.get(
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed"
        ),
        "bellman_signature_diagnostic": candidate.get("bellman_signature_diagnostic"),
        "gauged_regime_residual_enabled": candidate.get("gauged_regime_residual_enabled"),
        "gauged_regime_residual": candidate.get("gauged_regime_residual"),
        "selected_regime_features": candidate.get("selected_regime_features"),
        "selected_features": candidate.get("selected_regime_features"),
        "selected_v1_base_features": candidate.get("selected_v1_base_features"),
        "selected_v2_features": candidate.get("selected_v2_features"),
        "regime_residual_selector": candidate.get("regime_residual_selector"),
        "regime_residual_anchor": candidate.get("regime_residual_anchor"),
        "regime_feature_scales": candidate.get("regime_feature_scales"),
        "regime_feature_root_values": candidate.get("regime_feature_root_values"),
        "regime_signature_by_root_action": candidate.get("regime_signature_by_root_action"),
        "regime_signature_residual_norms": candidate.get("regime_signature_residual_norms"),
        "regime_signature_incremental_norms": candidate.get("regime_signature_incremental_norms"),
        "regime_weighted_signature_diagnostics": candidate.get("regime_weighted_signature_diagnostics"),
        "legacy_signature_rank": candidate.get("legacy_signature_rank"),
        "selected_signature_rank": candidate.get("selected_signature_rank"),
        "regime_residual_root_term": candidate.get("regime_residual_root_term"),
        "regime_feature_bank": candidate.get("regime_feature_bank"),
        "regime_feature_semantics": candidate.get("regime_feature_semantics"),
        "truncated_tuple_cuts_suppressed": candidate.get("truncated_tuple_cuts_suppressed"),
        "legacy_root_binding_action": incumbent.get("root_binding_action"),
        "candidate_root_binding_action": candidate.get("root_binding_action"),
        "legacy_root_action_margin": incumbent.get("root_action_margin"),
        "candidate_root_action_margin": candidate.get("root_action_margin"),
        "anchor": anchor,
        "v2_selection_payload_path": str(selection_payload_path) if selection_payload_path else None,
        "v2_selection_payload_summary": {
            "anchor": selection_payload.get("anchor") if isinstance(selection_payload, Mapping) else None,
            "feature_bank": selection_payload.get("feature_bank") if isinstance(selection_payload, Mapping) else None,
            "feature_semantics": selection_payload.get("feature_semantics") if isinstance(selection_payload, Mapping) else None,
            "selected_v1_base_features": selection_payload.get("selected_v1_base_features") if isinstance(selection_payload, Mapping) else None,
            "selected_v2_features": selection_payload.get("selected_v2_features") if isinstance(selection_payload, Mapping) else None,
            "weighted_row_count": ((selection_payload.get("diagnostics") or {}).get("weighted_row_count") if isinstance(selection_payload, Mapping) else None),
            "candidate_row_count": ((selection_payload.get("diagnostics") or {}).get("candidate_row_count") if isinstance(selection_payload, Mapping) else None),
            "legacy_signature_rank": ((selection_payload.get("diagnostics") or {}).get("legacy_signature_rank") if isinstance(selection_payload, Mapping) else None),
            "selected_signature_rank": ((selection_payload.get("diagnostics") or {}).get("selected_signature_rank") if isinstance(selection_payload, Mapping) else None),
            "selector_mode": ((selection_payload.get("diagnostics") or {}).get("selector_mode") if isinstance(selection_payload, Mapping) else None),
            "weight_mode": ((selection_payload.get("diagnostics") or {}).get("weight_mode") if isinstance(selection_payload, Mapping) else None),
            "dual_component_available": ((selection_payload.get("diagnostics") or {}).get("dual_component_available") if isinstance(selection_payload, Mapping) else None),
            "dual_positive_selector_row_count": ((selection_payload.get("diagnostics") or {}).get("dual_positive_selector_row_count") if isinstance(selection_payload, Mapping) else None),
            "nonzero_dual_row_count": ((selection_payload.get("diagnostics") or {}).get("nonzero_dual_row_count") if isinstance(selection_payload, Mapping) else None),
            "truncated_nonzero_dual_row_count": ((selection_payload.get("diagnostics") or {}).get("truncated_nonzero_dual_row_count") if isinstance(selection_payload, Mapping) else None),
            "max_dual_complementarity_abs": ((selection_payload.get("diagnostics") or {}).get("max_dual_complementarity_abs") if isinstance(selection_payload, Mapping) else None),
            "bellman_row_dual_validation": ((selection_payload.get("diagnostics") or {}).get("bellman_row_dual_validation") if isinstance(selection_payload, Mapping) else None),
        } if selection_payload else None,
        "delta_gap1_vs_anchor": delta_gap1_vs_anchor,
        "delta_ratio3_vs_anchor": delta_ratio3_vs_anchor,
        "delta_ratio2_vs_anchor": delta_ratio2_vs_anchor,
        "exact_minus_stop": exact_minus_stop,
        "stop_minus_myopic": stop_minus_myopic,
        "r2_threshold": float(r2_threshold),
        "r3_threshold": float(r3_threshold),
        "gate_exact_optimal": gate_exact_optimal,
        "gate_myopic_lt_stop_lt_exact": gate_myopic_lt_stop_lt_exact,
        "gate_candidate_r2_le_threshold": gate_candidate_r2_le_threshold,
        "gate_r3_le_threshold": gate_r3_le_threshold,
        "gate_all_target_conditions": gate_all_target_conditions,
        "incumbent": incumbent,
        "candidate": candidate,
    }


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _build_markdown(
    *,
    incumbent: RunnerSpec,
    candidate: RunnerSpec,
    rows: Iterable[Mapping[str, object]],
    payload: Mapping[str, object],
) -> str:
    lines = [
        "# Multigene Ratio4/Ratio5 Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Manifest: `{payload['manifest_path']}`",
        f"- Incumbent: `{incumbent.label}`",
        f"- Candidate: `{candidate.label}`",
        f"- Candidate `r2` threshold: `{payload['thresholds']['candidate_r2_max']}`",
        f"- `r3` threshold: `{payload['thresholds']['r3_max']}`",
        "",
        "| Setting | Family | Myopic value | Inc ratio2 | Inc ratio4 | Inc ratio3 | Inc ratio5 | Cand ratio2 | Cand ratio4 | Cand ratio3 | Cand ratio5 | Cand ratio5<ratio3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        inc = row["incumbent"]
        cand = row["candidate"]
        lines.append(
            "| {name} | {family} | {myopic:.6f} | {inc_r2} | {inc_r4} | {inc_r3} | {inc_r5} | {cand_r2} | {cand_r4} | {cand_r3} | {cand_r5} | {cand_better} |".format(
                name=row["setting_name"],
                family=row["family"],
                myopic=row["myopic_value"],
                inc_r2=_fmt_ratio(inc["ratio2"]),
                inc_r4=_fmt_ratio(inc["ratio4"]),
                inc_r3=_fmt_ratio(inc["ratio3"]),
                inc_r5=_fmt_ratio(inc["ratio5"]),
                cand_r2=_fmt_ratio(cand["ratio2"]),
                cand_r4=_fmt_ratio(cand["ratio4"]),
                cand_r3=_fmt_ratio(cand["ratio3"]),
                cand_r5=_fmt_ratio(cand["ratio5"]),
                cand_better=cand["ratio5_better_than_ratio3"],
            )
        )
    lines.extend(
        [
            "",
            f"- Candidate rows with `ratio5 < ratio3`: `{payload['summary']['candidate_ratio5_better_count']}/{payload['summary']['row_count']}`",
            f"- Incumbent rows with `ratio5 < ratio3`: `{payload['summary']['incumbent_ratio5_better_count']}/{payload['summary']['row_count']}`",
            f"- Rows with exact solver optimal: `{payload['summary']['gate_exact_optimal_count']}/{payload['summary']['row_count']}`",
            f"- Rows with `myopic < stop < exact`: `{payload['summary']['gate_ordering_count']}/{payload['summary']['row_count']}`",
            f"- Rows with candidate `r2 <= {payload['thresholds']['candidate_r2_max']}`: `{payload['summary']['gate_candidate_r2_count']}/{payload['summary']['row_count']}`",
            f"- Rows with `r3 <= {payload['thresholds']['r3_max']}`: `{payload['summary']['gate_r3_count']}/{payload['summary']['row_count']}`",
            f"- Rows satisfying all target gates: `{payload['summary']['target_success_count']}/{payload['summary']['row_count']}`",
        ]
            )
    if rows:
        lines.extend(
            [
                "",
                "## Target Gates",
                "",
                "| Setting | Exact Status | Exact-Stop | Stop-Myopic | Candidate r2 | r3 | All Target Gates |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| {name} | {status} | {exact_minus_stop:.6f} | {stop_minus_myopic:.6f} | {cand_r2} | {r3} | {gate_all} |".format(
                    name=row["setting_name"],
                    status=row["exact_solver_status"],
                    exact_minus_stop=row["exact_minus_stop"],
                    stop_minus_myopic=row["stop_minus_myopic"],
                    cand_r2=_fmt_ratio(row.get("candidate_ratio2")),
                    r3=_fmt_ratio(row.get("r3")),
                    gate_all=row["gate_all_target_conditions"],
                )
            )
        lines.extend(
            [
                "",
                "## Exact Runtime Tracking",
                "",
                "| Setting | Belief status | Belief mode | Belief states | Belief sec | Exact dual status | LP rows | LP vars | LP build sec | LP solve sec | Exact total sec |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {name} | {belief_status} | {belief_mode} | {belief_states} | {belief_sec} | {dual_status} | {lp_rows} | {lp_vars} | {lp_build} | {lp_solve} | {exact_total} |".format(
                    name=row["setting_name"],
                    belief_status=row.get("belief_map_progress_status") or "NA",
                    belief_mode=row.get("belief_map_build_mode") or "NA",
                    belief_states=row.get("belief_map_state_count") if row.get("belief_map_state_count") is not None else "",
                    belief_sec=_fmt_ratio(row.get("belief_map_build_time_sec")),
                    dual_status=row.get("exact_dual_progress_status") or "NA",
                    lp_rows=row.get("exact_dual_lp_constraint_count") if row.get("exact_dual_lp_constraint_count") is not None else "",
                    lp_vars=row.get("exact_dual_lp_variable_count") if row.get("exact_dual_lp_variable_count") is not None else "",
                    lp_build=_fmt_ratio(row.get("exact_dual_lp_build_time_sec")),
                    lp_solve=_fmt_ratio(row.get("exact_dual_lp_solve_time_sec")),
                    exact_total=_fmt_ratio(row.get("exact_dp_total_time_sec")),
                )
            )
        lines.extend(
            [
                "",
                "## Oracle Plumbing Diagnostics",
                "",
                "| Setting | Mode | Phi LP | Phi reconstructed | Oracle root term | Legacy residual root term | Coverage | Gauges | Suppressed truncated cuts | Active LP/Reconstruct |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            coverage = row.get("oracle_payload_coverage_count")
            missing = row.get("oracle_payload_missing_count")
            coverage_label = "NA" if coverage is None else f"{coverage}/{coverage + (missing or 0)}"
            active_label = f"{row.get('oracle_active_in_lp')}/{row.get('oracle_active_in_reconstruction')}"
            lines.append(
                "| {name} | {mode} | {phi_lp} | {phi_recon} | {oracle_root} | {resid_root} | {coverage} | {gauges} | {suppressed} | {active} |".format(
                    name=row["setting_name"],
                    mode=row.get("oracle_plumbing_mode") or "NA",
                    phi_lp=_fmt_ratio(row.get("phi_root_lp")),
                    phi_recon=_fmt_ratio(row.get("phi_root_reconstructed")),
                    oracle_root=_fmt_ratio(row.get("oracle_root_term")),
                    resid_root=_fmt_ratio(row.get("legacy_residual_root_term")),
                    coverage=coverage_label,
                    gauges=row.get("gauge_constraint_count"),
                    suppressed=row.get("rowgen_oracle_only_truncated_tuple_cuts_suppressed"),
                    active=active_label,
                )
            )
        lines.extend(
            [
                "",
                "## Gauged Regime Residual Diagnostics",
                "",
                "| Setting | Enabled | Feature bank | Selected features | Legacy rank | Selected rank | Regime root term | Truncated tuple cuts suppressed | Binding action legacy/candidate |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            selected = row.get("selected_regime_features") or []
            if isinstance(selected, list):
                selected_label = ", ".join(str(item) for item in selected) if selected else "none"
            else:
                selected_label = str(selected)
            binding = f"{row.get('legacy_root_binding_action')}/{row.get('candidate_root_binding_action')}"
            lines.append(
                "| {name} | {enabled} | {bank} | {selected} | {legacy_rank} | {selected_rank} | {root_term} | {suppressed} | {binding} |".format(
                    name=row["setting_name"],
                    enabled=row.get("gauged_regime_residual_enabled"),
                    bank=row.get("regime_feature_bank") or "NA",
                    selected=selected_label,
                    legacy_rank=row.get("legacy_signature_rank"),
                    selected_rank=row.get("selected_signature_rank"),
                    root_term=_fmt_ratio(row.get("regime_residual_root_term")),
                    suppressed=row.get("truncated_tuple_cuts_suppressed"),
                    binding=binding,
                )
            )
        if any(row.get("anchor") for row in rows):
            lines.extend(
                [
                    "",
                    "## V2 State-Pool Selector Diagnostics",
                    "",
                    "| Setting | Anchor | Selector | Weight mode | Dual available | Nonzero dual rows | Delta gap1 vs anchor | Delta ratio3 vs anchor | Delta ratio2 vs anchor | V1 base features | V2 new features | Weighted rows |",
                    "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
                ]
            )
            for row in rows:
                summary = row.get("v2_selection_payload_summary") or {}
                v1_features = summary.get("selected_v1_base_features") or row.get("selected_v1_base_features") or []
                v2_features = summary.get("selected_v2_features") or row.get("selected_v2_features") or []
                if isinstance(v1_features, list):
                    v1_label = ", ".join(str(item) for item in v1_features) if v1_features else "none"
                else:
                    v1_label = str(v1_features)
                if isinstance(v2_features, list):
                    v2_label = ", ".join(str(item) for item in v2_features) if v2_features else "none"
                else:
                    v2_label = str(v2_features)
                lines.append(
                    "| {name} | {anchor} | {selector} | {weight_mode} | {dual_avail} | {dual_rows} | {dgap} | {dr3} | {dr2} | {v1} | {v2} | {rows} |".format(
                        name=row["setting_name"],
                        anchor=summary.get("anchor") or row.get("regime_residual_anchor") or "NA",
                        selector=summary.get("selector_mode") or "NA",
                        weight_mode=summary.get("weight_mode") or "NA",
                        dual_avail=summary.get("dual_component_available"),
                        dual_rows=summary.get("nonzero_dual_row_count"),
                        dgap=_fmt_ratio(row.get("delta_gap1_vs_anchor")),
                        dr3=_fmt_ratio(row.get("delta_ratio3_vs_anchor")),
                        dr2=_fmt_ratio(row.get("delta_ratio2_vs_anchor")),
                        v1=v1_label,
                        v2=v2_label,
                        rows=summary.get("weighted_row_count"),
                    )
                )
    errors = payload.get("errors") or []
    if errors:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                "| Setting | Phase | Exact Status | Error |",
                "| --- | --- | --- | --- |",
            ]
        )
        for err in errors:
            lines.append(
                "| {setting} | {phase} | {status} | {error_type}: {error} |".format(
                    setting=err["setting_name"],
                    phase=err["phase"],
                    status=err.get("exact_solver_status", "Unknown"),
                    error_type=err["error_type"],
                    error=err["error"].replace("|", "\\|"),
                )
            )
    return "\n".join(lines) + "\n"


def _build_payload(
    *,
    manifest_path: str,
    settings: List[Setting],
    incumbent_spec: RunnerSpec,
    candidate_spec: RunnerSpec,
    rows: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    benchmark_tier: str,
    r2_threshold: float,
    r3_threshold: float,
    anchor_spec: Optional[RunnerSpec] = None,
) -> Dict[str, object]:
    return {
        "generated_at": _now_iso(),
        "manifest_path": manifest_path,
        "benchmark_tier": benchmark_tier,
        "settings": [setting.name for setting in settings],
        "benchmark_env": dict(FIXED_ENV),
        "incumbent_spec": {
            "label": incumbent_spec.label,
            "feature_bank": incumbent_spec.feature_bank,
            "selector_mode": incumbent_spec.selector_mode,
            "env": incumbent_spec.env,
        },
        "anchor_spec": (
            {
                "label": anchor_spec.label,
                "feature_bank": anchor_spec.feature_bank,
                "selector_mode": anchor_spec.selector_mode,
                "env": anchor_spec.env,
            }
            if anchor_spec is not None
            else None
        ),
        "candidate_spec": {
            "label": candidate_spec.label,
            "feature_bank": candidate_spec.feature_bank,
            "selector_mode": candidate_spec.selector_mode,
            "env": candidate_spec.env,
        },
        "thresholds": {
            "candidate_r2_max": float(r2_threshold),
            "r3_max": float(r3_threshold),
        },
        "rows": rows,
        "errors": errors,
        "summary": {
            "row_count": len(rows),
            "error_count": len(errors),
            "candidate_ratio5_better_count": sum(1 for row in rows if row["candidate"]["ratio5_better_than_ratio3"]),
            "incumbent_ratio5_better_count": sum(1 for row in rows if row["incumbent"]["ratio5_better_than_ratio3"]),
            "gate_exact_optimal_count": sum(1 for row in rows if row["gate_exact_optimal"]),
            "gate_ordering_count": sum(1 for row in rows if row["gate_myopic_lt_stop_lt_exact"]),
            "gate_candidate_r2_count": sum(1 for row in rows if row["gate_candidate_r2_le_threshold"]),
            "gate_r3_count": sum(1 for row in rows if row["gate_r3_le_threshold"]),
            "target_success_count": sum(1 for row in rows if row["gate_all_target_conditions"]),
        },
    }


def _write_outputs(
    *,
    output_json: Path,
    output_md: Path,
    incumbent_spec: RunnerSpec,
    candidate_spec: RunnerSpec,
    rows: List[Dict[str, object]],
    payload: Mapping[str, object],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(
        _build_markdown(
            incumbent=incumbent_spec,
            candidate=candidate_spec,
            rows=rows,
            payload=payload,
        ),
        encoding="utf-8",
    )


def run_settings(
    *,
    settings: List[Setting],
    manifest_path: str,
    incumbent_spec: RunnerSpec,
    candidate_spec: RunnerSpec,
    anchor_spec: Optional[RunnerSpec] = None,
    selection_dir: Optional[Path] = None,
    output_json: Path,
    output_md: Path,
    continue_on_error: bool = False,
    r2_threshold: float = 0.2,
    r3_threshold: float = 0.8,
    benchmark_tier: str = "multigene_ratio45",
    progress_prefix: str = "multigene-ratio45",
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    payload = _build_payload(
        manifest_path=manifest_path,
        settings=settings,
        incumbent_spec=incumbent_spec,
        candidate_spec=candidate_spec,
        rows=rows,
        errors=errors,
        benchmark_tier=benchmark_tier,
        r2_threshold=r2_threshold,
        r3_threshold=r3_threshold,
        anchor_spec=anchor_spec,
    )
    for setting in settings:
        phase = "incumbent_run"
        incumbent_solver_meta = _exact_solver_meta(FIXED_ENV)
        try:
            incumbent_metrics, incumbent_raw, incumbent_solver_meta = _run_setting(
                setting,
                incumbent_spec,
                benchmark_tier=benchmark_tier,
                progress_prefix=progress_prefix,
            )
            phase = "myopic_eval"
            myopic = _compute_myopic_value(setting, incumbent_raw)
            anchor_metrics = None
            anchor_raw = None
            selection_payload = None
            selection_payload_path = None
            effective_candidate_spec = candidate_spec
            if anchor_spec is not None:
                phase = "anchor_run"
                anchor_metrics, anchor_raw, _anchor_solver_meta = _run_setting(
                    setting,
                    anchor_spec,
                    benchmark_tier=benchmark_tier,
                    progress_prefix=progress_prefix,
                )
                phase = "v2_selection"
                output_selection_dir = selection_dir or (
                    output_json.parent / "gauged_regime_residual_adp_v2_selection_20260428"
                )
                selection_payload_path = safe_artifact_path(
                    output_selection_dir,
                    (setting.name, anchor_spec.label),
                    suffix="_selection.json",
                )
                anchor_mode = str(candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_ANCHOR", "v1")).strip().lower()
                feature_bank = resolve_feature_bank(
                    candidate_spec.feature_bank or candidate_spec.env.get("GAUGED_REGIME_FEATURE_BANK"),
                    require=True,
                )
                feature_semantics = feature_semantics_for_bank(feature_bank)
                selection_payload = _make_v2_selection_payload(
                    setting=setting,
                    anchor_results=anchor_raw,
                    anchor_mode=anchor_mode,
                    payload_path=selection_payload_path,
                    top_k=int(candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_V2_TOP_K", "5")),
                    min_ratio=float(candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_V2_MIN_SIGNATURE_RATIO", "0.10")),
                    incremental_tol=float(candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_V2_INCREMENTAL_TOL", "1e-8")),
                    feature_bank=feature_bank,
                    feature_semantics=feature_semantics,
                    selector_mode=(
                        candidate_spec.selector_mode
                        or candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_SELECTOR")
                        or "STATEFALLBACK"
                    ),
                    weight_mode=candidate_spec.env.get("GAUGED_REGIME_RESIDUAL_V2_WEIGHT_MODE"),
                    dual_component_required=_env_enabled(
                        candidate_spec.env,
                        "GAUGED_REGIME_RESIDUAL_V2_DUAL_REQUIRED",
                    ),
                )
                candidate_env = dict(candidate_spec.env)
                candidate_env["GAUGED_REGIME_RESIDUAL_V2_PAYLOAD_PATH"] = str(selection_payload_path)
                effective_candidate_spec = RunnerSpec(
                    label=candidate_spec.label,
                    env=candidate_env,
                    feature_bank=candidate_spec.feature_bank,
                    selector_mode=candidate_spec.selector_mode,
                )
            phase = "candidate_run"
            candidate_metrics, _candidate_raw, candidate_solver_meta = _run_setting(
                setting,
                effective_candidate_spec,
                benchmark_tier=benchmark_tier,
                progress_prefix=progress_prefix,
            )
        except Exception as exc:
            exact_solver_status = _extract_exact_solver_status(str(exc)) or "Unknown"
            errors.append(
                {
                    "setting_name": setting.name,
                    "family": setting.family,
                    "preset": setting.preset,
                    "allele_freqs": dict(setting.allele_freqs),
                    "a_scale": setting.a_scale,
                    "b_scale": setting.b_scale,
                    "fixed_cost": setting.fixed_cost,
                    "variable_cost": setting.variable_cost,
                    "delta_shift": setting.delta_shift,
                    "phase": phase,
                    "exact_solver_mode": incumbent_solver_meta["exact_solver_mode"],
                    "exact_dual_lp_backend": incumbent_solver_meta["exact_dual_lp_backend"],
                    "exact_solver_status": exact_solver_status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            payload = _build_payload(
                manifest_path=manifest_path,
                settings=settings,
                incumbent_spec=incumbent_spec,
                candidate_spec=candidate_spec,
                rows=rows,
                errors=errors,
                benchmark_tier=benchmark_tier,
                r2_threshold=r2_threshold,
                r3_threshold=r3_threshold,
                anchor_spec=anchor_spec,
            )
            _write_outputs(
                output_json=output_json,
                output_md=output_md,
                incumbent_spec=incumbent_spec,
                candidate_spec=candidate_spec,
                rows=rows,
                payload=payload,
            )
            if continue_on_error:
                continue
            raise

        candidate_solver_status = "Optimal"
        rows.append(
            _build_row(
                setting=setting,
                incumbent_metrics=incumbent_metrics,
                candidate_metrics=candidate_metrics,
                myopic=myopic,
                solver_meta=incumbent_solver_meta | {
                    "candidate_exact_solver_status": candidate_solver_status,
                    "candidate_exact_dual_lp_backend": candidate_solver_meta["exact_dual_lp_backend"],
                },
                r2_threshold=r2_threshold,
                r3_threshold=r3_threshold,
                anchor_metrics=anchor_metrics,
                selection_payload_path=selection_payload_path,
                selection_payload=selection_payload,
            )
        )
        payload = _build_payload(
            manifest_path=manifest_path,
            settings=settings,
            incumbent_spec=incumbent_spec,
            candidate_spec=candidate_spec,
            rows=rows,
            errors=errors,
            benchmark_tier=benchmark_tier,
            r2_threshold=r2_threshold,
            r3_threshold=r3_threshold,
            anchor_spec=anchor_spec,
        )
        _write_outputs(
            output_json=output_json,
            output_md=output_md,
            incumbent_spec=incumbent_spec,
            candidate_spec=candidate_spec,
            rows=rows,
            payload=payload,
        )
    if not settings:
        _write_outputs(
            output_json=output_json,
            output_md=output_md,
            incumbent_spec=incumbent_spec,
            candidate_spec=candidate_spec,
            rows=rows,
            payload=payload,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ratio4/ratio5 reruns on the newly found multigene settings.")
    parser.add_argument(
        "--manifest",
        default="documentation/multigene_ratio45_new_settings_manifest_20260422.json",
        help="JSON manifest with settings",
    )
    parser.add_argument(
        "--incumbent-spec",
        default=None,
        help="JSON spec file with {label, env}; defaults to legacy stage baseline",
    )
    parser.add_argument(
        "--candidate-spec",
        default=".claude/worktrees/bellman-active-rollout-20260420/documentation/bellman_active_rollout_candidate_policy_only_20260421.json",
        help="JSON spec file with {label, env}",
    )
    parser.add_argument(
        "--anchor-spec",
        default=None,
        help="optional JSON spec used to generate per-row V2 state-pool selection payloads",
    )
    parser.add_argument(
        "--selection-dir",
        default=None,
        help="directory for generated per-row V2 selection payloads",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="path for JSON output",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        help="path for Markdown output",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record per-setting failures and continue running the remaining settings",
    )
    parser.add_argument(
        "--setting-names",
        default="",
        help="optional comma-separated setting names to run from the manifest",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.2,
        help="target threshold for candidate ratio2",
    )
    parser.add_argument(
        "--r3-threshold",
        type=float,
        default=0.8,
        help="target threshold for ratio3",
    )
    args = parser.parse_args()

    settings = _load_settings(args.manifest)
    if args.setting_names.strip():
        requested = {name.strip() for name in args.setting_names.split(",") if name.strip()}
        settings = [setting for setting in settings if setting.name in requested]
        missing = requested - {setting.name for setting in settings}
        if missing:
            raise ValueError(f"Requested setting names not found in manifest: {sorted(missing)}")
    incumbent_spec = _load_spec(
        args.incumbent_spec,
        default_label="legacy_stage_incumbent",
        default_env=DEFAULT_INCUMBENT_ENV,
    )
    candidate_spec = _load_spec(
        args.candidate_spec,
        default_label="candidate",
    )
    anchor_spec = _load_spec(args.anchor_spec, default_label="anchor") if args.anchor_spec else None

    run_settings(
        settings=settings,
        manifest_path=args.manifest,
        incumbent_spec=incumbent_spec,
        candidate_spec=candidate_spec,
        anchor_spec=anchor_spec,
        selection_dir=Path(args.selection_dir) if args.selection_dir else None,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        continue_on_error=args.continue_on_error,
        r2_threshold=args.r2_threshold,
        r3_threshold=args.r3_threshold,
    )


if __name__ == "__main__":
    main()
