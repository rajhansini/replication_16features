from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx

from ..models.belief import InferenceResult

LEGACY_MODELS = {"legacy", "legacy_stage", "legacy_scalar"}
FROZEN_MODELS = {
    "simplex_quad",
    "entropy_spline",
    "softmax_testscore",
    "pair_product",
    "deepsets",
}
VALID_MODELS = LEGACY_MODELS | FROZEN_MODELS

ROLE_BUCKETS = {
    "founder": 0.0,
    "internal": 1.0,
    "leaf": 2.0,
    "isolated": 3.0,
}


def resolve_theta_model(theta_model: Optional[str] = None) -> Optional[str]:
    raw = theta_model if theta_model is not None else os.getenv("THETA_MODEL", "")
    model = raw.strip().lower()
    if model in {"", "none", "off"}:
        return None
    if model not in VALID_MODELS:
        raise ValueError(
            "Unknown THETA_MODEL="
            f"{model!r} (expected one of: {', '.join(sorted(VALID_MODELS))})."
        )
    return model


def resolve_theta_mode(theta_mode: Optional[str] = None, theta_model: Optional[str] = None) -> str:
    model = resolve_theta_model(theta_model)
    if model == "legacy_scalar":
        return "scalar"
    if model in FROZEN_MODELS or model == "legacy_stage":
        return "stage"
    raw = theta_mode if theta_mode is not None else os.getenv("THETA_MODE", "scalar")
    return raw.strip().lower()


def resolve_theta_model_spec_path(spec_path: Optional[str] = None) -> Optional[str]:
    raw = spec_path if spec_path is not None else os.getenv("THETA_MODEL_SPEC_PATH", "")
    path = raw.strip()
    return path or None


@lru_cache(maxsize=32)
def _load_spec_from_path(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Theta model spec must be a JSON object: {path}")
    return payload


def load_theta_model_spec(theta_model_spec_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = resolve_theta_model_spec_path(theta_model_spec_path)
    if not path:
        return None
    return dict(_load_spec_from_path(path))


def theta_model_signature(theta_model: Optional[str] = None, theta_model_spec_path: Optional[str] = None) -> Optional[str]:
    model = resolve_theta_model(theta_model)
    if model is None:
        return None
    payload: Dict[str, Any] = {"theta_model": model}
    spec = load_theta_model_spec(theta_model_spec_path)
    if spec is not None:
        payload["theta_model_spec"] = spec
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def theta_model_metadata(theta_model: Optional[str] = None, theta_model_spec_path: Optional[str] = None) -> Dict[str, Any]:
    model = resolve_theta_model(theta_model)
    path = resolve_theta_model_spec_path(theta_model_spec_path)
    spec = load_theta_model_spec(path) if path else None
    return {
        "theta_model": model,
        "theta_model_spec_path": str(Path(path).expanduser()) if path else None,
        "theta_model_signature": theta_model_signature(model, path),
        "theta_model_spec": spec,
    }


def theta_model_correction_active(theta_model: Optional[str] = None) -> bool:
    model = resolve_theta_model(theta_model)
    return model in FROZEN_MODELS


def evaluate_theta_model_correction(
    s,
    *,
    belief,
    individuals,
    gen_states,
    pedigree=None,
    theta_model: Optional[str] = None,
    theta_model_spec: Optional[Mapping[str, Any]] = None,
    theta_model_spec_path: Optional[str] = None,
) -> float:
    model = resolve_theta_model(theta_model)
    if model is None or model in LEGACY_MODELS:
        return 0.0

    spec = dict(theta_model_spec) if theta_model_spec is not None else load_theta_model_spec(theta_model_spec_path)
    if spec is None:
        raise ValueError(f"THETA_MODEL={model!r} requires THETA_MODEL_SPEC_PATH or explicit theta_model_spec.")
    spec_model = str(spec.get("model", model)).strip().lower()
    if spec_model != model:
        raise ValueError(
            f"Theta model spec mismatch: env/model={model!r} spec.model={spec_model!r}"
        )

    if len(_evidence_state(s)) >= len(individuals):
        return 0.0

    features = _state_features(
        s,
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        pedigree=pedigree,
    )

    if model == "simplex_quad":
        value = _eval_simplex_quad(spec, features)
    elif model == "entropy_spline":
        value = _eval_entropy_spline(spec, features)
    elif model == "softmax_testscore":
        value = _eval_softmax_testscore(spec, features)
    elif model == "pair_product":
        value = _eval_pair_product(spec, features)
    elif model == "deepsets":
        value = _eval_deepsets(spec, features)
    else:
        raise ValueError(f"Unsupported THETA_MODEL for correction evaluation: {model!r}")

    cap = float(spec.get("output_abs_cap", 0.0) or 0.0)
    if cap > 0.0:
        value = _clip(value, cap)
    return float(value)


def _evidence_state(state):
    if not isinstance(state, frozenset):
        raise AssertionError(
            f"State must be evidence-only frozenset[(person,outcome)], got {type(state).__name__}: {state!r}"
        )
    return state


def _posterior_entry(belief, state):
    posterior_entry, _ = belief[state]
    return posterior_entry


def _marginals(posterior_entry):
    if isinstance(posterior_entry, InferenceResult):
        return posterior_entry.marginals
    return posterior_entry


def _carrier_prob(person_probs: Mapping[int, float]) -> float:
    return float(person_probs.get(1, 0.0) + person_probs.get(2, 0.0))


def _normalized_entropy(person_probs: Mapping[int, float]) -> float:
    entropy = 0.0
    for state_prob in person_probs.values():
        prob = float(state_prob)
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    if not person_probs:
        return 0.0
    return float(entropy / math.log(max(2, len(person_probs))))


def _state_features(
    state,
    *,
    belief,
    individuals,
    gen_states,
    pedigree=None,
) -> Dict[str, Any]:
    del gen_states
    evidence = _evidence_state(state)
    tested = {person for person, _ in evidence}
    untested = [person for person in individuals if person not in tested]
    posterior_entry = _posterior_entry(belief, state)
    p_s = _marginals(posterior_entry)

    carrier_probs: Dict[str, float] = {}
    entropies: Dict[str, float] = {}
    for person in untested:
        person_probs = p_s.get(person, {})
        carrier_probs[person] = _carrier_prob(person_probs)
        entropies[person] = _normalized_entropy(person_probs)

    n = max(1, len(individuals))
    stage = len(evidence)
    tested_fraction = float(stage / n)
    carrier_values = sorted((carrier_probs[p] for p in untested), reverse=True)
    entropy_values = [entropies[p] for p in untested]

    pedigree_summary = _pedigree_summary(individuals=individuals, pedigree=pedigree)
    per_person_features = []
    for person in untested:
        role_bucket = pedigree_summary["role_bucket"].get(person, ROLE_BUCKETS["isolated"])
        depth = pedigree_summary["depth"].get(person, 0.0)
        per_person_features.append(
            {
                "person": person,
                "carrier_prob": carrier_probs[person],
                "entropy": entropies[person],
                "depth": depth,
                "role_bucket": role_bucket,
                "tested_fraction": tested_fraction,
            }
        )

    return {
        "stage": stage,
        "tested_fraction": tested_fraction,
        "untested_people": untested,
        "carrier_probs": carrier_probs,
        "entropies": entropies,
        "top_probs": carrier_values,
        "mean_carrier_prob": _mean(carrier_values),
        "mean_carrier_prob_sq": _mean([value * value for value in carrier_values]),
        "mean_entropy": _mean(entropy_values),
        "mean_entropy_sq": _mean([value * value for value in entropy_values]),
        "per_person_features": per_person_features,
        "pedigree_summary": pedigree_summary,
    }


def _pedigree_summary(*, individuals, pedigree=None) -> Dict[str, Any]:
    default_depth = {person: 0.0 for person in individuals}
    default_role = {person: ROLE_BUCKETS["isolated"] for person in individuals}
    if pedigree is None or not hasattr(pedigree, "graph"):
        return {
            "depth": default_depth,
            "role_bucket": default_role,
            "parent_child_edges": [],
            "coparent_pairs": [],
        }

    graph = pedigree.graph
    indegrees = dict(graph.in_degree())
    outdegrees = dict(graph.out_degree())

    depth = {}
    for person in individuals:
        if indegrees.get(person, 0) == 0:
            depth[person] = 0.0
            continue
        best = 0
        try:
            for founder in pedigree.get_founders():
                if founder == person:
                    best = max(best, 0)
                    continue
                if founder in graph and person in graph:
                    try:
                        best = max(best, nx.shortest_path_length(graph, founder, person))
                    except StopIteration:
                        continue
                    except Exception:
                        continue
        except Exception:
            best = 0
        depth[person] = float(max(0, best - 1))

    role_bucket = {}
    for person in individuals:
        indeg = indegrees.get(person, 0)
        outdeg = outdegrees.get(person, 0)
        if indeg == 0 and outdeg > 0:
            role_bucket[person] = ROLE_BUCKETS["founder"]
        elif indeg > 0 and outdeg > 0:
            role_bucket[person] = ROLE_BUCKETS["internal"]
        elif indeg > 0 and outdeg == 0:
            role_bucket[person] = ROLE_BUCKETS["leaf"]
        else:
            role_bucket[person] = ROLE_BUCKETS["isolated"]

    parent_child_edges = []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            parent_child_edges.append((parent, child, "pc"))

    coparent_pairs = []
    for child in pedigree.get_offspring():
        parents = sorted(pedigree.get_parents(child))
        if len(parents) == 2:
            coparent_pairs.append((parents[0], parents[1], "cp"))

    return {
        "depth": depth,
        "role_bucket": role_bucket,
        "parent_child_edges": parent_child_edges,
        "coparent_pairs": coparent_pairs,
    }


def _state_vector(features: Mapping[str, Any]) -> list[float]:
    top_probs = list(features.get("top_probs", []))
    p1 = top_probs[0] if len(top_probs) >= 1 else 0.0
    return [
        float(features.get("tested_fraction", 0.0)),
        float(features.get("mean_carrier_prob", 0.0)),
        float(features.get("mean_carrier_prob_sq", 0.0)),
        float(p1),
        float(features.get("mean_entropy", 0.0)),
        float(features.get("mean_entropy_sq", 0.0)),
        float(p1 * float(features.get("mean_entropy", 0.0))),
    ]


def _eval_simplex_quad(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    z = _state_vector(features)
    root_z = [float(x) for x in spec.get("root_z", [0.0] * len(z))]
    centered = [value - root_z[idx] for idx, value in enumerate(z)]
    atoms = []
    atoms.extend(centered)
    atoms.extend(value * value for value in centered)
    for left in range(len(centered)):
        for right in range(left + 1, len(centered)):
            atoms.append(centered[left] * centered[right])
    weights = [float(x) for x in spec.get("weights", [])]
    if len(weights) != len(atoms):
        raise ValueError(
            f"simplex_quad weights length mismatch: expected {len(atoms)} got {len(weights)}"
        )
    if weights:
        total = sum(weights)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"simplex_quad weights must sum to 1.0, got {total}")
        if any(weight < -1e-12 for weight in weights):
            raise ValueError("simplex_quad weights must be non-negative")
    return float(sum(weight * atom for weight, atom in zip(weights, atoms)))


def _eval_entropy_spline(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    beta = spec.get("beta")
    if not isinstance(beta, Sequence) or len(beta) != 4:
        raise ValueError("entropy_spline spec.beta must be a 4x4x4 tensor")
    k_basis = _bernstein_basis(float(features.get("tested_fraction", 0.0)))
    h_basis = _bernstein_basis(float(features.get("mean_entropy", 0.0)))
    p_basis = _bernstein_basis(float(features.get("mean_carrier_prob", 0.0)))
    value = 0.0
    for a_idx in range(4):
        for b_idx in range(4):
            for c_idx in range(4):
                value += (
                    float(beta[a_idx][b_idx][c_idx])
                    * k_basis[a_idx]
                    * h_basis[b_idx]
                    * p_basis[c_idx]
                )
    return float(value)


def _eval_softmax_testscore(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    weights = [float(x) for x in spec.get("w", [])]
    if len(weights) != 5:
        raise ValueError("softmax_testscore spec.w must have length 5")
    tau = max(1e-6, float(spec.get("tau", 1.0)))
    gamma = float(spec.get("gamma", 1.0))
    root_constant = float(spec.get("root_constant", 0.0))
    untested = features.get("per_person_features", [])
    if not untested:
        return -root_constant
    logits = []
    for entry in untested:
        x_i = [
            float(entry.get("carrier_prob", 0.0)),
            float(entry.get("entropy", 0.0)),
            float(entry.get("depth", 0.0)),
            float(entry.get("role_bucket", 0.0)),
            1.0,
        ]
        logits.append(sum(weight * value for weight, value in zip(weights, x_i)))
    exp_terms = [math.exp(logit / tau) for logit in logits]
    pooled = tau * math.log(sum(exp_terms) / max(1, len(exp_terms)))
    return float(gamma * pooled - root_constant)


def _eval_pair_product(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    alpha = {str(k): float(v) for k, v in dict(spec.get("alpha", {})).items()}
    beta = {str(k): float(v) for k, v in dict(spec.get("beta", {})).items()}
    gamma = {str(k): float(v) for k, v in dict(spec.get("gamma", {})).items()}
    root_constant = float(spec.get("root_constant", 0.0))
    carrier = features.get("carrier_probs", {})
    entropy = features.get("entropies", {})
    summary = features.get("pedigree_summary", {})
    value = 0.0
    for parent, child, bucket in summary.get("parent_child_edges", []):
        value += alpha.get(bucket, 0.0) * float(carrier.get(parent, 0.0)) * float(carrier.get(child, 0.0))
        value += gamma.get(bucket, 0.0) * float(entropy.get(parent, 0.0)) * float(entropy.get(child, 0.0))
    for left, right, bucket in summary.get("coparent_pairs", []):
        value += beta.get(bucket, 0.0) * float(carrier.get(left, 0.0)) * float(carrier.get(right, 0.0))
    return float(value - root_constant)


def _eval_deepsets(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    W1 = spec.get("W1", [])
    b1 = [float(x) for x in spec.get("b1", [])]
    W2 = spec.get("W2", [])
    b2 = [float(x) for x in spec.get("b2", [])]
    V1 = spec.get("V1", [])
    c1 = [float(x) for x in spec.get("c1", [])]
    v2 = [float(x) for x in spec.get("v2", [])]
    c2 = float(spec.get("c2", 0.0))
    root_constant = float(spec.get("root_constant", 0.0))

    def _phi(x_i):
        hidden = _tanh_vec(_affine(W1, b1, x_i))
        return _affine(W2, b2, hidden)

    latent = None
    for entry in features.get("per_person_features", []):
        x_i = [
            float(entry.get("carrier_prob", 0.0)),
            float(entry.get("entropy", 0.0)),
            float(entry.get("depth", 0.0)),
            float(entry.get("role_bucket", 0.0)),
            float(entry.get("tested_fraction", 0.0)),
            1.0,
        ]
        contrib = _phi(x_i)
        if latent is None:
            latent = contrib
        else:
            latent = [left + right for left, right in zip(latent, contrib)]
    if latent is None:
        latent = [0.0 for _ in range(len(V1[0]) if V1 else len(v2))]
    pooled = _tanh_vec(_affine(V1, c1, latent))
    return float(sum(weight * value for weight, value in zip(v2, pooled)) + c2 - root_constant)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _bernstein_basis(value: float) -> list[float]:
    x = max(0.0, min(1.0, value))
    return [
        (1.0 - x) ** 3,
        3.0 * x * (1.0 - x) ** 2,
        3.0 * (x**2) * (1.0 - x),
        x**3,
    ]


def _affine(matrix: Sequence[Sequence[float]], bias: Sequence[float], vector: Sequence[float]) -> list[float]:
    if not matrix:
        return [float(x) for x in bias]
    output = []
    for row_idx, row in enumerate(matrix):
        total = float(bias[row_idx]) if row_idx < len(bias) else 0.0
        for weight, value in zip(row, vector):
            total += float(weight) * float(value)
        output.append(total)
    return output


def _tanh_vec(values: Sequence[float]) -> list[float]:
    return [math.tanh(float(value)) for value in values]
