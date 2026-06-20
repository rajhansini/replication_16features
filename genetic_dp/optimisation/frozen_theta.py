from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..models.belief import InferenceResult, lift_single_gene_posteriors_to_genes


THETA_MODEL_CHOICES = (
    "legacy",
    "simplex_quad",
    "entropy_spline",
    "softmax_testscore",
    "pair_product",
    "deepsets",
)


def resolve_theta_model(theta_model: Optional[str] = None, theta_star=None) -> str:
    if theta_model:
        model = theta_model.strip().lower()
    elif isinstance(theta_star, Mapping) and isinstance(theta_star.get("model"), str):
        model = str(theta_star.get("model")).strip().lower()
    else:
        model = os.getenv("THETA_MODEL", "legacy").strip().lower() or "legacy"
    if model not in THETA_MODEL_CHOICES:
        raise ValueError(
            "Unknown THETA_MODEL="
            f"{model!r} (expected one of: {', '.join(THETA_MODEL_CHOICES)})"
        )
    return model


@lru_cache(maxsize=8)
def _load_theta_spec_from_path(path_str: str) -> dict:
    path = Path(path_str).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen theta spec must be a JSON object, got {type(payload).__name__}.")
    return payload


def resolve_theta_spec(theta_star=None, theta_model: Optional[str] = None) -> Optional[dict]:
    model = resolve_theta_model(theta_model=theta_model, theta_star=theta_star)
    if model == "legacy":
        return None
    if isinstance(theta_star, Mapping):
        payload = dict(theta_star)
        payload.setdefault("model", model)
        return payload
    path = os.getenv("THETA_MODEL_SPEC_PATH", "").strip()
    if not path:
        raise ValueError(
            "Frozen theta model requested but THETA_MODEL_SPEC_PATH is unset "
            "and no inline theta spec was provided."
        )
    payload = dict(_load_theta_spec_from_path(path))
    payload.setdefault("model", model)
    return payload


def _posterior_entry_from_belief_entry(entry):
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[0]
    return entry


def _posterior_maps(
    posterior_entry,
    *,
    individuals: Sequence[str],
    genes: Optional[Sequence[str]] = None,
):
    gene_list = tuple(genes) if genes else tuple()
    if isinstance(posterior_entry, InferenceResult):
        p_state = posterior_entry.marginals
        per_gene_probs = posterior_entry.get_per_gene_probs() if gene_list else None
    else:
        p_state = posterior_entry
        per_gene_probs = None
    if gene_list:
        if not per_gene_probs:
            per_gene_probs = lift_single_gene_posteriors_to_genes(p_state, gene_list)
        return p_state, per_gene_probs
    return p_state, None


def _person_prob_vector(
    person: str,
    *,
    p_state,
    per_gene_probs,
    gene_list: Sequence[str],
) -> np.ndarray:
    if gene_list:
        rows = []
        for gene in gene_list:
            gene_probs = (per_gene_probs or {}).get(gene, {})
            dist = gene_probs.get(person)
            if not isinstance(dist, Mapping):
                continue
            rows.append(
                np.array(
                    [
                        float(dist.get(0, 0.0)),
                        float(dist.get(1, 0.0)),
                        float(dist.get(2, 0.0)),
                    ],
                    dtype=float,
                )
            )
        if rows:
            stacked = np.vstack(rows)
            return np.clip(np.mean(stacked, axis=0), 0.0, 1.0)
    person_dist = p_state.get(person, {}) if isinstance(p_state, Mapping) else {}
    return np.array(
        [
            float(person_dist.get(0, 0.0)),
            float(person_dist.get(1, 0.0)),
            float(person_dist.get(2, 0.0)),
        ],
        dtype=float,
    )


def _safe_entropy(q: np.ndarray) -> float:
    q_safe = np.clip(q, 1e-12, 1.0)
    return float(-np.sum(q_safe * np.log(q_safe)))


def _person_summary_rows(
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    genes: Optional[Sequence[str]] = None,
):
    p_state, per_gene_probs = _posterior_maps(
        posterior_entry,
        individuals=individuals,
        genes=genes,
    )
    tested_people = {person for person, _ in evidence_state}
    gene_list = tuple(genes) if genes else tuple()
    rows = []
    for person in individuals:
        q = _person_prob_vector(
            person,
            p_state=p_state,
            per_gene_probs=per_gene_probs,
            gene_list=gene_list,
        )
        total = float(np.sum(q))
        if total > 0.0:
            q = q / total
        p12 = float(q[1] + q[2])
        entropy = _safe_entropy(q)
        impurity = float(1.0 - np.sum(q**2))
        tested = 1.0 if person in tested_people else 0.0
        rows.append(
            {
                "person": person,
                "q": q,
                "p12": p12,
                "entropy": entropy,
                "impurity": impurity,
                "tested": tested,
                "untested": 1.0 - tested,
            }
        )
    return rows


def _global_stage_features(evidence_state: frozenset, individuals: Sequence[str]) -> list[float]:
    n = max(1, len(individuals))
    stage = len(evidence_state)
    untested = len(individuals) - stage
    stage_frac = float(stage) / float(n)
    untested_frac = float(untested) / float(n)
    return [
        1.0,
        float(stage),
        float(stage * stage),
        float(untested),
        stage_frac,
        untested_frac,
    ]


def _simplex_quad_features(
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    genes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    rows = _person_summary_rows(
        evidence_state=evidence_state,
        posterior_entry=posterior_entry,
        individuals=individuals,
        genes=genes,
    )
    features = _global_stage_features(evidence_state, individuals)
    for group_name in ("all", "tested", "untested"):
        bucket = []
        for row in rows:
            if group_name == "tested" and row["tested"] < 0.5:
                continue
            if group_name == "untested" and row["untested"] < 0.5:
                continue
            bucket.append(row)
        if not bucket:
            features.extend([0.0] * 9)
            continue
        q = np.vstack([row["q"] for row in bucket])
        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        features.extend(
            [
                float(np.sum(q0)),
                float(np.sum(q1)),
                float(np.sum(q2)),
                float(np.sum(q0**2)),
                float(np.sum(q1**2)),
                float(np.sum(q2**2)),
                float(np.sum(q0 * q1)),
                float(np.sum(q0 * q2)),
                float(np.sum(q1 * q2)),
            ]
        )
    return np.asarray(features, dtype=float)


def _entropy_spline_features(
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    genes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    rows = _person_summary_rows(
        evidence_state=evidence_state,
        posterior_entry=posterior_entry,
        individuals=individuals,
        genes=genes,
    )
    features = _global_stage_features(evidence_state, individuals)
    knots = (0.25, 0.5, 0.75)
    for group_name in ("all", "tested", "untested"):
        bucket = []
        for row in rows:
            if group_name == "tested" and row["tested"] < 0.5:
                continue
            if group_name == "untested" and row["untested"] < 0.5:
                continue
            bucket.append(row)
        if not bucket:
            features.extend([0.0] * (4 + len(knots)))
            continue
        p12 = np.asarray([row["p12"] for row in bucket], dtype=float)
        entropy = np.asarray([row["entropy"] for row in bucket], dtype=float)
        impurity = np.asarray([row["impurity"] for row in bucket], dtype=float)
        features.extend(
            [
                float(np.sum(p12)),
                float(np.sum(p12**2)),
                float(np.sum(entropy)),
                float(np.sum(impurity)),
            ]
        )
        for knot in knots:
            features.append(float(np.sum(np.maximum(p12 - knot, 0.0))))
    return np.asarray(features, dtype=float)


def _pair_product_features(
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    genes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    rows = _person_summary_rows(
        evidence_state=evidence_state,
        posterior_entry=posterior_entry,
        individuals=individuals,
        genes=genes,
    )
    features = _global_stage_features(evidence_state, individuals)
    for group_name in ("all", "untested"):
        bucket = []
        for row in rows:
            if group_name == "untested" and row["untested"] < 0.5:
                continue
            bucket.append(row)
        if not bucket:
            features.extend([0.0] * 7)
            continue
        p12 = np.asarray([row["p12"] for row in bucket], dtype=float)
        entropy = np.asarray([row["entropy"] for row in bucket], dtype=float)
        impurity = np.asarray([row["impurity"] for row in bucket], dtype=float)
        pair_p12 = 0.5 * ((np.sum(p12) ** 2) - np.sum(p12**2))
        pair_entropy = 0.5 * ((np.sum(entropy) ** 2) - np.sum(entropy**2))
        pair_impurity = 0.5 * ((np.sum(impurity) ** 2) - np.sum(impurity**2))
        pair_mix = (np.sum(p12) * np.sum(entropy)) - np.sum(p12 * entropy)
        features.extend(
            [
                float(np.sum(p12)),
                float(np.sum(entropy)),
                float(np.sum(impurity)),
                float(pair_p12),
                float(pair_entropy),
                float(pair_impurity),
                float(pair_mix),
            ]
        )
    return np.asarray(features, dtype=float)


def _base_person_vector(row: Mapping[str, object]) -> np.ndarray:
    q = np.asarray(row["q"], dtype=float)
    return np.asarray(
        [
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(row["p12"]),
            float(row["entropy"]),
            float(row["impurity"]),
            float(row["tested"]),
            float(row["untested"]),
        ],
        dtype=float,
    )


def _deepsets_features(
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    spec: Mapping[str, object],
    genes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    rows = _person_summary_rows(
        evidence_state=evidence_state,
        posterior_entry=posterior_entry,
        individuals=individuals,
        genes=genes,
    )
    hidden_weights = np.asarray(spec.get("hidden_weights", []), dtype=float)
    hidden_bias = np.asarray(spec.get("hidden_bias", []), dtype=float)
    if hidden_weights.ndim != 2 or hidden_bias.ndim != 1:
        raise ValueError("DeepSets spec requires 2D hidden_weights and 1D hidden_bias.")
    hidden_total = np.zeros(hidden_bias.shape[0], dtype=float)
    for row in rows:
        z = _base_person_vector(row)
        hidden_total += np.tanh(hidden_weights @ z + hidden_bias)
    n = max(1.0, float(len(rows)))
    features = _global_stage_features(evidence_state, individuals)
    features.extend(hidden_total.tolist())
    features.extend((hidden_total / n).tolist())
    return np.asarray(features, dtype=float)


def theta_feature_vector(
    model: str,
    *,
    evidence_state: frozenset,
    posterior_entry,
    individuals: Sequence[str],
    gen_states: Sequence[int],
    spec: Optional[Mapping[str, object]] = None,
    genes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    del gen_states  # current frozen theta models only use posterior probabilities.
    if model == "simplex_quad":
        return _simplex_quad_features(
            evidence_state=evidence_state,
            posterior_entry=posterior_entry,
            individuals=individuals,
            genes=genes,
        )
    if model == "entropy_spline":
        return _entropy_spline_features(
            evidence_state=evidence_state,
            posterior_entry=posterior_entry,
            individuals=individuals,
            genes=genes,
        )
    if model == "pair_product":
        return _pair_product_features(
            evidence_state=evidence_state,
            posterior_entry=posterior_entry,
            individuals=individuals,
            genes=genes,
        )
    if model == "deepsets":
        if spec is None:
            raise ValueError("DeepSets feature construction requires a spec.")
        return _deepsets_features(
            evidence_state=evidence_state,
            posterior_entry=posterior_entry,
            individuals=individuals,
            spec=spec,
            genes=genes,
        )
    raise ValueError(f"theta_feature_vector does not support model={model!r}")


def evaluate_frozen_theta(
    s,
    *,
    theta_star,
    individuals: Sequence[str],
    gen_states: Sequence[int],
    belief_entry=None,
    genes: Optional[Sequence[str]] = None,
    theta_model: Optional[str] = None,
) -> float:
    model = resolve_theta_model(theta_model=theta_model, theta_star=theta_star)
    if model == "legacy":
        raise ValueError("evaluate_frozen_theta called for legacy theta.")
    spec = resolve_theta_spec(theta_star=theta_star, theta_model=model)
    if not isinstance(s, frozenset):
        raise AssertionError(
            f"State must be evidence-only frozenset[(person,outcome)], got {type(s).__name__}: {s!r}"
        )
    if len(s) >= len(individuals):
        return 0.0
    posterior_entry = _posterior_entry_from_belief_entry(belief_entry)
    if posterior_entry is None:
        raise ValueError("Frozen theta evaluation requires belief_entry/posterior_entry.")
    if model in {"simplex_quad", "entropy_spline", "pair_product", "deepsets"}:
        features = theta_feature_vector(
            model,
            evidence_state=s,
            posterior_entry=posterior_entry,
            individuals=individuals,
            gen_states=gen_states,
            spec=spec,
            genes=genes,
        )
        weights = np.asarray(spec.get("weights", []), dtype=float)
        bias = float(spec.get("bias", 0.0))
        if weights.shape != features.shape:
            raise ValueError(
                f"Frozen theta weights shape mismatch for {model}: "
                f"weights={weights.shape}, features={features.shape}."
            )
        return float(bias + float(np.dot(weights, features)))
    if model == "softmax_testscore":
        rows = _person_summary_rows(
            evidence_state=s,
            posterior_entry=posterior_entry,
            individuals=individuals,
            genes=genes,
        )
        score_weights = np.asarray(spec.get("score_weights", []), dtype=float)
        if score_weights.shape != (5,):
            raise ValueError("softmax_testscore requires score_weights of shape (5,).")
        tau = float(spec.get("tau", 1.0))
        tau = max(tau, 1e-6)
        scale = float(spec.get("scale", 1.0))
        bias = float(spec.get("bias", 0.0))
        stage_weight = float(spec.get("stage_weight", 0.0))
        scores = []
        for row in rows:
            if row["untested"] < 0.5:
                continue
            q = np.asarray(row["q"], dtype=float)
            z = np.asarray(
                [
                    float(row["p12"]),
                    float(row["entropy"]),
                    float(row["impurity"]),
                    float(q[1]),
                    float(q[2]),
                ],
                dtype=float,
            )
            scores.append(float(np.dot(score_weights, z)))
        if not scores:
            return float(bias + stage_weight * float(len(s)))
        max_score = max(scores)
        stabilized = np.exp((np.asarray(scores, dtype=float) - max_score) / tau)
        pooled = max_score + tau * math.log(float(np.sum(stabilized)))
        return float(bias + stage_weight * float(len(s)) + scale * pooled)
    raise ValueError(f"Unsupported frozen theta model: {model!r}")
