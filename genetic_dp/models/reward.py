import numpy as np
from typing import Dict, Mapping, Optional

def _carrier_probability_from_dist(dist: Mapping[int, float]) -> float:
    return dist.get(1, 0.0) + dist.get(2, 0.0)

def _base_constant(c_map, c_gene_map, individual):
    base = c_map.get(individual, 0.0)
    if not c_gene_map:
        return base
    return base - sum(
        gene_constants.get(individual, 0.0)
        for gene_constants in c_gene_map.values()
    )

def _resolve_coeff(
    base_map: Mapping[str, float],
    gene_map: Optional[Mapping[str, Mapping[str, float]]],
    individual: str,
    gene: Optional[str] = None,
) -> float:
    if gene is not None and gene_map:
        gene_coeffs = gene_map.get(gene)
        if gene_coeffs is not None and individual in gene_coeffs:
            return gene_coeffs[individual]
    return base_map.get(individual, 0.0)

def _extract_single_gene_p12(p_vals, individual: str):
    try:                        # tupledict path
        return p_vals[individual, 1] + p_vals[individual, 2]
    except Exception:
        pass
    person = p_vals[individual]
    return person[1] + person[2]

def r_reward(
    i,
    p_vals,
    a,
    b,
    c,
    delta,
    *,
    per_gene_probs: Optional[Mapping[str, Mapping[str, Mapping[int, float]]]] = None,
    a_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    b_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    c_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    delta_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
):
    """
    Works with tupledict *or* nested-dict p_vals.
    Supports optional per-gene posterior dictionaries to compute additive rewards.
    """
    if per_gene_probs:
        total = _base_constant(c, c_gene, i)
        for gene, gene_probs in per_gene_probs.items():
            if i not in gene_probs:
                continue
            p12 = _carrier_probability_from_dist(gene_probs[i])
            coeff_a = _resolve_coeff(a, a_gene, i, gene)
            coeff_b = _resolve_coeff(b, b_gene, i, gene)
            coeff_delta = _resolve_coeff(delta, delta_gene, i, gene)
            total += (
                coeff_a * (p12 - coeff_delta * p12**2)
                + coeff_b * (p12 - p12**2)
                + (c_gene.get(gene, {}).get(i, 0.0) if c_gene else 0.0)
            )
        return total

    p12 = _extract_single_gene_p12(p_vals, i)
    p12_np = np.array(p12)
    return (
        a[i] * (p12_np - delta[i] * p12_np**2)
        + b[i] * (p12_np - p12_np**2)
        + c[i]
    )

def r_reward_p(
    i,
    p12,
    a,
    b,
    c,
    delta,
    *,
    per_gene_p12: Optional[Mapping[str, float]] = None,
    a_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    b_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    c_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    delta_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
):
    if per_gene_p12:
        total = _base_constant(c, c_gene, i)
        for gene, mass in per_gene_p12.items():
            coeff_a = _resolve_coeff(a, a_gene, i, gene)
            coeff_b = _resolve_coeff(b, b_gene, i, gene)
            coeff_delta = _resolve_coeff(delta, delta_gene, i, gene)
            total += (
                coeff_a * (mass - coeff_delta * mass**2)
                + coeff_b * (mass - mass**2)
                + (c_gene.get(gene, {}).get(i, 0.0) if c_gene else 0.0)
            )
        return total

    p12_np = np.array(p12)
    return (
        a[i] * (p12_np - delta[i] * p12_np**2)
        + b[i] * (p12_np - p12_np**2)
        + c[i]
    )

def r_reward_test(
    i,
    p_vals,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    *,
    per_gene_probs: Optional[Mapping[str, Mapping[str, Mapping[int, float]]]] = None,
    a_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    c_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    delta_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
):
    """
    Works with tupledict *or* nested-dict p_vals.
    """
    if per_gene_probs:
        total = _base_constant(c, c_gene, i)
        prob_none_positive = 1.0
        for gene, gene_probs in per_gene_probs.items():
            if i not in gene_probs:
                continue
            p12 = _carrier_probability_from_dist(gene_probs[i])
            coeff_a = _resolve_coeff(a, a_gene, i, gene)
            coeff_delta = _resolve_coeff(delta, delta_gene, i, gene)
            total += coeff_a * (1 - coeff_delta) * p12
            if c_gene:
                total += c_gene.get(gene, {}).get(i, 0.0)
            prob_none_positive *= 1.0 - p12
        prob_any_positive = 1.0 - prob_none_positive
        return total - fixed_cost - variable_cost * prob_any_positive

    p12 = _extract_single_gene_p12(p_vals, i)
    p12_np = np.array(p12)
    return a[i]*(1- delta[i])*p12_np + c[i] -fixed_cost- variable_cost*p12_np

def r_reward_testp(
    i,
    p12,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    *,
    per_gene_p12: Optional[Mapping[str, float]] = None,
    a_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    c_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
    delta_gene: Optional[Mapping[str, Mapping[str, float]]] = None,
):
    if per_gene_p12:
        total = _base_constant(c, c_gene, i)
        prob_none_positive = 1.0
        for gene, mass in per_gene_p12.items():
            coeff_a = _resolve_coeff(a, a_gene, i, gene)
            coeff_delta = _resolve_coeff(delta, delta_gene, i, gene)
            total += coeff_a * (1 - coeff_delta) * mass
            if c_gene:
                total += c_gene.get(gene, {}).get(i, 0.0)
            prob_none_positive *= 1.0 - mass
        prob_any_positive = 1.0 - prob_none_positive
        return total - fixed_cost - variable_cost * prob_any_positive

    p12_np = np.array(p12)
    return a[i]*(1- delta[i])*p12_np + c[i] -fixed_cost- variable_cost*p12_np
