from __future__ import annotations

from statistics import median
from typing import Any, Mapping, Sequence

from genetic_dp.policy_phase_diagram import finite_float


EPSILON = 1e-12
VALUE_TOL = 1e-8


def exact_stop_opportunity(v_star: Any, v_stop: Any) -> float | None:
    v_star_f = finite_float(v_star)
    v_stop_f = finite_float(v_stop)
    if v_star_f is None or v_stop_f is None:
        return None
    denom = float(v_star_f - v_stop_f)
    return denom if denom > EPSILON else None


def ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    num = finite_float(numerator)
    den = finite_float(denominator)
    if num is None or den is None or den <= EPSILON:
        return None
    return float(num / den)


def policy_panel(
    *,
    v_star: Any,
    v_stop: Any,
    v_myopic: Any,
    v_policy: Any,
    incumbent_value: Any = None,
) -> dict[str, Any]:
    v_star_f = finite_float(v_star)
    v_stop_f = finite_float(v_stop)
    v_myopic_f = finite_float(v_myopic)
    v_policy_f = finite_float(v_policy)
    incumbent_f = finite_float(incumbent_value)
    v_best_simple = None
    if v_stop_f is not None and v_myopic_f is not None:
        v_best_simple = max(v_stop_f, v_myopic_f)
    denom = exact_stop_opportunity(v_star_f, v_stop_f)
    policy_gap = None if v_star_f is None or v_policy_f is None else float(v_star_f - v_policy_f)
    myopic_regret = None if v_star_f is None or v_myopic_f is None else float(v_star_f - v_myopic_f)
    best_simple_regret = None if v_star_f is None or v_best_simple is None else float(v_star_f - v_best_simple)
    recovery_vs_myopic = ratio_or_none(
        None if v_policy_f is None or v_myopic_f is None else v_policy_f - v_myopic_f,
        myopic_regret,
    )
    recovery_vs_best_simple = ratio_or_none(
        None if v_policy_f is None or v_best_simple is None else v_policy_f - v_best_simple,
        best_simple_regret,
    )
    return {
        "V_star": v_star_f,
        "V_stop": v_stop_f,
        "V_myopic": v_myopic_f,
        "V_best_simple": v_best_simple,
        "V_policy": v_policy_f,
        "policy_gap": policy_gap,
        "ratio2": ratio_or_none(policy_gap, denom),
        "myopic_regret": myopic_regret,
        "best_simple_regret": best_simple_regret,
        "recovery_vs_myopic": recovery_vs_myopic,
        "recovery_vs_best_simple": recovery_vs_best_simple,
        "no_worse_than_stop": None if v_policy_f is None or v_stop_f is None else bool(v_policy_f + VALUE_TOL >= v_stop_f),
        "no_worse_than_myopic": None if v_policy_f is None or v_myopic_f is None else bool(v_policy_f + VALUE_TOL >= v_myopic_f),
        "no_worse_than_best_simple": None if v_policy_f is None or v_best_simple is None else bool(v_policy_f + VALUE_TOL >= v_best_simple),
        "no_worse_than_incumbent": None if v_policy_f is None or incumbent_f is None else bool(v_policy_f + VALUE_TOL >= incumbent_f),
        "policy_eval_status": "exact" if v_policy_f is not None else "heuristic",
        "lower_bound_valid": v_policy_f is not None,
    }


def certificate_panel(row: Mapping[str, Any]) -> dict[str, Any]:
    v_star = finite_float(row.get("V_star_root") or row.get("V_star"))
    v_stop = finite_float(row.get("V_stop_root") or row.get("V_stop"))
    denom = exact_stop_opportunity(v_star, v_stop)
    grr2_phi = finite_float(row.get("ADP_root_value_phi_GRR2") or row.get("ADP_root_value_phi_grr2"))
    grr2d_phi = finite_float(row.get("ADP_root_value_phi_GRR2D") or row.get("ADP_root_value_phi_grr2d"))
    grr2_gap1 = finite_float(row.get("gap1_GRR2") or row.get("certificate_gap1_grr2"))
    grr2d_gap1 = finite_float(row.get("gap1_GRR2D") or row.get("certificate_gap1_grr2d"))
    if grr2_phi is None and v_star is not None and grr2_gap1 is not None:
        grr2_phi = float(v_star + grr2_gap1)
    if grr2d_phi is None and v_star is not None and grr2d_gap1 is not None:
        grr2d_phi = float(v_star + grr2d_gap1)
    grr2_ratio3_source = finite_float(row.get("ratio3_GRR2") or row.get("certificate_ratio3_grr2"))
    grr2d_ratio3_source = finite_float(row.get("ratio3_GRR2D") or row.get("certificate_ratio3_grr2d"))
    grr2_ratio3 = ratio_or_none(grr2_gap1, denom)
    grr2d_ratio3 = ratio_or_none(grr2d_gap1, denom)
    u_source = "GRR2D" if grr2d_phi is not None else "GRR2"
    u_ref = grr2d_phi if grr2d_phi is not None else grr2_phi
    ratio3_ref = grr2d_ratio3 if u_source == "GRR2D" else grr2_ratio3
    gap1_ref = grr2d_gap1 if u_source == "GRR2D" else grr2_gap1
    status = "certified" if gap1_ref is not None and gap1_ref >= -VALUE_TOL else "heuristic"
    return {
        "V_star": v_star,
        "V_stop": v_stop,
        "exact_stop_opportunity": denom,
        "GRR2_ADP_root_value_phi": grr2_phi,
        "GRR2_gap1": grr2_gap1,
        "GRR2_ratio3": grr2_ratio3,
        "GRR2_ratio3_source": grr2_ratio3_source,
        "GRR2_topK_hit_exact": row.get("GRR2_topK_contains_exact", row.get("grr2_topk_covers_optimal")),
        "GRR2_upper_bound_status": "certified" if grr2_gap1 is not None and grr2_gap1 >= -VALUE_TOL else "heuristic",
        "GRR2D_ADP_root_value_phi": grr2d_phi,
        "GRR2D_gap1": grr2d_gap1,
        "GRR2D_ratio3": grr2d_ratio3,
        "GRR2D_ratio3_source": grr2d_ratio3_source,
        "GRR2D_topK_hit_exact": row.get("GRR2D_topK_contains_exact", row.get("grr2d_topk_covers_optimal")),
        "GRR2D_upper_bound_status": "certified" if grr2d_gap1 is not None and grr2d_gap1 >= -VALUE_TOL else "heuristic",
        "U_reference_source": u_source,
        "U_reference": u_ref,
        "U_reference_gap1": gap1_ref,
        "U_reference_ratio3": ratio3_ref,
        "upper_bound_status": status,
    }


def bound_panel(
    *,
    certificate: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    u_ref = finite_float(certificate.get("U_reference"))
    v_stop = finite_float(certificate.get("V_stop"))
    denom = finite_float(certificate.get("exact_stop_opportunity"))
    ratio3 = finite_float(certificate.get("U_reference_ratio3"))
    ratio2 = finite_float(policy.get("ratio2"))
    l_policy = finite_float(policy.get("V_policy"))
    bound_gap = None if u_ref is None or l_policy is None else float(u_ref - l_policy)
    bound_ratio = ratio_or_none(bound_gap, denom)
    ratio_sum = None if ratio3 is None or ratio2 is None else float(ratio3 + ratio2)
    identity_error = None if bound_ratio is None or ratio_sum is None else abs(bound_ratio - ratio_sum)
    u_stop_denom = None if u_ref is None or v_stop is None else float(u_ref - v_stop)
    return {
        "U_reference_source": certificate.get("U_reference_source"),
        "U_reference": u_ref,
        "L_policy": l_policy,
        "bound_gap": bound_gap,
        "bound_ratio_exact": bound_ratio,
        "ratio3_plus_ratio2": ratio_sum,
        "ratio_identity_abs_error": identity_error,
        "ratio3_plus_ratio2_check": None if identity_error is None else bool(identity_error <= VALUE_TOL),
        "certified_gap": bound_gap,
        "certified_relative_gap": ratio_or_none(bound_gap, u_stop_denom),
        "certified_opportunity_closed": ratio_or_none(
            None if l_policy is None or v_stop is None else l_policy - v_stop,
            u_stop_denom,
        ),
        "certified_opportunity_closed_exact": ratio_or_none(
            None if l_policy is None or v_stop is None else l_policy - v_stop,
            u_stop_denom,
        ),
    }


def value_stats(values: Sequence[float]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if finite_float(value) is not None]
    if not numeric:
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(numeric)
    p90_idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "mean": float(sum(numeric) / len(numeric)),
        "median": float(median(numeric)),
        "p90": float(ordered[p90_idx]),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
    }
