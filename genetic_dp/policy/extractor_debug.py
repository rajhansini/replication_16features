"""
Instrumented copy of genetic_dp/policy/extractor.py to help chase the depth-0
Gap2 regression. This mirrors the core functions with verbose print statements
around the suspected bug surfaces (phi_values vs phi_hat usage, tuple/shift
handling, stop vs test comparisons, and depth semantics).

Usage:
    from genetic_dp.policy import extractor_debug as xd
    action = xd.best_action_debug(...)

Keep this file for investigation only; do not ship in production builds.
"""

from ..models.belief import (
    ensure_belief,
    lift_single_gene_posteriors_to_genes,
    InferenceResult,
)
from ..optimisation.postprocess import phi_hat
from ..models.reward import r_reward, r_reward_testp

TOL = 1e-6


def _log(msg: str):
    print(f"[extractor_debug] {msg}")


def _rollout_value_debug(*, s, depth, phi_values=None, **kwargs):
    _log(f"_rollout_value depth={depth} state={s}")
    key = (s, depth)
    memo = kwargs.get("memo", {})
    if key in memo:
        _log(f"  memo hit {key} -> {memo[key]}")
        return memo[key]

    if s not in kwargs["belief"]:
        try:
            ensure_belief(
                s,
                belief=kwargs["belief"],
                infer=kwargs["infer"],
                I=kwargs["individuals"],
                gen_states=kwargs["gen_states"],
            )
        except ValueError:
            _log("  missing belief; returning 0.0")
            memo[key] = 0.0
            return 0.0

    if depth < 0:
        val = None if phi_values is None else phi_values.get(s)
        if val is None:
            val = phi_hat(
                s,
                theta_star=kwargs["theta_star"],
                W_star=kwargs["W_star"],
                belief=kwargs["belief"],
                gen_states=kwargs["gen_states"],
                individuals=kwargs["individuals"],
                tuple_pmfs=kwargs.get("tuple_pmfs"),
                tuple_mode=kwargs.get("tuple_mode", False),
            )
        _log(f"  base depth<0 -> {val}")
        memo[key] = val
        return val

    val_stop = stop_value_debug(s, **{k: v for k, v in kwargs.items() if k != "memo"})
    best_val = val_stop
    tested = dict(s)
    p_s, _ = kwargs["belief"][s]

    for i in kwargs["individuals"]:
        if i in tested:
            continue
        q = _test_q_value_debug(
            s,
            i,
            depth=depth - 1,
            phi_values=phi_values,
            p_s=p_s,
            memo=memo,
            **kwargs,
        )
        _log(f"  candidate i={i} depth={depth} q={q} stop={val_stop}")
        if q > best_val + TOL:
            best_val = q
    memo[key] = best_val
    return best_val


def _test_q_value_debug(
    s,
    i,
    *,
    depth,
    phi_values=None,
    p_s=None,
    memo=None,
    **kwargs,
):
    _log(f"_test_q_value depth={depth} state={s} i={i}")
    if p_s is None:
        p_s, _ = kwargs["belief"][s]
    gene_posteriors = None
    belief_gene = kwargs.get("belief_gene")
    genes = kwargs.get("genes")
    if belief_gene and s in belief_gene:
        gene_posteriors = belief_gene[s]
    elif genes:
        if isinstance(p_s, InferenceResult):
            gene_posteriors = p_s.get_per_gene_probs()
        else:
            gene_posteriors = lift_single_gene_posteriors_to_genes(p_s, genes)
        if belief_gene is not None:
            belief_gene[s] = gene_posteriors

    p12 = p_s[i][1] + p_s[i][2]
    per_gene_p12 = None
    if gene_posteriors:
        carrier = {}
        for gene, probs in gene_posteriors.items():
            if i in probs:
                carrier[gene] = probs[i].get(1, 0.0) + probs[i].get(2, 0.0)
        per_gene_p12 = carrier or None

    r_i = r_reward_testp(
        i,
        p12,
        kwargs["a"],
        kwargs["b"],
        kwargs["c"],
        kwargs["delta"],
        kwargs["fixed_cost"],
        kwargs["variable_cost"],
        per_gene_p12=per_gene_p12,
        a_gene=kwargs.get("a_gene"),
        c_gene=kwargs.get("c_gene"),
        delta_gene=kwargs.get("delta_gene"),
    )

    tuple_pmfs = kwargs.get("tuple_pmfs")
    tuple_mode = kwargs.get("tuple_mode", False)
    tuple_dist = None
    if tuple_mode:
        if tuple_pmfs:
            tuple_dist = tuple_pmfs.get(s, {}).get(i)
        if tuple_dist is None and isinstance(p_s, InferenceResult) and p_s.has_tuple_pmfs():
            tuple_dist = p_s.get_tuple_pmfs().get(i)

    gene_span = None
    W_star = kwargs["W_star"]
    if tuple_mode:
        sample_keys = W_star.get(i, {})
        if sample_keys:
            sample_key = next(iter(sample_keys))
            if isinstance(sample_key, tuple):
                gene_span = len(sample_key)

    def _normalize_outcome_for_w(value):
        if not tuple_mode or gene_span is None:
            return value
        if isinstance(value, tuple):
            if len(value) == gene_span:
                return value
            if len(value) == 1:
                return tuple(value[0] for _ in range(gene_span))
            return tuple(value[:gene_span])
        return tuple(value for _ in range(gene_span))

    if tuple_dist:
        shift = -sum(prob * W_star[i][_normalize_outcome_for_w(outcome)] for outcome, prob in tuple_dist.items())
    else:
        shift = -sum(p_s[i][g] * W_star[i][_normalize_outcome_for_w(g)] for g in kwargs["gen_states"])

    _log(f"  r_i={r_i} shift={shift} tuple_dist={'yes' if tuple_dist else 'no'} depth={depth}")
    total = r_i + shift

    if tuple_dist:
        for outcome, prob in tuple_dist.items():
            if prob <= 0.0:
                continue
            normalized_outcome = _normalize_outcome_for_w(outcome)
            succ = frozenset(dict(s, **{i: normalized_outcome}).items())
            try:
                ensure_belief(
                    succ,
                    belief=kwargs["belief"],
                    infer=kwargs["infer"],
                    I=kwargs["individuals"],
                    gen_states=kwargs["gen_states"],
                )
            except ValueError:
                continue
            if depth < 0:
                future_val = None if phi_values is None else phi_values.get(succ)
                if future_val is None:
                    future_val = phi_hat(
                        succ,
                        theta_star=kwargs["theta_star"],
                        W_star=kwargs["W_star"],
                        belief=kwargs["belief"],
                        gen_states=kwargs["gen_states"],
                        individuals=kwargs["individuals"],
                        tuple_pmfs=kwargs.get("tuple_pmfs"),
                        tuple_mode=tuple_mode,
                    )
            else:
                future_val = _rollout_value_debug(
                    s=succ,
                    depth=depth,
                    phi_values=phi_values,
                    memo=memo,
                    **kwargs,
                )
            _log(f"    succ tuple={outcome} prob={prob} future={future_val}")
            total += prob * future_val
    else:
        for g in kwargs["gen_states"]:
            prob = p_s[i][g]
            if prob <= 0.0:
                continue
            normalized_outcome = _normalize_outcome_for_w(g)
            succ = frozenset(s | {(i, normalized_outcome)})
            try:
                ensure_belief(
                    succ,
                    belief=kwargs["belief"],
                    infer=kwargs["infer"],
                    I=kwargs["individuals"],
                    gen_states=kwargs["gen_states"],
                )
            except ValueError:
                continue
            if depth < 0:
                future_val = None if phi_values is None else phi_values.get(succ)
                if future_val is None:
                    future_val = phi_hat(
                        succ,
                        theta_star=kwargs["theta_star"],
                        W_star=kwargs["W_star"],
                        belief=kwargs["belief"],
                        gen_states=kwargs["gen_states"],
                        individuals=kwargs["individuals"],
                        tuple_pmfs=kwargs.get("tuple_pmfs"),
                        tuple_mode=kwargs.get("tuple_mode", False),
                    )
            else:
                future_val = _rollout_value_debug(
                    s=succ,
                    depth=depth,
                    phi_values=phi_values,
                    memo=memo,
                    **kwargs,
                )
            _log(f"    succ scalar={g} prob={prob} future={future_val}")
            total += prob * future_val

    _log(f"  return total={total}")
    return total


def q_test_debug(s, i, *, phi_values=None, **kwargs):
    return _test_q_value_debug(s, i, depth=0, phi_values=phi_values, **kwargs)


def stop_value_debug(
    s,
    *,
    belief,
    a,
    b,
    c,
    delta,
    individuals,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
):
    posterior_entry, _ = belief[s]
    p_s = posterior_entry
    gene_probs = None
    if genes:
        if belief_gene and s in belief_gene:
            gene_probs = belief_gene[s]
        elif isinstance(posterior_entry, InferenceResult):
            gene_probs = posterior_entry.get_per_gene_probs()
            if belief_gene is not None:
                belief_gene[s] = gene_probs
        else:
            gene_probs = lift_single_gene_posteriors_to_genes(p_s, genes)
            if belief_gene is not None:
                belief_gene[s] = gene_probs

    val = sum(
        r_reward(
            k,
            p_s,
            a,
            b,
            c,
            delta,
            per_gene_probs=gene_probs,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
        )
        for k in individuals if k not in dict(s)
    )
    _log(f"stop_value state={s} -> {val}")
    return val


def best_action_debug(
    s, *,
    theta_star, W_star, belief,
    individuals, gen_states,
    r_reward_testp, a, b, c, delta, infer, fixed_cost, variable_cost,
    lookahead_depth=0,
    _memo=None,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    tuple_pmfs=None,
    tuple_mode=False,
    phi_values=None,
):
    if _memo is None:
        _memo = {}
    _log(f"best_action depth={lookahead_depth} state={s}")

    if lookahead_depth <= 0:
        phi_s = None if phi_values is None else phi_values.get(s)
        if phi_s is None:
            phi_s = phi_hat(
                s,
                theta_star=theta_star,
                W_star=W_star,
                belief=belief,
                gen_states=gen_states,
                individuals=individuals,
                tuple_pmfs=tuple_pmfs,
                tuple_mode=tuple_mode,
            )
        val_stop = stop_value_debug(
            s,
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
        _log(f"  phi_s={phi_s} val_stop={val_stop}")
        if abs(phi_s - val_stop) <= TOL or val_stop >= phi_s:
            _log("  choose STOP (tight or dominant)")
            return ("stop", None, val_stop)

        best_i, best_q = None, -float("inf")
        for i in individuals:
            if i in dict(s):
                continue
            q_i = q_test_debug(
                s,
                i,
                theta_star=theta_star,
                W_star=W_star,
                belief=belief,
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
            )
            _log(f"  i={i} q_i={q_i}")
            if abs(phi_s - q_i) <= TOL:
                _log("  choose TEST (tight)")
                return ("test", i, q_i)
            if q_i > best_q:
                best_q, best_i = q_i, i

        return ("test", best_i, best_q) if best_i is not None else ("stop", None, val_stop)

    val_stop = stop_value_debug(
        s,
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
    best_val = val_stop
    best_action_choice = ("stop", None, val_stop)

    tested = dict(s)
    p_s, _ = belief[s]
    for i in individuals:
        if i in tested:
            continue
        q_i = _test_q_value_debug(
            s,
            i,
            depth=lookahead_depth - 1,
            theta_star=theta_star,
            W_star=W_star,
            belief=belief,
            individuals=individuals,
            gen_states=gen_states,
            r_reward_testp=r_reward_testp,
            a=a,
            b=b,
            c=c,
            delta=delta,
            infer=infer,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            memo=_memo,
            p_s=p_s,
            belief_gene=belief_gene,
            genes=genes,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
            tuple_pmfs=tuple_pmfs,
            tuple_mode=tuple_mode,
            phi_values=phi_values,
        )
        _log(f"  LA i={i} q_i={q_i} stop={val_stop}")
        if q_i > best_val + TOL:
            best_val = q_i
            best_action_choice = ("test", i, q_i)

    return best_action_choice

