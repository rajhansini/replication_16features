from ..models.belief import (
    ensure_belief,
    ensure_belief_with_tuples,
    lift_single_gene_posteriors_to_genes,
    InferenceResult,
)
from ..optimisation.postprocess import phi_hat
from ..models.reward import r_reward

TOL = 1e-6


def _evidence_state(state):
    if not isinstance(state, frozenset):
        raise AssertionError(
            f"State must be evidence-only frozenset[(person,outcome)], got {type(state).__name__}: {state!r}"
        )
    return state


def _merge_state(state, person, outcome):
    evidence = _evidence_state(state)
    state_dict = dict(evidence)
    state_dict[person] = outcome
    return frozenset(state_dict.items())


def _rollout_value(
    s,
    *,
    depth,
    theta_star,
    theta_mode=None,
    pedigree=None,
    theta_model=None,
    theta_model_spec=None,
    W_star,
    belief,
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
    memo,
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
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
):
    """Recursive value estimate using limited lookahead.

    Falls back to the dual-derived approximation when ``depth`` drops below zero.
    ``memo`` avoids recomputing the same (state, depth) pairs.
    """

    key = (s, depth)
    if key in memo:
        return memo[key]

    # Terminal boundary condition: once everyone is tested there is no future
    # reward, so V(s)=0 for all fully-tested states (independent of outcomes).
    if len(_evidence_state(s)) >= len(individuals):
        memo[key] = 0.0
        return 0.0

    if s not in belief:
        try:
            if tuple_mode and genes:
                ensure_belief_with_tuples(
                    s,
                    belief=belief,
                    infer=infer,
                    I=individuals,
                    gen_states=gen_states,
                    genes=genes,
                )
            else:
                ensure_belief(
                    s,
                    belief=belief,
                    infer=infer,
                    I=individuals,
                    gen_states=gen_states,
                )
        except ValueError:
            memo[key] = 0.0
            return 0.0
    elif tuple_mode and genes:
        try:
            ensure_belief_with_tuples(
                s,
                belief=belief,
                infer=infer,
                I=individuals,
                gen_states=gen_states,
                genes=genes,
            )
        except ValueError:
            memo[key] = 0.0
            return 0.0

    if tuple_mode and tuple_pmfs is not None:
        entry = belief[s][0] if isinstance(belief[s], tuple) else belief[s]
        if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
            tuple_pmfs[s] = entry.get_tuple_pmfs()

    # Base case: rely on the dual solution's Φ̂ estimate.
    if depth < 0:
        val = phi_values.get(s) if phi_values is not None else None
        if val is None:
            val = phi_hat(
                s,
                theta_star=theta_star,
                W_star=W_star,
                belief=belief,
                gen_states=gen_states,
                individuals=individuals,
                theta_mode=theta_mode,
                pedigree=pedigree,
                tuple_pmfs=tuple_pmfs,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=W_edge_star,
                pedigree_edges=pedigree_edges,
                W_trio_star=W_trio_star,
                pedigree_trios=pedigree_trios,
                feature_cache=feature_cache,
                infer=infer,
                genes=genes,
                theta_model=theta_model,
                theta_model_spec=theta_model_spec,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
        memo[key] = val
        return val

    val_stop = stop_value(
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

    tested = dict(_evidence_state(s))
    p_s, _ = belief[s]
    best_val = val_stop

    for i in individuals:
        if i in tested:
            continue

        test_val = _test_q_value(
            s,
            i,
            depth=depth - 1,
            theta_star=theta_star,
            theta_mode=theta_mode,
            pedigree=pedigree,
            theta_model=theta_model,
            theta_model_spec=theta_model_spec,
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
            memo=memo,
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
            aaub_star=aaub_star,
            W_edge_star=W_edge_star,
            pedigree_edges=pedigree_edges,
            W_trio_star=W_trio_star,
            pedigree_trios=pedigree_trios,
            feature_cache=feature_cache,
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
        )

        if test_val > best_val + TOL:
            best_val = test_val

    memo[key] = best_val
    return best_val


def _test_q_value(
    s,
    i,
    *,
    depth,
    theta_star,
    theta_mode=None,
    pedigree=None,
    theta_model=None,
    theta_model_spec=None,
    W_star,
    belief,
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
    memo,
    p_s=None,
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
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
):
    """Evaluate the Q-value for testing individual ``i`` with lookahead.

    ``depth`` tracks the remaining number of future tests we are allowed to
    expand *after* committing to test ``i``. When it becomes negative the
    recursion falls back to the Φ̂ approximation.
    """

    if p_s is None:
        p_s, _ = belief[s]
    gene_posteriors = None
    if belief_gene and s in belief_gene:
        gene_posteriors = belief_gene[s]
    elif genes:
        if isinstance(p_s, InferenceResult):
            gene_posteriors = p_s.get_per_gene_probs()
        else:
            gene_posteriors = lift_single_gene_posteriors_to_genes(p_s, genes)
        if belief_gene is not None:
            belief_gene[s] = gene_posteriors

    # immediate reward uses current carrier probability
    p12 = p_s[i][1] + p_s[i][2]
    def _per_gene_p12_map(gene_probs, person):
        if not gene_probs:
            return None
        carrier = {}
        for gene, probs in gene_probs.items():
            if person not in probs:
                continue
            carrier_prob = probs[person].get(1, 0.0) + probs[person].get(2, 0.0)
            carrier[gene] = carrier_prob
        return carrier or None

    per_gene_p12 = _per_gene_p12_map(gene_posteriors, i)
    r_i = r_reward_testp(
        i,
        p12,
        a,
        b,
        c,
        delta,
        fixed_cost,
        variable_cost,
        per_gene_p12=per_gene_p12,
        a_gene=a_gene,
        c_gene=c_gene,
        delta_gene=delta_gene,
    )

    tuple_dist = None
    if tuple_mode:
        if tuple_pmfs:
            tuple_dist = tuple_pmfs.get(s, {}).get(i)
        if tuple_dist is None and isinstance(p_s, InferenceResult) and p_s.has_tuple_pmfs():
            tuple_dist = p_s.get_tuple_pmfs().get(i)
        if tuple_dist is None and genes:
            ensure_belief_with_tuples(
                s,
                belief=belief,
                infer=infer,
                I=individuals,
                gen_states=gen_states,
                genes=genes,
            )
            entry = belief[s][0] if isinstance(belief[s], tuple) else belief[s]
            if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
                if tuple_pmfs is not None:
                    tuple_pmfs[s] = entry.get_tuple_pmfs()
                tuple_dist = entry.get_tuple_pmfs().get(i)

    gene_span = None
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

    total = r_i

    if tuple_dist:
        for outcome, prob in tuple_dist.items():
            if prob <= 0.0:
                continue
            normalized_outcome = _normalize_outcome_for_w(outcome)
            succ = _merge_state(s, i, normalized_outcome)
            if len(_evidence_state(succ)) >= len(individuals):
                continue
            observed_val = outcome[0] if isinstance(outcome, tuple) else outcome
            try:
                if tuple_mode and genes:
                    ensure_belief_with_tuples(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                        genes=genes,
                    )
                else:
                    ensure_belief(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                    )
                if belief_gene is not None and genes and succ not in belief_gene:
                    succ_entry = belief[succ][0]
                    if isinstance(succ_entry, InferenceResult):
                        belief_gene[succ] = succ_entry.get_per_gene_probs()
                    else:
                        belief_gene[succ] = lift_single_gene_posteriors_to_genes(succ_entry, genes)
                if tuple_mode and tuple_pmfs is not None:
                    succ_entry = belief[succ][0] if isinstance(belief[succ], tuple) else belief[succ]
                    if isinstance(succ_entry, InferenceResult) and succ_entry.has_tuple_pmfs():
                        tuple_pmfs[succ] = succ_entry.get_tuple_pmfs()
            except ValueError:
                continue

            if depth < 0:
                future_val = phi_values.get(succ) if phi_values is not None else None
                if future_val is None:
                    future_val = phi_hat(
                        succ,
                        theta_star=theta_star,
                        W_star=W_star,
                        belief=belief,
                        gen_states=gen_states,
                        individuals=individuals,
                        theta_mode=theta_mode,
                        pedigree=pedigree,
                        tuple_pmfs=tuple_pmfs,
                        tuple_mode=tuple_mode,
                        aaub_star=aaub_star,
                        W_edge_star=W_edge_star,
                        pedigree_edges=pedigree_edges,
                        W_trio_star=W_trio_star,
                        pedigree_trios=pedigree_trios,
                        feature_cache=feature_cache,
                        infer=infer,
                        genes=genes,
                        theta_model=theta_model,
                        theta_model_spec=theta_model_spec,
                        myopic_adp_star=myopic_adp_star,
                        oracle_adp_star=oracle_adp_star,
                        regime_residual_star=regime_residual_star,
                    )
            else:
                future_val = _rollout_value(
                    succ,
                    depth=depth,
                    theta_star=theta_star,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    theta_model=theta_model,
                    theta_model_spec=theta_model_spec,
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
                    memo=memo,
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
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
            total += prob * future_val
    else:
        for g in gen_states:
            prob = p_s[i][g]
            if prob <= 0.0:
                continue

            normalized_outcome = _normalize_outcome_for_w(g)
            succ = _merge_state(s, i, normalized_outcome)
            if len(_evidence_state(succ)) >= len(individuals):
                continue
            try:
                if tuple_mode and genes:
                    ensure_belief_with_tuples(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                        genes=genes,
                    )
                else:
                    ensure_belief(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                    )
                if belief_gene is not None and genes and succ not in belief_gene:
                    succ_entry = belief[succ][0]
                    if isinstance(succ_entry, InferenceResult):
                        belief_gene[succ] = succ_entry.get_per_gene_probs()
                    else:
                        belief_gene[succ] = lift_single_gene_posteriors_to_genes(succ_entry, genes)
                if tuple_mode and tuple_pmfs is not None:
                    succ_entry = belief[succ][0] if isinstance(belief[succ], tuple) else belief[succ]
                    if isinstance(succ_entry, InferenceResult) and succ_entry.has_tuple_pmfs():
                        tuple_pmfs[succ] = succ_entry.get_tuple_pmfs()
            except ValueError:
                # impossible successor → contributes 0
                continue

            if depth < 0:
                future_val = phi_values.get(succ) if phi_values is not None else None
                if future_val is None:
                    future_val = phi_hat(
                        succ,
                        theta_star=theta_star,
                        W_star=W_star,
                        belief=belief,
                        gen_states=gen_states,
                        individuals=individuals,
                        theta_mode=theta_mode,
                        pedigree=pedigree,
                        tuple_pmfs=tuple_pmfs,
                        tuple_mode=tuple_mode,
                        aaub_star=aaub_star,
                        W_edge_star=W_edge_star,
                        pedigree_edges=pedigree_edges,
                        W_trio_star=W_trio_star,
                        pedigree_trios=pedigree_trios,
                        feature_cache=feature_cache,
                        infer=infer,
                        genes=genes,
                        theta_model=theta_model,
                        theta_model_spec=theta_model_spec,
                        myopic_adp_star=myopic_adp_star,
                        oracle_adp_star=oracle_adp_star,
                        regime_residual_star=regime_residual_star,
                    )
            else:
                future_val = _rollout_value(
                    succ,
                    depth=depth,
                    theta_star=theta_star,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    theta_model=theta_model,
                    theta_model_spec=theta_model_spec,
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
                    memo=memo,
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
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
            total += prob * future_val

    return total

def q_test(
    s, i, *,
    theta_star, W_star, belief,
    theta_mode=None,
    pedigree=None,
    theta_model=None,
    theta_model_spec=None,
    gen_states, r_reward_testp, a, b, c, delta, individuals, infer, fixed_cost, variable_cost,
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
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
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

    # immediate reward
    p12 = p_s[i][1] + p_s[i][2]
    def _per_gene_p12_map(gene_probs, person):
        if not gene_probs:
            return None
        carrier = {}
        for gene, probs in gene_probs.items():
            if person not in probs:
                continue
            carrier_prob = probs[person].get(1, 0.0) + probs[person].get(2, 0.0)
            carrier[gene] = carrier_prob
        return carrier or None

    per_gene_p12 = _per_gene_p12_map(gene_probs, i)
    r_i = r_reward_testp(
        i,
        p12,
        a,
        b,
        c,
        delta,
        fixed_cost,
        variable_cost,
        per_gene_p12=per_gene_p12,
        a_gene=a_gene,
        c_gene=c_gene,
        delta_gene=delta_gene,
    )

    tuple_dist = None
    if tuple_mode:
        if tuple_pmfs:
            tuple_dist = tuple_pmfs.get(s, {}).get(i)
        if tuple_dist is None and isinstance(p_s, InferenceResult) and p_s.has_tuple_pmfs():
            tuple_dist = p_s.get_tuple_pmfs().get(i)
        if tuple_dist is None and genes:
            ensure_belief_with_tuples(
                s,
                belief=belief,
                infer=infer,
                I=individuals,
                gen_states=gen_states,
                genes=genes,
            )
            entry = belief[s][0] if isinstance(belief[s], tuple) else belief[s]
            if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
                if tuple_pmfs is not None:
                    tuple_pmfs[s] = entry.get_tuple_pmfs()
                tuple_dist = entry.get_tuple_pmfs().get(i)

    gene_span = None
    if tuple_mode:
        sample_keys = W_star.get(i, {})
        if sample_keys:
            sample_key = next(iter(sample_keys))
            if isinstance(sample_key, tuple):
                gene_span = len(sample_key)

    def _normalize_outcome_for_w(value):
        if tuple_mode and gene_span:
            if isinstance(value, tuple):
                if len(value) == gene_span:
                    return tuple(value)
                return tuple(value[:gene_span])
            return tuple(value for _ in range(gene_span))
        return value

    # successor value
    V_i = 0.0

    if tuple_dist:
        for outcome, prob in tuple_dist.items():
            if prob <= 0.0:
                continue
            normalized_outcome = _normalize_outcome_for_w(outcome)
            succ = _merge_state(s, i, normalized_outcome)
            if len(_evidence_state(succ)) >= len(individuals):
                continue
            try:
                if tuple_mode and genes:
                    ensure_belief_with_tuples(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                        genes=genes,
                    )
                else:
                    ensure_belief(succ, belief=belief,
                                  infer=infer, I=individuals, gen_states=gen_states)
            except ValueError:
                continue
            if tuple_mode and tuple_pmfs is not None:
                succ_entry = belief[succ][0] if isinstance(belief[succ], tuple) else belief[succ]
                if isinstance(succ_entry, InferenceResult) and succ_entry.has_tuple_pmfs():
                    tuple_pmfs[succ] = succ_entry.get_tuple_pmfs()
            succ_val = phi_values.get(succ) if phi_values is not None else None
            if succ_val is None:
                succ_val = phi_hat(
                    succ,
                    theta_star=theta_star, W_star=W_star,
                    belief=belief, gen_states=gen_states, individuals=individuals,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    tuple_pmfs=tuple_pmfs,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=W_edge_star,
                    pedigree_edges=pedigree_edges,
                    W_trio_star=W_trio_star,
                    pedigree_trios=pedigree_trios,
                    feature_cache=feature_cache,
                    infer=infer,
                    genes=genes,
                    theta_model=theta_model,
                    theta_model_spec=theta_model_spec,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
            V_i += prob * succ_val
    else:
        for g in gen_states:
            prob = p_s[i][g]
            if prob <= 0.0:
                continue
            normalized_outcome = _normalize_outcome_for_w(g)
            succ = _merge_state(s, i, normalized_outcome)
            if len(_evidence_state(succ)) >= len(individuals):
                continue
            try:
                if tuple_mode and genes:
                    ensure_belief_with_tuples(
                        succ,
                        belief=belief,
                        infer=infer,
                        I=individuals,
                        gen_states=gen_states,
                        genes=genes,
                    )
                else:
                    ensure_belief(succ, belief=belief,
                                  infer=infer, I=individuals, gen_states=gen_states)
            except ValueError:
                continue
            if tuple_mode and tuple_pmfs is not None:
                succ_entry = belief[succ][0] if isinstance(belief[succ], tuple) else belief[succ]
                if isinstance(succ_entry, InferenceResult) and succ_entry.has_tuple_pmfs():
                    tuple_pmfs[succ] = succ_entry.get_tuple_pmfs()
            succ_val = phi_values.get(succ) if phi_values is not None else None
            if succ_val is None:
                succ_val = phi_hat(
                    succ,
                    theta_star=theta_star, W_star=W_star,
                    belief=belief, gen_states=gen_states, individuals=individuals,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    tuple_pmfs=tuple_pmfs,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=W_edge_star,
                    pedigree_edges=pedigree_edges,
                    W_trio_star=W_trio_star,
                    pedigree_trios=pedigree_trios,
                    feature_cache=feature_cache,
                    infer=infer,
                    genes=genes,
                    theta_model=theta_model,
                    theta_model_spec=theta_model_spec,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
            V_i += prob * succ_val

    return r_i + V_i

def stop_value(
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

    return sum(
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
        for k in individuals if k not in dict(_evidence_state(s))
    )

def best_action(
    s, *,
    theta_star, W_star, belief,
    theta_mode=None,
    pedigree=None,
    theta_model=None,
    theta_model_spec=None,
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
    aaub_star=None,
    W_edge_star=None,
    pedigree_edges=None,
    W_trio_star=None,
    pedigree_trios=None,
    feature_cache=None,
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
):
    """Return the greedy/rollout action for state ``s``.

    When ``lookahead_depth`` is zero we preserve the legacy behaviour that
    selects the action whose Bellman inequality is tight under the dual
    solution. Positive depth triggers a limited-lookahead rollout that uses
    exact posteriors for the first few steps before falling back to Φ̂.
    """

    if _memo is None:
        _memo = {}

    if lookahead_depth <= 0:
        val_stop = stop_value(
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
        best_action_choice = ("stop", None, val_stop)
        best_q = val_stop
        for i in individuals:
            if i in dict(_evidence_state(s)):
                continue
            q_i = q_test(
                s,
                i,
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
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
            if q_i > best_q + TOL:
                best_q = q_i
                best_action_choice = ("test", i, q_i)

        return best_action_choice

    # Lookahead case
    val_stop = stop_value(
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

    tested = dict(_evidence_state(s))
    p_s, _ = belief[s]

    best_action_choice = ("stop", None, val_stop)
    best_val = val_stop

    for i in individuals:
        if i in tested:
            continue

        q_i = _test_q_value(
            s,
            i,
            depth=lookahead_depth - 1,
            theta_star=theta_star,
            theta_mode=theta_mode,
            pedigree=pedigree,
            theta_model=theta_model,
            theta_model_spec=theta_model_spec,
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
            aaub_star=aaub_star,
            W_edge_star=W_edge_star,
            pedigree_edges=pedigree_edges,
            W_trio_star=W_trio_star,
            pedigree_trios=pedigree_trios,
            feature_cache=feature_cache,
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
        )

        if q_i > best_val + TOL:
            best_val = q_i
            best_action_choice = ("test", i, q_i)

    return best_action_choice
