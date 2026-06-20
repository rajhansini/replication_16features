from collections.abc import Mapping
from dataclasses import dataclass

from ..models.belief import (
    propagate_all_marginals_safe,
    lift_single_gene_posteriors_to_genes,
    propagate_multigene_marginals,
    InferenceResult,
    ensure_belief_with_tuples,
)
from ..policy.extractor import best_action
from ..models.reward import r_reward, r_reward_test, r_reward_testp

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

def exact_value_under_policy(
    policy,
    belief,                 # will grow!
    individuals, gen_states,
    r_reward_test, a, b, c, delta,
    infer,                   # needed when we meet a new state
    theta_star, W_star,      # needed to compute policy for new states
    fixed_cost, variable_cost,
    *,
    theta_mode=None,
    pedigree=None,
    theta_model=None,
    theta_model_spec=None,
    lookahead_depth=0,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    tuple_pmfs=None,
    aaub_star=None,
    W_edge_star=None,
    pedigree_edges=None,
    W_trio_star=None,
    pedigree_trios=None,
    feature_cache=None,
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
    fallback_policy=None,
    start_states=None,
    strict_mode=False,
):
    if feature_cache is None:
        feature_cache = {}
    V = {}                  # memo: state → exact value
    tuple_mode = bool(tuple_pmfs)

    def V_rec(s):
        if s in V:
            return V[s]

        # Ensure the belief for the current state exists
        if s not in belief:
            evidence = dict(_evidence_state(s))
            try:
                if genes:
                    p_post = propagate_multigene_marginals(
                        infer,
                        individuals,
                        gen_states,
                        evidence,
                        genes,
                        aggregate_only=not tuple_mode,
                    )
                else:
                    raw_post = propagate_all_marginals_safe(infer, individuals, gen_states, evidence)
                    p_post = InferenceResult(
                        raw_post,
                        gene_order=("gene",),
                        gen_states=gen_states,
                    )
                z_post = {
                    j: {g: 1.0 if evidence.get(j) == g else 0.0 for g in gen_states}
                    for j in individuals
                }
                belief[s] = (p_post, z_post)
                if belief_gene is not None and genes:
                    per_gene_map = (
                        p_post.get_per_gene_probs()
                        if isinstance(p_post, InferenceResult)
                        else lift_single_gene_posteriors_to_genes(p_post, genes)
                    )
                    belief_gene[s] = per_gene_map
                if (
                    tuple_pmfs is not None
                    and tuple_mode
                    and isinstance(p_post, InferenceResult)
                    and p_post.has_tuple_pmfs()
                ):
                    tuple_pmfs[s] = p_post.get_tuple_pmfs()
            except ValueError:
                # Impossible evidence combination - return 0 value
                V[s] = 0.0
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
                V[s] = 0.0
                return 0.0

        # Get policy for this state (compute if not already known).
        selected_policy = policy
        if s not in policy:
            if fallback_policy is not None and s in fallback_policy:
                selected_policy = fallback_policy
            elif strict_mode:
                raise KeyError(
                    f"strict_mode=True: state {s!r} not in policy. "
                    "Baseline policies must precompute the full policy map "
                    "over all reachable states."
                )
            else:
                policy[s] = best_action(
                    s,
                    theta_star=theta_star,
                    W_star=W_star,
                    belief=belief,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    theta_model=theta_model,
                    theta_model_spec=theta_model_spec,
                    individuals=individuals,
                    gen_states=gen_states,
                    r_reward_testp=r_reward_testp,
                    a=a, b=b, c=c, delta=delta, infer=infer,
                    fixed_cost=fixed_cost, variable_cost=variable_cost,
                    lookahead_depth=lookahead_depth,
                    belief_gene=belief_gene,
                    genes=genes,
                    a_gene=a_gene,
                    b_gene=b_gene,
                    c_gene=c_gene,
                    delta_gene=delta_gene,
                    tuple_pmfs=tuple_pmfs,
                    tuple_mode=tuple_mode,
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
                selected_policy = policy

        act, who, _ = selected_policy[s]
        posterior_entry, _ = belief[s]
        p_s = posterior_entry
        tested = {i for i, _ in _evidence_state(s)}
        
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

        if act == "stop":
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
                for k in individuals if k not in tested
            )
        else:                               # test
            i   = who
            r_i = r_reward_test(
                i,
                p_s,
                a,
                b,
                c,
                delta,
                fixed_cost,
                variable_cost,
                per_gene_probs=gene_probs,
                a_gene=a_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )

            exp_succ = 0.0
            tuple_dist = None
            if tuple_mode:
                if tuple_pmfs:
                    tuple_dist = tuple_pmfs.get(s, {}).get(i)
                if tuple_dist is None and isinstance(posterior_entry, InferenceResult) and posterior_entry.has_tuple_pmfs():
                    tuple_dist = posterior_entry.get_tuple_pmfs().get(i)
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
                if tuple_mode and tuple_dist is None:
                    raise RuntimeError(
                        "CRITICAL FAILURE: Missing tuple PMFs during policy evaluation. "
                        f"State={s!r} person={i!r}. "
                        "THIS RUN IS INVALID — STOP AND FIX BEFORE CONTINUING."
                    )

            if tuple_dist:
                for outcome, prob in tuple_dist.items():
                    if prob <= 0.0:
                        continue
                    succ = _merge_state(s, i, outcome)
                    if len(_evidence_state(succ)) == len(individuals):
                        continue
                    try:
                        if belief_gene is not None and genes and succ not in belief_gene and succ in belief:
                            belief_gene[succ] = lift_single_gene_posteriors_to_genes(belief[succ][0], genes)
                        exp_succ += prob * V_rec(succ)
                    except ValueError:
                        continue
            else:
                for g in gen_states:
                    prob = p_s[i][g]
                    if prob <= 0.0:
                        continue
                    succ = _merge_state(s, i, g)
                    if len(_evidence_state(succ)) == len(individuals):
                        continue
                    try:
                        if belief_gene is not None and genes and succ not in belief_gene and succ in belief:
                            belief_gene[succ] = lift_single_gene_posteriors_to_genes(belief[succ][0], genes)
                        exp_succ += prob * V_rec(succ)
                    except ValueError:
                        continue

            val = r_i + exp_succ

        V[s] = val
        return val

    if start_states is None:
        start_states = tuple(policy.keys())
    else:
        start_states = tuple(start_states)

    # Evaluate every requested start state; new ones appear lazily.
    for s in start_states:
        V_rec(s)

    return V


@dataclass(frozen=True)
class PolicyEvaluationResult:
    start_state: frozenset
    value: float
    values: Mapping[frozenset, float]
    used_fallback_policy: bool


@dataclass(frozen=True)
class PolicyComparisonResult:
    candidate: PolicyEvaluationResult
    incumbent: PolicyEvaluationResult
    delta: float
    incumbent_safe: bool
    epsilon: float


@dataclass(frozen=True)
class ActionComparisonResult:
    state: frozenset
    candidate_action: object
    incumbent_action: object
    candidate: PolicyEvaluationResult
    incumbent: PolicyEvaluationResult
    delta: float
    incumbent_safe: bool
    epsilon: float


def evaluate_policy_from_state(
    policy,
    *,
    state,
    fallback_policy=None,
    **kwargs,
):
    values = exact_value_under_policy(
        policy=policy,
        fallback_policy=fallback_policy,
        start_states=(state,),
        **kwargs,
    )
    if state not in values:
        raise KeyError(f"Policy evaluation did not produce a value for start state {state!r}.")
    return PolicyEvaluationResult(
        start_state=state,
        value=float(values[state]),
        values=values,
        used_fallback_policy=fallback_policy is not None,
    )


def evaluate_policy_from_root(
    policy,
    *,
    root_state=frozenset(),
    fallback_policy=None,
    **kwargs,
):
    return evaluate_policy_from_state(
        policy,
        state=root_state,
        fallback_policy=fallback_policy,
        **kwargs,
    )


def compare_policies_from_root(
    candidate_policy,
    incumbent_policy,
    *,
    root_state=frozenset(),
    epsilon=0.0,
    candidate_fallback_policy=None,
    incumbent_fallback_policy=None,
    **kwargs,
):
    candidate_result = evaluate_policy_from_root(
        candidate_policy,
        root_state=root_state,
        fallback_policy=incumbent_policy if candidate_fallback_policy is None else candidate_fallback_policy,
        **kwargs,
    )
    incumbent_result = evaluate_policy_from_root(
        incumbent_policy,
        root_state=root_state,
        fallback_policy=incumbent_policy if incumbent_fallback_policy is None else incumbent_fallback_policy,
        **kwargs,
    )
    delta = candidate_result.value - incumbent_result.value
    return PolicyComparisonResult(
        candidate=candidate_result,
        incumbent=incumbent_result,
        delta=float(delta),
        incumbent_safe=bool(delta >= -float(epsilon)),
        epsilon=float(epsilon),
    )


def evaluate_action_against_policy(
    state,
    action,
    continuation_policy,
    *,
    fallback_policy=None,
    **kwargs,
):
    values = exact_value_under_policy(
        policy={state: action},
        fallback_policy=continuation_policy if fallback_policy is None else fallback_policy,
        start_states=(state,),
        **kwargs,
    )
    if state not in values:
        raise KeyError(f"Action evaluation did not produce a value for state {state!r}.")
    return PolicyEvaluationResult(
        start_state=state,
        value=float(values[state]),
        values=values,
        used_fallback_policy=True,
    )


def compare_action_against_incumbent(
    state,
    action,
    incumbent_policy,
    *,
    incumbent_action=None,
    epsilon=0.0,
    fallback_policy=None,
    **kwargs,
):
    if incumbent_action is None:
        try:
            incumbent_action = incumbent_policy[state]
        except KeyError as exc:
            raise KeyError(
                f"Incumbent policy does not define an action for state {state!r}."
            ) from exc

    candidate_result = evaluate_action_against_policy(
        state,
        action,
        incumbent_policy,
        fallback_policy=fallback_policy,
        **kwargs,
    )
    incumbent_result = evaluate_action_against_policy(
        state,
        incumbent_action,
        incumbent_policy,
        fallback_policy=incumbent_policy if fallback_policy is None else fallback_policy,
        **kwargs,
    )
    delta = candidate_result.value - incumbent_result.value
    return ActionComparisonResult(
        state=state,
        candidate_action=action,
        incumbent_action=incumbent_action,
        candidate=candidate_result,
        incumbent=incumbent_result,
        delta=float(delta),
        incumbent_safe=bool(delta >= -float(epsilon)),
        epsilon=float(epsilon),
    )
