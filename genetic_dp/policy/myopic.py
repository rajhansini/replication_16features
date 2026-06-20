from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..models.belief import InferenceResult, lift_single_gene_posteriors_to_genes
from ..models.reward import r_reward_test
from .baselines import myopic_greedy
from .evaluator import exact_value_under_policy


def _entry_payload(entry):
    return entry[0] if isinstance(entry, tuple) else entry


def _entry_tuple_pmfs(entry):
    payload = _entry_payload(entry)
    if isinstance(payload, InferenceResult) and payload.has_tuple_pmfs():
        return payload.get_tuple_pmfs()
    return None


def _entry_per_gene(entry):
    payload = _entry_payload(entry)
    if isinstance(payload, InferenceResult):
        return payload.get_per_gene_probs()
    return None


def _state_outcomes(*, state, person, belief, gen_states):
    entry = belief[state]
    tuple_pmfs = _entry_tuple_pmfs(entry)
    if tuple_pmfs is not None:
        for outcome, prob in tuple_pmfs.get(person, {}).items():
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


def _merge_state(state, person, outcome):
    evidence = dict(state)
    evidence[person] = outcome
    return frozenset(evidence.items())


def _dummy_w_star(individuals, gen_states, genes=None):
    if genes:
        from itertools import product

        tuple_outcomes = tuple(product(gen_states, repeat=len(tuple(genes))))
        return {person: {outcome: 0.0 for outcome in tuple_outcomes} for person in individuals}
    return {person: {outcome: 0.0 for outcome in gen_states} for person in individuals}


@dataclass(frozen=True)
class MyopicPolicyEvaluation:
    policy: Mapping[frozenset, object]
    values: Mapping[frozenset, float]
    belief_gene: Mapping[frozenset, object]
    tuple_pmfs: Mapping[frozenset, object]
    root_value: float | None


def build_myopic_policy_map(
    *,
    belief,
    individuals,
    gen_states,
    infer,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    state_pool: Iterable[frozenset] | None = None,
):
    """Build a complete evidence-state map for the one-step myopic policy."""

    policy = {}
    local_belief_gene = dict(belief_gene or {})
    local_tuple_pmfs = {}
    tuple_mode = bool(genes)

    for state, entry in list(belief.items()):
        per_gene = _entry_per_gene(entry)
        if per_gene is not None:
            local_belief_gene[state] = per_gene
        tuple_pmfs = _entry_tuple_pmfs(entry)
        if tuple_pmfs is not None:
            local_tuple_pmfs[state] = tuple_pmfs

    def _record_action(state):
        action = myopic_greedy(
            state,
            belief=belief,
            individuals=individuals,
            gen_states=gen_states,
            infer=infer,
            a=a,
            b=b,
            c=c,
            delta=delta,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            belief_gene=local_belief_gene,
            genes=genes,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
            tuple_mode=tuple_mode,
        )
        policy[state] = action
        entry = belief.get(state)
        if entry is not None:
            per_gene = _entry_per_gene(entry)
            if per_gene is not None:
                local_belief_gene[state] = per_gene
            tuple_pmfs = _entry_tuple_pmfs(entry)
            if tuple_pmfs is not None:
                local_tuple_pmfs[state] = tuple_pmfs
        return action

    if state_pool is not None:
        for state in state_pool:
            if len(state) >= len(individuals):
                policy[state] = ("stop", None, 0.0)
                continue
            _record_action(state)
        return policy, local_belief_gene, local_tuple_pmfs

    stack = [frozenset()]
    seen = {frozenset()}
    while stack:
        state = stack.pop()
        action = _record_action(state)
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
            if len(succ) >= len(individuals) or succ in seen:
                continue
            seen.add(succ)
            stack.append(succ)

    return policy, local_belief_gene, local_tuple_pmfs


def evaluate_myopic_policy(
    *,
    belief,
    individuals,
    gen_states,
    infer,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    belief_gene=None,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    state_pool: Iterable[frozenset] | None = None,
) -> MyopicPolicyEvaluation:
    """Evaluate the exact value of the one-step myopic policy."""

    policy, local_belief_gene, local_tuple_pmfs = build_myopic_policy_map(
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        infer=infer,
        a=a,
        b=b,
        c=c,
        delta=delta,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        belief_gene=belief_gene,
        genes=genes,
        a_gene=a_gene,
        b_gene=b_gene,
        c_gene=c_gene,
        delta_gene=delta_gene,
        state_pool=state_pool,
    )
    if not policy:
        return MyopicPolicyEvaluation(policy, {}, local_belief_gene, local_tuple_pmfs, None)

    values = exact_value_under_policy(
        policy=dict(policy),
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        r_reward_test=r_reward_test,
        a=a,
        b=b,
        c=c,
        delta=delta,
        infer=infer,
        theta_star=0.0,
        W_star=_dummy_w_star(individuals, gen_states, genes=genes),
        theta_mode="scalar",
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        lookahead_depth=0,
        belief_gene=local_belief_gene,
        genes=genes,
        a_gene=a_gene,
        b_gene=b_gene,
        c_gene=c_gene,
        delta_gene=delta_gene,
        tuple_pmfs=local_tuple_pmfs if local_tuple_pmfs else None,
        strict_mode=True,
        start_states=tuple(policy.keys()),
    )
    return MyopicPolicyEvaluation(
        policy=policy,
        values=values,
        belief_gene=local_belief_gene,
        tuple_pmfs=local_tuple_pmfs,
        root_value=values.get(frozenset()),
    )


def carrier_probabilities(entry, genes=None):
    payload = _entry_payload(entry)
    if genes and isinstance(payload, InferenceResult):
        return payload.get_per_gene_probs()
    if isinstance(payload, InferenceResult):
        return payload.marginals
    return payload
