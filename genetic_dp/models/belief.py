import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Optional, Sequence

import numpy as np

from .genetics_cpd import genotype_node_name
from .outcomes import canonical_gene_order

@dataclass
class GeneBelief:
    """
    Container for gene-indexed marginal beliefs. Maintains compatibility with the
    existing single-gene representation by allowing callers to materialize a single
    aggregate slice while tracking per-gene views for the forthcoming additive model.
    """
    individuals: Tuple[str, ...]
    genes: Tuple[str, ...]
    gen_states: Tuple[int, ...]
    per_gene: Dict[str, Dict[str, Dict[int, float]]] = field(default_factory=dict)

    @classmethod
    def from_single_gene(
        cls,
        posteriors: Dict[str, Dict[int, float]],
        individuals: Iterable[str],
        gen_states: Iterable[int],
        gene_name: str = "gene",
    ) -> "GeneBelief":
        per_gene = {gene_name: posteriors}
        return cls(
            individuals=tuple(individuals),
            genes=(gene_name,),
            gen_states=tuple(gen_states),
            per_gene=per_gene,
        )

    def as_single_gene(self) -> Dict[str, Dict[int, float]]:
        if len(self.genes) != 1:
            raise ValueError("GeneBelief.as_single_gene requires exactly one gene")
        gene = self.genes[0]
        return self.per_gene.get(gene, {})


class InferenceResult(Mapping):
    """
    Mapping-like wrapper for inference outputs. Exposes per-individual
    marginals while optionally storing joint tuple probabilities and
    per-gene decompositions.
    """

    def __init__(
        self,
        marginals: Dict[str, Dict[int, float]],
        tuple_pmfs: Optional[Dict[str, Dict[Tuple[int, ...], float]]] = None,
        *,
        per_gene: Optional[Dict[str, Dict[str, Dict[int, float]]]] = None,
        gene_order: Optional[Sequence[str]] = None,
        gen_states: Optional[Sequence[int]] = None,
    ):
        self.marginals = {person: dict(probs) for person, probs in marginals.items()}
        self._tuple_pmfs = (
            {person: dict(outcomes) for person, outcomes in tuple_pmfs.items()}
            if tuple_pmfs is not None
            else {}
        )
        self._gene_order = tuple(gene_order) if gene_order is not None else None
        self._gen_states = tuple(gen_states) if gen_states is not None else None

        # Internal storage for gene-first and person-first marginals.
        self._per_gene_gene_first: Dict[str, Dict[str, Dict[int, float]]] = {}
        self._per_gene_person_first: Dict[str, Dict[str, Dict[int, float]]] = {}

        if per_gene is not None:
            self._initialise_per_gene(per_gene)

    def _initialise_per_gene(self, per_gene: Dict[str, Dict[str, Dict[int, float]]]) -> None:
        """
        Normalise `per_gene` into our internal representations. Accepts either
        gene→person or person→gene layouts; the canonical storage is gene-first.
        """
        keys = list(per_gene.keys())
        if not keys:
            return

        # Decide whether `per_gene` is gene-first or person-first.
        treat_as_gene_first = False
        if self._gene_order:
            treat_as_gene_first = all(g in per_gene for g in self._gene_order)
        else:
            sample_value = per_gene[keys[0]]
            if isinstance(sample_value, Mapping):
                inner_keys = list(sample_value.keys())
                treat_as_gene_first = inner_keys and isinstance(inner_keys[0], str)

        if treat_as_gene_first:
            for gene, per_person in per_gene.items():
                self._per_gene_gene_first[gene] = {
                    person: dict(state_probs) for person, state_probs in per_person.items()
                }
        else:
            for person, per_gene_map in per_gene.items():
                self._per_gene_person_first[person] = {
                    gene: dict(state_probs) for gene, state_probs in per_gene_map.items()
                }
            if self._gene_order:
                for gene in self._gene_order:
                    self._per_gene_gene_first.setdefault(gene, {})
            for person, gene_map in self._per_gene_person_first.items():
                for gene, state_probs in gene_map.items():
                    self._per_gene_gene_first.setdefault(gene, {})[person] = dict(state_probs)

        # Keep the public marginals aligned with the primary gene, if known.
        if self._gene_order and self._per_gene_gene_first:
            primary_gene = self._gene_order[0]
            primary_map = self._per_gene_gene_first.get(primary_gene, {})
            for person, probs in primary_map.items():
                self.marginals[person] = dict(probs)

    def __getitem__(self, key):
        return self.marginals[key]

    def __iter__(self):
        return iter(self.marginals)

    def __len__(self):
        return len(self.marginals)

    def has_tuple_pmfs(self) -> bool:
        return bool(getattr(self, "_tuple_pmfs", {}))

    def get_tuple_pmfs(self) -> Dict[str, Dict[Tuple[int, ...], float]]:
        return getattr(self, "_tuple_pmfs", {})

    def get_gene_order(self) -> Optional[Tuple[str, ...]]:
        return self._gene_order

    def get_per_gene_probs(self) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Return gene→person→{state:prob}. Lazily derives it from tuple PMFs if
        explicit per-gene marginals are not yet materialised.
        """
        if self._per_gene_gene_first:
            return self._per_gene_gene_first

        tuple_pmfs = getattr(self, "_tuple_pmfs", {})
        if not tuple_pmfs or not self._gene_order:
            return self._per_gene_gene_first

        # Build the set of genotype states from either supplied domain or data.
        if self._gen_states is not None:
            gen_states = self._gen_states
        else:
            state_values = set()
            for outcomes in tuple_pmfs.values():
                for assignment in outcomes:
                    state_values.update(assignment)
            gen_states = tuple(sorted(state_values))
            self._gen_states = gen_states

        per_gene_gene_first: Dict[str, Dict[str, Dict[int, float]]] = {
            gene: {} for gene in self._gene_order
        }
        for person, outcomes in tuple_pmfs.items():
            for gene in self._gene_order:
                per_gene_gene_first[gene].setdefault(
                    person, {state: 0.0 for state in gen_states}
                )
            for assignment, prob in outcomes.items():
                if prob <= 0.0:
                    continue
                for idx, gene in enumerate(self._gene_order):
                    value = assignment[idx]
                    per_gene_gene_first[gene][person][value] = (
                        per_gene_gene_first[gene][person].get(value, 0.0) + prob
                    )

        self._per_gene_gene_first = per_gene_gene_first
        # Keep marginals in sync with primary gene.
        primary_gene = self._gene_order[0]
        primary_map = self._per_gene_gene_first.get(primary_gene, {})
        for person, probs in primary_map.items():
            self.marginals[person] = dict(probs)
        return self._per_gene_gene_first

    def get_person_gene_marginals(self) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Return person→gene→{state:prob}. Derived from the gene-first store.
        """
        if self._per_gene_person_first:
            return self._per_gene_person_first

        per_gene = self.get_per_gene_probs()
        if not per_gene:
            return {}

        person_first: Dict[str, Dict[str, Dict[int, float]]] = {}
        for gene, per_person in per_gene.items():
            for person, probs in per_person.items():
                person_first.setdefault(person, {})[gene] = dict(probs)

        self._per_gene_person_first = person_first
        return self._per_gene_person_first

    def __eq__(self, other):
        if isinstance(other, InferenceResult):
            return (
                self.marginals == other.marginals
                and self.get_tuple_pmfs() == other.get_tuple_pmfs()
                and self.get_per_gene_probs() == other.get_per_gene_probs()
            )
        if isinstance(other, Mapping):
            return self.marginals == dict(other)
        return NotImplemented

    def __repr__(self):
        tuple_pmfs = getattr(self, "_tuple_pmfs", {})
        return (
            "InferenceResult("
            f"marginals={self.marginals}, "
            f"tuple_pmfs={tuple_pmfs}, "
            f"per_gene={self._per_gene_gene_first})"
        )

def sanitize_marginals(p_raw, evidence, eps=1e-12):
    """
    Replace NaN / ±inf with 0, then renormalise.
    p_raw : dict {g: prob}   (may contain nan)
    Returns a *new* dict summing to 1.0   OR raises ValueError if all probs vanish.
    """
    p = {g: (0.0 if (val is None or math.isnan(val) or math.isinf(val))
                   else max(val, 0.0))    # forbid negatives
         for g, val in p_raw.items()}

    tot = sum(p.values())
    if tot < eps:
        raise ValueError("Inconsistent evidence: all genotype probs collapsed to 0",evidence )

    return {g: v / tot for g, v in p.items()}

def propagate_all_marginals(infer, I, gen_states, evid):
    """
    evid: dict person→observed genotype (hard evidence)
    returns p_s: dict person→{g: P(person=g | evid)}
    """
    scalar_evid = _normalize_scalar_evidence(evid)
    if hasattr(infer, "posterior"):
        result = infer.posterior(scalar_evid)
        if isinstance(result, InferenceResult):
            return result
        per_gene = {person: {"gene": dict(probs)} for person, probs in result.items()}
        return InferenceResult(
            marginals=dict(result),
            per_gene=per_gene,
            gene_order=("gene",),
            gen_states=gen_states,
        )
    # for those *observed* it's a delta:
    p_s = {
        i: {g: 1.0 if g == scalar_evid[i] else 0.0 for g in gen_states}
        for i in scalar_evid
    }

    # now collect marginals for everyone else
    untested = [i for i in I if i not in scalar_evid]
    if untested:
        q = infer.query(variables=untested, evidence=scalar_evid, joint=False)
        # q could be a dict { var: DiscreteFactor } or a single DiscreteFactor
        for var in untested:
            factor = q[var] if isinstance(q, dict) else q
            p_s[var] = {g: float(factor.values[g]) for g in gen_states}

    per_gene = {
        person: {"gene": dict(probs)} for person, probs in p_s.items()
    }
    return InferenceResult(
        marginals=p_s,
        per_gene=per_gene,
        gene_order=("gene",),
        gen_states=gen_states,
    )


def propagate_multigene_marginals(infer, I, gen_states, evid, genes, *, aggregate_only=False):
    """
    Multi-gene aware posterior propagation.
    When aggregate_only=True, projects onto the first listed gene to maintain
    backward compatibility. When False, also returns joint tuple probabilities.
    """
    if not genes:
        return propagate_all_marginals(infer, I, gen_states, evid)
    if hasattr(infer, "posterior"):
        result = infer.posterior(evid)
        if not isinstance(result, InferenceResult):
            return InferenceResult(
                dict(result),
                gene_order=canonical_gene_order(genes),
                gen_states=gen_states,
            )
        if not aggregate_only:
            return result
        gene_order = canonical_gene_order(genes)
        primary_gene = gene_order[0]
        per_gene = result.get_per_gene_probs()
        primary_slice = per_gene.get(primary_gene, {})
        marginals = {person: dict(dist) for person, dist in primary_slice.items()}
        replicated = {gene: {person: dict(dist) for person, dist in marginals.items()} for gene in gene_order}
        return InferenceResult(
            marginals=marginals,
            per_gene=replicated,
            gene_order=gene_order,
            gen_states=gen_states,
        )

    gene_order: Sequence[str] = canonical_gene_order(genes)
    primary_gene = gene_order[0]
    gene_evidence = {}
    for person, g in evid.items():
        if isinstance(g, tuple):
            for idx, gene in enumerate(gene_order):
                value = g[idx] if idx < len(g) else g[0]
                gene_var = genotype_node_name(person, gene)
                gene_evidence[gene_var] = value
        else:
            for gene in gene_order:
                gene_var = genotype_node_name(person, gene)
                gene_evidence[gene_var] = g

    # Deterministic assignments for observed individuals
    marginals: Dict[str, Dict[int, float]] = {}
    per_gene_marginals: Dict[str, Dict[str, Dict[int, float]]] = {
        gene: {} for gene in gene_order
    }
    for person, g in evid.items():
        if isinstance(g, tuple):
            for idx, gene in enumerate(gene_order):
                value = g[idx] if idx < len(g) else g[0]
                per_gene_marginals[gene][person] = {
                    val: 1.0 if val == value else 0.0 for val in gen_states
                }
        else:
            for gene in gene_order:
                per_gene_marginals[gene][person] = {
                    val: 1.0 if val == g else 0.0 for val in gen_states
                }
        marginals[person] = dict(per_gene_marginals[primary_gene][person])

    untested = [i for i in I if i not in evid]
    if not untested:
        return InferenceResult(
            marginals=marginals,
            per_gene=per_gene_marginals,
            gene_order=gene_order,
            gen_states=gen_states,
        )

    if aggregate_only:
        query_nodes = [genotype_node_name(i, primary_gene) for i in untested]
        q = infer.query(variables=query_nodes, evidence=gene_evidence, joint=False)
        for node in query_nodes:
            person = node.split("::", 1)[0]
            factor = q[node] if isinstance(q, dict) else q
            marginal = {g: float(factor.values[g]) for g in gen_states}
            marginals[person] = marginal
            per_gene_marginals[primary_gene][person] = dict(marginal)
            # replicate onto other genes when only aggregate data is available
            for gene in gene_order[1:]:
                per_gene_marginals.setdefault(gene, {})
                per_gene_marginals[gene][person] = dict(marginal)
        return InferenceResult(
            marginals=marginals,
            per_gene=per_gene_marginals,
            gene_order=gene_order,
            gen_states=gen_states,
        )

    joint_variables = [genotype_node_name(i, gene) for i in untested for gene in gene_order]
    joint_query = infer.query(variables=joint_variables, evidence=gene_evidence, joint=True)
    joint_table = np.array(joint_query.values).reshape([len(gen_states)] * len(joint_variables))

    tuple_length = len(gene_order)
    tuple_pmfs: Dict[str, Dict[Tuple[int, ...], float]] = {person: {} for person in untested}
    for person in untested:
        marginals[person] = {g: 0.0 for g in gen_states}
        for gene in gene_order:
            per_gene_marginals[gene][person] = {state: 0.0 for state in gen_states}

    for index in np.ndindex(joint_table.shape):
        prob = float(joint_table[index])
        if prob <= 0.0:
            continue
        for person_idx, person in enumerate(untested):
            offset = person_idx * tuple_length
            tuple_vals = tuple(
                gen_states[index[offset + gene_idx]] for gene_idx in range(tuple_length)
            )
            tuple_pmfs[person][tuple_vals] = tuple_pmfs[person].get(tuple_vals, 0.0) + prob
            for gene_idx, gene in enumerate(gene_order):
                value = tuple_vals[gene_idx]
                per_gene_marginals[gene][person][value] += prob

    for person in untested:
        marginals[person] = dict(per_gene_marginals[primary_gene][person])

    return InferenceResult(
        marginals=marginals,
        tuple_pmfs=tuple_pmfs,
        per_gene=per_gene_marginals,
        gene_order=gene_order,
        gen_states=gen_states,
    )

def propagate_all_marginals_safe(infer, I, gen_states, evidence, eps=1e-12):
    scalar_evid = _normalize_scalar_evidence(evidence)
    if hasattr(infer, "posterior"):
        posterior = infer.posterior(scalar_evid)
        marginals = posterior.marginals if isinstance(posterior, InferenceResult) else dict(posterior)
        if any(sum(dist.values()) < eps for dist in marginals.values()):
            raise ValueError(f"Inconsistent evidence: P{evidence}=0")
        return {person: dict(dist) for person, dist in marginals.items()}
    # --- 1.  Feasibility check ------------------------------------
    prob_factor = infer.query(
        variables=[],           # ask for the scalar
        evidence=scalar_evid,
        joint=True              #  <-- key change
    )
    p_evid = float(prob_factor.values.item())     # scalar → float
    if p_evid < eps:
        raise ValueError(f"Inconsistent evidence: P{evidence}=0")
        # 2) Hard evidence → deterministic delta
    p_s = {}
    for person, g in scalar_evid.items():
        p_s[person] = {val: 1.0 if val == g else 0.0 for val in gen_states}

    # 3) Soft posteriors for the remaining people
    untested = [i for i in I if i not in scalar_evid]
    if untested:
        q = infer.query(variables=untested, evidence=scalar_evid, joint=False)
        for var in untested:
            factor = q[var] if isinstance(q, dict) else q
            p_s[var] = {g: float(factor.values[g]) for g in gen_states}

    return p_s

def ensure_belief(state, *, belief, infer, I, gen_states):
    """
    Guarantee `belief[state]` exists.
    If absent, build it from scratch via inference.
    """
    if state in belief:
        return

    evidence_state = state
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[0], frozenset)
        and isinstance(state[1], tuple)
    ):
        evidence_state = state[0]

    evidence = dict(evidence_state)              # {person: genotype, …}
    p_post   = propagate_all_marginals_safe(     # ← the NaN-proof wrapper
                    infer, I, gen_states, evidence)

    z_post   = { j: {g: 1.0 if evidence.get(j) == g else 0.0
                     for g in gen_states}
                 for j in I }

    belief[state] = (p_post, z_post)


def ensure_belief_with_tuples(
    state,
    *,
    belief,
    infer,
    I,
    gen_states,
    genes,
):
    """
    Guarantee `belief[state]` exists with tuple PMFs (multi-gene mode).
    If absent or missing tuple PMFs, recompute via multi-gene inference.
    """
    if state in belief:
        entry = belief[state]
        posterior = entry[0] if isinstance(entry, tuple) else entry
        if isinstance(posterior, InferenceResult) and posterior.has_tuple_pmfs():
            return

    evidence_state = state
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[0], frozenset)
        and isinstance(state[1], tuple)
    ):
        evidence_state = state[0]

    evidence = dict(evidence_state)
    posterior = propagate_multigene_marginals(
        infer,
        I,
        gen_states,
        evidence,
        genes,
        aggregate_only=False,
    )
    belief[state] = (posterior, {})

def lift_single_gene_posteriors_to_genes(
    posteriors: Dict[str, Dict[int, float]],
    genes: Optional[Iterable[str]],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Helper that embeds the legacy single-gene posterior structure into a per-gene
    container. Multi-gene models will replace the replication logic with true
    per-gene inference in later refactors.
    """
    if not genes:
        return {"gene": posteriors}
    gene_list = list(genes)
    if len(gene_list) == 1:
        return {gene_list[0]: posteriors}
    return {gene: {ind: probs.copy() for ind, probs in posteriors.items()} for gene in gene_list}

def ensure_gene_indexed_belief(
    state,
    *,
    belief,
    infer,
    I,
    gen_states,
    genes: Optional[Iterable[str]] = None,
):
    """
    Materialise a per-gene-view of the belief at state when the base solver has
    only computed single-gene posteriors. The result mirrors the existing belief
    tuple while supplying gene-indexed marginals for downstream consumers.
    """
    ensure_belief(state, belief=belief, infer=infer, I=I, gen_states=gen_states)
    posteriors, z_post = belief[state]
    gene_posteriors = lift_single_gene_posteriors_to_genes(posteriors, genes)
    return gene_posteriors, z_post
def _normalize_scalar_evidence(evid):
    normalized = {}
    for person, value in evid.items():
        if isinstance(value, tuple):
            if len(value) == 1:
                normalized[person] = value[0]
            else:
                raise ValueError(
                    f"Tuple evidence {value!r} received in scalar path; "
                    "multi-gene states should use propagate_multigene_marginals."
                )
        else:
            normalized[person] = value
    return normalized


def _extract_observed_value(value, gene_idx=None):
    if not isinstance(value, tuple):
        return value
    if gene_idx is None or gene_idx >= len(value):
        return value[0]
    return value[gene_idx]


# ---------------------------------------------------------------------------
# Pairwise (parent-child) marginals for edge features
# ---------------------------------------------------------------------------

def _compute_one_pairwise(infer, gen_states, query_evid, parent, child,
                          parent_var, child_var, raw_evid, gene_idx=None):
    """Compute joint P(g_parent, g_child | evidence) for a single edge.

    Returns dict {(g_p, g_c): float} over all genotype-state pairs.
    """
    parent_tested = parent in raw_evid
    child_tested = child in raw_evid

    if parent_tested and child_tested:
        obs_p = _extract_observed_value(raw_evid[parent], gene_idx=gene_idx)
        obs_c = _extract_observed_value(raw_evid[child], gene_idx=gene_idx)
        return {
            (gp, gc): (1.0 if gp == obs_p and gc == obs_c else 0.0)
            for gp in gen_states for gc in gen_states
        }

    # BeliefMapInference path: approximate as product of marginals.
    if hasattr(infer, "posterior"):
        result = infer.posterior(raw_evid)
        if isinstance(result, InferenceResult):
            marginals = result.marginals
        elif isinstance(result, Mapping):
            marginals = dict(result)
        else:
            marginals = {}
        uniform = {g: 1.0 / len(gen_states) for g in gen_states}
        p_parent = marginals.get(parent, uniform)
        p_child = marginals.get(child, uniform)
        return {
            (gp, gc): float(p_parent.get(gp, 0.0)) * float(p_child.get(gc, 0.0))
            for gp in gen_states for gc in gen_states
        }

    # pgmpy VariableElimination path
    query_vars = []
    if not parent_tested:
        query_vars.append(parent_var)
    if not child_tested:
        query_vars.append(child_var)

    if len(query_vars) == 2:
        # Neither tested: full joint query
        factor = infer.query(variables=[parent_var, child_var],
                             evidence=query_evid, joint=True)
        pair_dist = {}
        for gp in gen_states:
            for gc in gen_states:
                pair_dist[(gp, gc)] = float(factor.values[gp, gc])
        return pair_dist

    # One tested, one untested: marginal on untested conditioned on evidence
    factor = infer.query(variables=query_vars, evidence=query_evid, joint=False)
    if isinstance(factor, dict):
        factor = factor[query_vars[0]]
    if parent_tested:
        obs_p = query_evid[parent_var]
        return {
            (gp, gc): (float(factor.values[gc]) if gp == obs_p else 0.0)
            for gp in gen_states for gc in gen_states
        }
    else:
        obs_c = query_evid[child_var]
        return {
            (gp, gc): (float(factor.values[gp]) if gc == obs_c else 0.0)
            for gp in gen_states for gc in gen_states
        }


def _compute_one_trio(
    infer,
    gen_states,
    query_evid,
    parent1,
    parent2,
    child,
    parent1_var,
    parent2_var,
    child_var,
    raw_evid,
    gene_idx=None,
):
    members = (
        (parent1, parent1_var),
        (parent2, parent2_var),
        (child, child_var),
    )
    observed = {
        person: _extract_observed_value(raw_evid[person], gene_idx=gene_idx)
        for person, _ in members
        if person in raw_evid
    }

    if len(observed) == 3:
        return {
            (g_parent1, g_parent2, g_child): (
                1.0
                if g_parent1 == observed[parent1]
                and g_parent2 == observed[parent2]
                and g_child == observed[child]
                else 0.0
            )
            for g_parent1 in gen_states
            for g_parent2 in gen_states
            for g_child in gen_states
        }

    if hasattr(infer, "posterior"):
        result = infer.posterior(raw_evid)
        if isinstance(result, InferenceResult):
            marginals = result.marginals
        elif isinstance(result, Mapping):
            marginals = dict(result)
        else:
            marginals = {}
        uniform = {g: 1.0 / len(gen_states) for g in gen_states}
        p_parent1 = marginals.get(parent1, uniform)
        p_parent2 = marginals.get(parent2, uniform)
        p_child = marginals.get(child, uniform)
        return {
            (g_parent1, g_parent2, g_child): (
                float(p_parent1.get(g_parent1, 0.0))
                * float(p_parent2.get(g_parent2, 0.0))
                * float(p_child.get(g_child, 0.0))
            )
            for g_parent1 in gen_states
            for g_parent2 in gen_states
            for g_child in gen_states
        }

    query_vars = [var for person, var in members if person not in observed]
    factor_values = None
    if query_vars:
        factor = infer.query(variables=query_vars, evidence=query_evid, joint=True)
        factor_values = np.asarray(factor.values, dtype=float).reshape([len(gen_states)] * len(query_vars))
        query_index = {var: idx for idx, var in enumerate(query_vars)}
    else:
        query_index = {}

    trio_dist = {}
    for g_parent1, g_parent2, g_child in itertools.product(gen_states, repeat=3):
        assignment = {
            parent1_var: g_parent1,
            parent2_var: g_parent2,
            child_var: g_child,
        }
        inconsistent = False
        for person, var in members:
            if person in observed and assignment[var] != observed[person]:
                inconsistent = True
                break
        if inconsistent:
            trio_dist[(g_parent1, g_parent2, g_child)] = 0.0
            continue
        if not query_vars:
            trio_dist[(g_parent1, g_parent2, g_child)] = 1.0
            continue
        value_index = tuple(assignment[var] for var in query_vars)
        trio_dist[(g_parent1, g_parent2, g_child)] = float(factor_values[value_index])
    return trio_dist


def get_pairwise_marginals(infer, I, gen_states, evid, edges, genes=None):
    """Compute pairwise joint marginals P(g_parent, g_child | evidence) for
    each parent-child edge in the pedigree.

    Parameters
    ----------
    infer : pgmpy VariableElimination or BeliefMapInference
    I : list of individual names
    gen_states : list of genotype states (e.g. [0, 1, 2])
    evid : dict {person: observed_genotype} — current evidence
    edges : list of (parent, child) tuples from pedigree.graph.edges()
    genes : optional list of gene names for multi-gene mode

    Returns
    -------
    Single-gene: Dict[(parent, child): Dict[(g_p, g_c): float]]
    Multi-gene:  Dict[gene: Dict[(parent, child): Dict[(g_p, g_c): float]]]
    """
    if not edges:
        return {} if not genes else {gene: {} for gene in genes}

    if genes:
        result = {}
        gene_list = list(genes)
        for gene_idx, gene in enumerate(gene_list):
            gene_evid = {}
            for person, g in evid.items():
                if isinstance(g, tuple):
                    gene_evid[genotype_node_name(person, gene)] = (
                        g[gene_idx] if gene_idx < len(g) else g[0]
                    )
                else:
                    gene_evid[genotype_node_name(person, gene)] = g

            gene_pairs = {}
            for parent, child in edges:
                parent_var = genotype_node_name(parent, gene)
                child_var = genotype_node_name(child, gene)
                gene_pairs[(parent, child)] = _compute_one_pairwise(
                    infer, gen_states, gene_evid,
                    parent, child, parent_var, child_var,
                    evid,
                    gene_idx=gene_idx,
                )
            result[gene] = gene_pairs
        return result

    # Single-gene path
    scalar_evid = _normalize_scalar_evidence(evid)
    pair_marginals = {}
    for parent, child in edges:
        pair_marginals[(parent, child)] = _compute_one_pairwise(
            infer, gen_states, scalar_evid,
            parent, child, parent, child,
            evid,
        )
    return pair_marginals


def get_trio_marginals(infer, I, gen_states, evid, trios, genes=None):
    if not trios:
        return {} if not genes else {gene: {} for gene in genes}

    if genes:
        result = {}
        gene_list = list(genes)
        for gene_idx, gene in enumerate(gene_list):
            gene_evid = {}
            for person, g in evid.items():
                if isinstance(g, tuple):
                    gene_evid[genotype_node_name(person, gene)] = (
                        g[gene_idx] if gene_idx < len(g) else g[0]
                    )
                else:
                    gene_evid[genotype_node_name(person, gene)] = g

            gene_trios = {}
            for parent1, parent2, child in trios:
                parent1_var = genotype_node_name(parent1, gene)
                parent2_var = genotype_node_name(parent2, gene)
                child_var = genotype_node_name(child, gene)
                gene_trios[(parent1, parent2, child)] = _compute_one_trio(
                    infer,
                    gen_states,
                    gene_evid,
                    parent1,
                    parent2,
                    child,
                    parent1_var,
                    parent2_var,
                    child_var,
                    evid,
                    gene_idx=gene_idx,
                )
            result[gene] = gene_trios
        return result

    scalar_evid = _normalize_scalar_evidence(evid)
    trio_marginals = {}
    for parent1, parent2, child in trios:
        trio_marginals[(parent1, parent2, child)] = _compute_one_trio(
            infer,
            gen_states,
            scalar_evid,
            parent1,
            parent2,
            child,
            parent1,
            parent2,
            child,
            evid,
        )
    return trio_marginals
