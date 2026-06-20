"""
Utilities for representing multi-gene outcomes and converting between
dictionary and tuple-based encodings. These helpers are the foundation for
lifting the existing single-gene state representation to tuple-valued outcomes
without changing the rest of the solver yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple, FrozenSet


GeneOrder = Tuple[str, ...]
GenotypeTuple = Tuple[int, ...]
PersonOutcome = Tuple[str, GenotypeTuple]


def canonical_gene_order(genes: Iterable[str]) -> GeneOrder:
    """
    Produce a deterministic gene ordering for tuple encodings.

    Parameters
    ----------
    genes:
        Iterable of gene names.
    """
    return tuple(sorted(genes))


def encode_gene_tuple(
    gene_prob: Mapping[str, int],
    gene_order: Sequence[str],
) -> GenotypeTuple:
    """
    Encode a per-gene genotype dictionary into a tuple that respects the
    canonical gene ordering.
    """
    return tuple(gene_prob.get(gene) for gene in gene_order)


def decode_gene_tuple(
    encoded: GenotypeTuple,
    gene_order: Sequence[str],
) -> Dict[str, int]:
    """
    Decode a tuple representation back into a dictionary keyed by gene name.
    """
    return {gene: encoded[idx] for idx, gene in enumerate(gene_order)}


def encode_person_outcome(
    person: str,
    gene_prob: Mapping[str, int],
    gene_order: Sequence[str],
) -> PersonOutcome:
    """
    Pack a person's per-gene genotype assignment into the canonical tuple form.
    """
    return person, encode_gene_tuple(gene_prob, gene_order)


def decode_person_outcome(
    outcome: PersonOutcome,
    gene_order: Sequence[str],
) -> Tuple[str, Dict[str, int]]:
    """
    Unpack a person's tuple-based genotype assignment.
    """
    person, encoded = outcome
    return person, decode_gene_tuple(encoded, gene_order)


def encode_state_evidence(
    evidence: Mapping[Tuple[str, str], int],
    gene_order: Sequence[str],
) -> FrozenSet[PersonOutcome]:
    """
    Convert evidence of the form { (person, gene): genotype } into the
    tuple-based frozenset representation compatible with canonicalization.
    """
    per_person: Dict[str, Dict[str, int]] = {}
    for (person, gene), outcome in evidence.items():
        per_person.setdefault(person, {})[gene] = outcome
    encoded = {
        encode_person_outcome(person, gene_map, gene_order)
        for person, gene_map in per_person.items()
    }
    return frozenset(encoded)


def decode_state_evidence(
    state: FrozenSet[PersonOutcome],
    gene_order: Sequence[str],
) -> Dict[Tuple[str, str], int]:
    """
    Convert the tuple-based state back into {(person, gene): genotype} evidence.
    """
    evidence: Dict[Tuple[str, str], int] = {}
    for person, gene_tuple in state:
        decoded = decode_gene_tuple(gene_tuple, gene_order)
        for gene, value in decoded.items():
            evidence[(person, gene)] = value
    return evidence


@dataclass(frozen=True)
class OutcomeKey:
    """
    Hashable wrapper for a person's multi-gene outcome. This is a convenience
    layer that can be used in caches without exposing implementation details.
    """

    person: str
    outcome: GenotypeTuple

    def as_tuple(self) -> PersonOutcome:
        return (self.person, self.outcome)


def _normalize_outcome_for_projection(
    outcome: Mapping[str, int] | Sequence[int] | int,
    gene_order: Sequence[str],
) -> GenotypeTuple:
    """
    Normalise a raw outcome into a tuple aligned with the provided gene order.
    """
    if isinstance(outcome, Mapping):
        return encode_gene_tuple(outcome, gene_order)

    if isinstance(outcome, tuple):
        if len(outcome) == 1 and len(gene_order) == 1:
            return (outcome[0],)
        if len(outcome) != len(gene_order):
            raise ValueError(
                f"Outcome length {len(outcome)} does not match gene order {len(gene_order)}"
            )
        return tuple(outcome)

    if len(gene_order) == 1:
        return (outcome,)  # type: ignore[return-value]

    raise ValueError(
        "Scalar outcome provided for multiple genes; expected tuple or mapping."
    )


def project_state_by_gene(
    state: FrozenSet[PersonOutcome],
    gene_order: Sequence[str],
) -> Dict[str, FrozenSet[Tuple[str, int]]]:
    """
    Project a tuple-valued state onto per-gene sub-states.

    Returns a mapping gene -> frozenset({(person, genotype)}), using a
    deterministic person ordering within each projection.
    """
    ordered_genes = tuple(gene_order)
    projections: Dict[str, list[Tuple[str, int]]] = {gene: [] for gene in ordered_genes}

    for person, raw_outcome in state:
        normalized = _normalize_outcome_for_projection(raw_outcome, ordered_genes)
        for idx, gene in enumerate(ordered_genes):
            projections[gene].append((person, normalized[idx]))

    return {
        gene: frozenset(sorted(assignments, key=lambda pair: pair[0]))
        for gene, assignments in projections.items()
    }


def project_successor_by_gene(
    state: FrozenSet[PersonOutcome],
    person: str,
    outcome: Mapping[str, int] | Sequence[int] | int,
    gene_order: Sequence[str],
) -> Dict[str, FrozenSet[Tuple[str, int]]]:
    """
    Project a successor (state ∪ {(person, outcome)}) into per-gene sub-states.
    """
    base_projection = {
        gene: set(assignments) for gene, assignments in project_state_by_gene(state, gene_order).items()
    }
    ordered_genes = tuple(gene_order)
    normalized = _normalize_outcome_for_projection(outcome, ordered_genes)

    for idx, gene in enumerate(ordered_genes):
        assignments = base_projection.get(gene)
        if assignments is None:
            assignments = set()
            base_projection[gene] = assignments
        assignments = {pair for pair in assignments if pair[0] != person}
        assignments.add((person, normalized[idx]))
        base_projection[gene] = assignments

    return {
        gene: frozenset(sorted(assignments, key=lambda pair: pair[0]))
        for gene, assignments in base_projection.items()
    }
