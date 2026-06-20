from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

from ..models.belief import InferenceResult
from ..models.outcomes import canonical_gene_order


@dataclass(frozen=True)
class SimpleBayesianNetwork:
    """
    Minimal BN container used by DFVR feasibility checks.

    The DFVR implementation only requires a `get_cpds(node)` method returning an
    object with `values`, `variable_card`, and `evidence` (or `get_evidence()`).
    """

    cpds_by_var: Mapping[str, Any]

    def get_cpds(self, node: str) -> Any:
        return self.cpds_by_var.get(node)


class BeliefMapInference:
    """
    Lightweight inference facade backed by a fully materialized belief map.

    This is a pgmpy-free fallback used when pgmpy/pandas aren't available.
    Callers should prefer `posterior(evidence)` via the belief propagation helpers.
    """

    def __init__(
        self,
        belief_by_state: Mapping[frozenset, InferenceResult],
        *,
        genes: Optional[Sequence[str]] = None,
        model: Any = None,
    ):
        self._belief_by_state = belief_by_state
        self._gene_order: Tuple[str, ...] = tuple(canonical_gene_order(genes)) if genes else tuple()
        self.model = model

    def posterior(self, evidence: Mapping[Hashable, Any]) -> InferenceResult:
        """
        Return posterior marginals (and optional tuple PMFs) for `evidence`.

        Evidence is keyed by person name; values may be:
        - scalar genotype (0/1/2)
        - tuple/list genotypes per gene in `gene_order`
        """
        state = self._canonical_state(evidence)
        posterior = self._belief_by_state.get(state)
        if posterior is None:
            raise KeyError(f"Evidence state not found in belief map: {dict(state)}")
        return posterior

    def _canonical_state(self, evidence: Mapping[Hashable, Any]) -> frozenset:
        if not evidence:
            return frozenset()

        items: Dict[Hashable, Any] = {}
        for person, raw_value in evidence.items():
            if raw_value is None:
                continue
            if self._gene_order:
                items[person] = self._coerce_tuple_outcome(raw_value)
            else:
                if isinstance(raw_value, (list, tuple)):
                    if len(raw_value) != 1:
                        raise ValueError(f"Scalar evidence expects 1-tuple, got {raw_value!r}")
                    raw_value = raw_value[0]
                items[person] = int(raw_value)
        return frozenset(items.items())

    def _coerce_tuple_outcome(self, value: Any) -> Tuple[int, ...]:
        if isinstance(value, list):
            value = tuple(value)
        if isinstance(value, tuple):
            if not value:
                raise ValueError("Empty tuple evidence is invalid in multi-gene mode")
            if len(value) == 1:
                return tuple(int(value[0]) for _ in self._gene_order)
            if len(value) < len(self._gene_order):
                # Pad by repeating the first component to keep evidence deterministic.
                padded = tuple(int(value[idx] if idx < len(value) else value[0]) for idx in range(len(self._gene_order)))
                return padded
            return tuple(int(value[idx]) for idx in range(len(self._gene_order)))
        if isinstance(value, (int, float)):
            scalar = int(value)
            return tuple(scalar for _ in self._gene_order)
        raise TypeError(f"Unsupported evidence value {value!r} for multi-gene inference")

