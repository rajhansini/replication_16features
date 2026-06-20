from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .belief import InferenceResult


State = frozenset[tuple[str, Any]]
T = TypeVar("T")


class BeliefAccessor(Mapping):
    """Mapping-compatible belief access with explicit posterior helpers."""

    storage_mode = "abstract"

    def posterior(self, state: State) -> InferenceResult:
        return self[state]

    def per_gene_probs(self, state: State) -> dict[str, dict[str, dict[int, float]]]:
        return self.posterior(state).get_per_gene_probs()

    def tuple_dist(self, state: State, person: str) -> dict[Any, float]:
        return dict(self.posterior(state).get_tuple_pmfs().get(person, {}))

    def metadata(self) -> dict[str, Any]:
        return {"storage_mode": self.storage_mode, "state_count": len(self)}

    def materialized_result_count(self) -> int:
        return 0


class DictBeliefAccessor(BeliefAccessor):
    """BeliefAccessor wrapper for the current materialized dict representation."""

    storage_mode = "dict"

    def __init__(self, belief: Mapping[State, Any]):
        self._belief = belief

    def __getitem__(self, state: State) -> InferenceResult:
        entry = self._belief[state]
        if isinstance(entry, InferenceResult):
            return entry
        if isinstance(entry, tuple) and entry and isinstance(entry[0], InferenceResult):
            return entry[0]
        if isinstance(entry, Mapping):
            return InferenceResult({person: dict(probs) for person, probs in entry.items()})
        raise TypeError(f"Unsupported belief entry for {state!r}: {type(entry)!r}")

    def __iter__(self) -> Iterator[State]:
        return iter(self._belief)

    def __len__(self) -> int:
        return len(self._belief)

    def __contains__(self, state: object) -> bool:
        return state in self._belief

    def metadata(self) -> dict[str, Any]:
        return {
            "storage_mode": self.storage_mode,
            "state_count": len(self),
            "materialized_result_count": len(self._belief),
        }

    def materialized_result_count(self) -> int:
        return len(self._belief)


def _observed_people(state: State) -> tuple[str, ...]:
    return tuple(sorted(person for person, _ in state))


def _mask_counts(belief: Mapping[State, Any]) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for state in belief:
        mask = _observed_people(state)
        counts[mask] = counts.get(mask, 0) + 1
    return counts


class FactorizedBeliefAccessor(BeliefAccessor):
    """
    Lazy multigene belief accessor backed by per-gene scalar belief maps.

    The accessor counts and accesses combined states without storing a
    dict[state, InferenceResult] for the full multigene state universe.
    """

    storage_mode = "factorized_compact"

    def __init__(
        self,
        *,
        individuals: Iterable[str],
        genes: Iterable[str],
        gen_states: Iterable[int],
        single_gene_beliefs: Mapping[str, Mapping[State, Any]],
        cache_limit: int = 256,
    ):
        self.individuals = tuple(individuals)
        self.genes = tuple(genes)
        self.gen_states = tuple(gen_states)
        self.single_gene_beliefs = {gene: dict(belief) for gene, belief in single_gene_beliefs.items()}
        missing = [gene for gene in self.genes if gene not in self.single_gene_beliefs]
        if missing:
            raise ValueError(f"missing scalar belief maps for genes: {missing}")
        self.exact_gen_states = tuple(itertools.product(self.gen_states, repeat=len(self.genes)))
        self._mask_counts_by_gene = {
            gene: _mask_counts(belief) for gene, belief in self.single_gene_beliefs.items()
        }
        self._state_count_by_mask = self._compute_state_count_by_mask()
        self._state_count = sum(self._state_count_by_mask.values())
        self._cache_limit = max(0, int(cache_limit))
        self._cache: dict[State, InferenceResult] = {}
        self._cache_order: list[State] = []

    def _compute_state_count_by_mask(self) -> dict[tuple[str, ...], int]:
        masks: set[tuple[str, ...]] = set()
        for counts in self._mask_counts_by_gene.values():
            masks.update(counts)
        out: dict[tuple[str, ...], int] = {}
        for mask in masks:
            product = 1
            for gene in self.genes:
                product *= self._mask_counts_by_gene[gene].get(mask, 0)
            if product:
                out[mask] = product
        return dict(sorted(out.items(), key=lambda item: (len(item[0]), item[0])))

    def __len__(self) -> int:
        return self._state_count

    def __iter__(self) -> Iterator[State]:
        return self.iter_states()

    def __contains__(self, state: object) -> bool:
        if not isinstance(state, frozenset):
            return False
        try:
            self._project_state(state)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _project_state(self, state: State) -> dict[str, State]:
        projected: dict[str, dict[str, int]] = {gene: {} for gene in self.genes}
        for person, outcome in state:
            if person not in self.individuals:
                raise KeyError(person)
            if len(tuple(outcome)) != len(self.genes):
                raise ValueError(f"state outcome {outcome!r} does not match genes {self.genes!r}")
            for idx, gene in enumerate(self.genes):
                projected[gene][person] = int(tuple(outcome)[idx])
        return {gene: frozenset(values.items()) for gene, values in projected.items()}

    def __getitem__(self, state: State) -> InferenceResult:
        if state in self._cache:
            return self._cache[state]
        projected = self._project_state(state)
        per_gene_probs: dict[str, dict[str, dict[int, float]]] = {}
        for gene in self.genes:
            scalar_state = projected[gene]
            scalar_entry = self.single_gene_beliefs[gene][scalar_state]
            if isinstance(scalar_entry, InferenceResult):
                per_gene_probs[gene] = {
                    person: dict(probs) for person, probs in scalar_entry.marginals.items()
                }
            elif isinstance(scalar_entry, Mapping):
                per_gene_probs[gene] = {person: dict(probs) for person, probs in scalar_entry.items()}
            else:
                raise TypeError(f"unsupported scalar belief entry: {type(scalar_entry)!r}")

        tuple_pmfs: dict[str, dict[Any, float]] = {}
        for person in self.individuals:
            pmf: dict[Any, float] = {}
            total = 0.0
            for outcome in self.exact_gen_states:
                prob = 1.0
                for idx, gene in enumerate(self.genes):
                    prob *= float(per_gene_probs[gene][person].get(outcome[idx], 0.0))
                    if prob <= 0.0:
                        break
                pmf[outcome] = prob
                total += prob
            if total > 0.0 and abs(total - 1.0) > 1e-12:
                pmf = {outcome: prob / total for outcome, prob in pmf.items()}
            tuple_pmfs[person] = pmf

        primary_gene = self.genes[0]
        result = InferenceResult(
            marginals={person: dict(per_gene_probs[primary_gene][person]) for person in self.individuals},
            tuple_pmfs=tuple_pmfs,
            per_gene=per_gene_probs,
            gene_order=self.genes,
            gen_states=self.gen_states,
        )
        if self._cache_limit:
            self._cache[state] = result
            self._cache_order.append(state)
            while len(self._cache_order) > self._cache_limit:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return result

    def iter_states(self, *, max_states: int | None = None) -> Iterator[State]:
        yielded = 0
        scalar_states_by_gene = {
            gene: list(self.single_gene_beliefs[gene].keys()) for gene in self.genes
        }
        by_gene_mask: dict[str, dict[tuple[str, ...], list[State]]] = {}
        for gene, states in scalar_states_by_gene.items():
            mask_map: dict[tuple[str, ...], list[State]] = {}
            for state in states:
                mask_map.setdefault(_observed_people(state), []).append(state)
            by_gene_mask[gene] = mask_map

        for mask in self._state_count_by_mask:
            groups = [by_gene_mask[gene].get(mask, []) for gene in self.genes]
            for combo in itertools.product(*groups):
                person_outcomes: dict[str, list[int]] = {person: [] for person in mask}
                for gene_state in combo:
                    for person, scalar_outcome in gene_state:
                        person_outcomes[person].append(int(scalar_outcome))
                yield frozenset((person, tuple(values)) for person, values in person_outcomes.items())
                yielded += 1
                if max_states is not None and yielded >= max_states:
                    return

    def state_count_by_mask(self) -> dict[tuple[str, ...], int]:
        return dict(self._state_count_by_mask)

    def materialized_result_count(self) -> int:
        return len(self._cache)

    def metadata(self) -> dict[str, Any]:
        scalar_state_counts = {
            gene: len(self.single_gene_beliefs[gene]) for gene in self.genes
        }
        return {
            "storage_mode": self.storage_mode,
            "state_count": len(self),
            "gene_count": len(self.genes),
            "individual_count": len(self.individuals),
            "tuple_outcome_count": len(self.exact_gen_states),
            "scalar_state_counts": scalar_state_counts,
            "mask_count": len(self._state_count_by_mask),
            "materialized_result_count": self.materialized_result_count(),
            "cache_limit": self._cache_limit,
        }


@dataclass(frozen=True)
class PhiAccessor(Generic[T]):
    values: Mapping[T, float]

    def __getitem__(self, key: T) -> float:
        return float(self.values[key])

    def get(self, key: T, default: float | None = None) -> float | None:
        value = self.values.get(key, default)  # type: ignore[arg-type]
        return None if value is None else float(value)


@dataclass(frozen=True)
class PolicyAccessor(Generic[T]):
    actions: Mapping[T, Any]

    def __getitem__(self, key: T) -> Any:
        return self.actions[key]

    def get(self, key: T, default: Any = None) -> Any:
        return self.actions.get(key, default)  # type: ignore[arg-type]
