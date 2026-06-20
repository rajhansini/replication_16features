from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Iterator, List, Sequence, Tuple

StateKey = Tuple[int, Tuple[int, ...]]


@dataclass(frozen=True)
class IndexedState:
    state_id: int
    mask: int
    value_indices: Tuple[int, ...]


class StateIndexer:
    """
    Enumerate and index every partial assignment state for a pedigree.

    Each state is represented by:
      • mask: bitset indicating which individuals are observed
      • value_indices: tuple of genotype indices aligned with ascending person order

    The mapping is deterministic (mask ascending, genotype combinations lexicographically).
    """

    def __init__(self, individuals: Sequence[str], gen_states: Sequence[Hashable]):
        if not individuals:
            raise ValueError("StateIndexer requires at least one individual.")
        if not gen_states:
            raise ValueError("StateIndexer requires at least one genotype state.")

        self._individuals: Tuple[str, ...] = tuple(individuals)
        self._person_to_bit: Dict[str, int] = {
            person: idx for idx, person in enumerate(self._individuals)
        }

        self._gen_states: Tuple[Hashable, ...] = tuple(
            self._canonical_value(value) for value in gen_states
        )
        self._gen_state_to_index: Dict[Hashable, int] = {
            value: idx for idx, value in enumerate(self._gen_states)
        }

        self._state_lookup: Dict[StateKey, int] = {}
        self._id_to_key: List[StateKey] = []
        self._mask_cache: Dict[int, Tuple[int, ...]] = {}
        self._mask_offsets: Dict[int, Tuple[int, int]] = {}
        self._built = False

    @staticmethod
    def _canonical_value(value: Hashable) -> Hashable:
        if isinstance(value, list):
            return tuple(value)
        return value

    def _mask_to_indices(self, mask: int) -> Tuple[int, ...]:
        cached = self._mask_cache.get(mask)
        if cached is not None:
            return cached
        indices = tuple(
            idx
            for idx in range(len(self._individuals))
            if mask & (1 << idx)
        )
        self._mask_cache[mask] = indices
        return indices

    def _register(self, mask: int, value_indices: Tuple[int, ...]) -> None:
        state_id = len(self._id_to_key)
        key = (mask, value_indices)
        self._state_lookup[key] = state_id
        self._id_to_key.append(key)

    def _build_index(self) -> None:
        if self._built:
            return
        num_individuals = len(self._individuals)
        genotype_range = range(len(self._gen_states))
        for mask in range(1 << num_individuals):
            subset_indices = self._mask_to_indices(mask)
            subset_len = len(subset_indices)
            start_idx = len(self._id_to_key)
            if subset_len == 0:
                self._register(mask, ())
            else:
                for value_indices in itertools.product(genotype_range, repeat=subset_len):
                    self._register(mask, tuple(value_indices))
            span = len(self._id_to_key) - start_idx
            self._mask_offsets[mask] = (start_idx, span)
        self._built = True

    @property
    def total_states(self) -> int:
        self._build_index()
        return len(self._id_to_key)

    @property
    def individuals(self) -> Tuple[str, ...]:
        return self._individuals

    @property
    def gen_states(self) -> Tuple[Hashable, ...]:
        return self._gen_states

    def iter_indexed_states(self) -> Iterator[IndexedState]:
        self._build_index()
        for state_id, (mask, value_indices) in enumerate(self._id_to_key):
            yield IndexedState(state_id=state_id, mask=mask, value_indices=value_indices)

    def iter_masks(self) -> Iterator[int]:
        num_individuals = len(self._individuals)
        for mask in range(1 << num_individuals):
            yield mask

    def mask_offset(self, mask: int) -> Tuple[int, int]:
        self._build_index()
        try:
            return self._mask_offsets[mask]
        except KeyError as exc:
            raise KeyError(f"Mask {mask} not present in indexer.") from exc

    def mask_metadata(self):
        self._build_index()
        metadata = []
        for mask in self.iter_masks():
            start, span = self._mask_offsets.get(mask, (0, 0))
            subset_indices = self._mask_to_indices(mask)
            metadata.append(
                {
                    "mask": mask,
                    "subset_indices": subset_indices,
                    "start": start,
                    "span": span,
                }
            )
        return metadata

    def materialize(self, indexed_state: IndexedState) -> frozenset:
        subset_indices = self._mask_to_indices(indexed_state.mask)
        assignments = []
        for idx, value_idx in zip(subset_indices, indexed_state.value_indices):
            person = self._individuals[idx]
            genotype = self._gen_states[value_idx]
            assignments.append((person, genotype))
        return frozenset(assignments)

    def materialize_by_id(self, state_id: int) -> frozenset:
        self._build_index()
        mask, value_indices = self._id_to_key[state_id]
        indexed_state = IndexedState(state_id=state_id, mask=mask, value_indices=value_indices)
        return self.materialize(indexed_state)

    def key_from_state(self, state: frozenset) -> StateKey:
        mask = 0
        ordered = sorted(state, key=lambda pair: self._person_to_bit[pair[0]])
        value_indices: List[int] = []
        for person, genotype in ordered:
            if person not in self._person_to_bit:
                raise KeyError(f"Unknown individual '{person}' in state {state!r}")
            bit = self._person_to_bit[person]
            mask |= 1 << bit
            canonical_value = self._canonical_value(genotype)
            try:
                value_idx = self._gen_state_to_index[canonical_value]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown genotype value {genotype!r} for person '{person}'"
                ) from exc
            value_indices.append(value_idx)
        return mask, tuple(value_indices)

    def state_id(self, state: frozenset) -> int:
        key = self.key_from_state(state)
        self._build_index()
        try:
            return self._state_lookup[key]
        except KeyError as exc:
            raise KeyError(f"State {state!r} not found in index") from exc

    def state_key(self, state_id: int) -> StateKey:
        self._build_index()
        return self._id_to_key[state_id]

    def state_id_from_mask_values(self, mask: int, value_indices: Tuple[int, ...]) -> int:
        self._build_index()
        start, span = self._mask_offsets[mask]
        subset_len = len(self._mask_to_indices(mask))
        if len(value_indices) != subset_len:
            raise ValueError(
                f"Value tuple length {len(value_indices)} incompatible with mask {mask}"
            )
        if subset_len == 0:
            return start
        base = len(self._gen_states)
        offset = 0
        for value_idx in value_indices:
            if value_idx < 0 or value_idx >= base:
                raise ValueError(f"Genotype index {value_idx} out of bounds for base {base}")
            offset = offset * base + value_idx
        if offset >= span:
            raise ValueError(f"Computed offset {offset} exceeds span {span} for mask {mask}")
        return start + offset

    def subset_indices(self, mask: int) -> Tuple[int, ...]:
        return self._mask_to_indices(mask)

    def encode_assignment(self, assignment: Sequence[Hashable]) -> Tuple[int, ...]:
        if len(assignment) != len(self._individuals):
            raise ValueError(
                f"Assignment length {len(assignment)} does not match individuals {len(self._individuals)}"
            )
        encoded = []
        for value in assignment:
            canonical = self._canonical_value(value)
            try:
                encoded.append(self._gen_state_to_index[canonical])
            except KeyError as exc:
                raise KeyError(f"Unknown genotype value {value!r}") from exc
        return tuple(encoded)

    def successor_state_id(self, state_id: int, person: str, genotype: Hashable) -> int:
        self._build_index()
        if person not in self._person_to_bit:
            raise KeyError(f"Unknown individual '{person}'")
        mask, value_indices = self._id_to_key[state_id]
        person_bit = self._person_to_bit[person]
        if mask & (1 << person_bit):
            raise ValueError(f"Person '{person}' already observed in state_id={state_id}")

        canonical_value = self._canonical_value(genotype)
        try:
            genotype_idx = self._gen_state_to_index[canonical_value]
        except KeyError as exc:
            raise KeyError(f"Unknown genotype value {genotype!r}") from exc

        new_mask = mask | (1 << person_bit)
        subset_indices = self._mask_to_indices(new_mask)

        new_values: List[int] = []
        existing_iter = iter(value_indices)
        for idx in subset_indices:
            if idx == person_bit:
                new_values.append(genotype_idx)
            else:
                try:
                    new_values.append(next(existing_iter))
                except StopIteration as exc:
                    raise RuntimeError("Inconsistent state value length encountered.") from exc
        key = (new_mask, tuple(new_values))
        try:
            return self._state_lookup[key]
        except KeyError as exc:
            raise KeyError(f"Successor state not found for key={key}") from exc
