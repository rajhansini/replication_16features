from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from ..models import belief as belief_mod
from ..models.belief import InferenceResult


Trio = Tuple[str, str, str]
Pair = Tuple[str, str]
GenotypeTriple = Tuple[int, int, int]


def _state_key(evidence_state: object) -> frozenset:
    if isinstance(evidence_state, frozenset):
        return evidence_state
    if isinstance(evidence_state, Mapping):
        return frozenset(dict(evidence_state).items())
    raise TypeError(f"Unsupported evidence_state type: {type(evidence_state).__name__}")


def _posterior_marginals(posterior_entry: object) -> Mapping[str, Mapping[int, float]]:
    if isinstance(posterior_entry, InferenceResult):
        return posterior_entry.marginals
    if isinstance(posterior_entry, Mapping):
        return posterior_entry
    if hasattr(posterior_entry, "marginals"):
        return getattr(posterior_entry, "marginals")
    raise TypeError(f"Unsupported posterior entry type: {type(posterior_entry).__name__}")


def _per_gene_probs(
    posterior_entry: object,
    genes: Sequence[str],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    if not genes:
        return {}
    if isinstance(posterior_entry, InferenceResult):
        return posterior_entry.get_per_gene_probs() or {}
    if hasattr(posterior_entry, "get_per_gene_probs"):
        return getattr(posterior_entry, "get_per_gene_probs")() or {}
    return {}


def _normalise_trio(value: object) -> Optional[Trio]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return tuple(value)  # type: ignore[return-value]
    return None


def _normalise_share_key(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def resolve_trio_blocks(payload: object) -> Dict[str, Mapping]:
    if not isinstance(payload, Mapping):
        return {}
    blocks = payload.get("__blocks__", ())
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        blocks = ()
    resolved: Dict[str, Mapping] = {}
    for block in blocks:
        block_map = payload.get(block)
        if isinstance(block_map, Mapping):
            resolved[str(block)] = block_map
    if not resolved:
        for candidate in ("raw", "pure3"):
            block_map = payload.get(candidate)
            if isinstance(block_map, Mapping):
                resolved[candidate] = block_map
    if not resolved and "__mode__" not in payload:
        return {"raw": payload}
    return resolved


def build_trio_share_key_map(
    pedigree_trios: Sequence[Trio],
    *,
    trio_sharing: str = "free",
    pedigree_depths: Optional[Mapping[str, int]] = None,
    share_key_entries: Optional[Iterable[object]] = None,
) -> Tuple[Dict[Trio, object], Tuple[object, ...]]:
    share_key_map: Dict[Trio, object] = {}
    if share_key_entries is not None:
        for entry in share_key_entries:
            if not isinstance(entry, Mapping):
                continue
            trio = _normalise_trio(entry.get("trio"))
            if trio is None:
                continue
            share_key_map[trio] = _normalise_share_key(entry.get("share_key"))

    depths = pedigree_depths or {}
    for trio in pedigree_trios:
        if trio in share_key_map:
            continue
        if trio_sharing == "child_depth":
            share_key_map[trio] = int(depths.get(trio[2], 0))
        else:
            share_key_map[trio] = trio

    ordered_share_keys = []
    seen = set()
    for trio in pedigree_trios:
        share_key = share_key_map[trio]
        if share_key in seen:
            continue
        seen.add(share_key)
        ordered_share_keys.append(share_key)
    return share_key_map, tuple(ordered_share_keys)


def compute_trio_pure3_value(
    *,
    joint_prob: float,
    fm_prob: float,
    fc_prob: float,
    mc_prob: float,
    parent1_prob: float,
    parent2_prob: float,
    child_prob: float,
) -> float:
    return (
        joint_prob
        - fm_prob * child_prob
        - fc_prob * parent2_prob
        - mc_prob * parent1_prob
        + 2.0 * parent1_prob * parent2_prob * child_prob
    )


def _empty_bundle(
    *,
    evidence_state: frozenset,
    pedigree_trios: Sequence[Trio],
    genes: Sequence[str],
    gen_states: Sequence[int],
    trio_sharing: str,
    share_key_map: Dict[Trio, object],
    share_keys: Tuple[object, ...],
    materialization_sec: float,
    cache_hits: int,
    cache_misses: int,
) -> Dict[str, object]:
    return {
        "evidence_state": evidence_state,
        "trios": tuple(pedigree_trios),
        "pairs": tuple(),
        "genes": tuple(genes),
        "gen_states": tuple(gen_states),
        "sharing": trio_sharing,
        "share_key_map": share_key_map,
        "share_keys": share_keys,
        "pairwise_marginals": {} if genes else {},
        "trio_marginals": {} if genes else {},
        "pure3": {} if genes else {},
        "cache": {
            "hits": int(cache_hits),
            "misses": int(cache_misses),
        },
        "telemetry": {
            "materialization_sec": float(materialization_sec),
        },
    }


def build_state_feature_bundle(
    *,
    infer: object,
    individuals: Sequence[str],
    gen_states: Sequence[int],
    evidence_state: object,
    pedigree_trios: Sequence[Trio],
    posterior_entry: object,
    genes: Optional[Sequence[str]] = None,
    feature_cache: Optional[dict] = None,
    trio_sharing: str = "free",
    pedigree_depths: Optional[Mapping[str, int]] = None,
    share_key_map: Optional[Mapping[Trio, object]] = None,
) -> Dict[str, object]:
    state_key = _state_key(evidence_state)
    gene_list = tuple(genes) if genes else tuple()
    trio_list = tuple(tuple(trio) for trio in pedigree_trios)
    if share_key_map is None:
        resolved_share_key_map, share_keys = build_trio_share_key_map(
            trio_list,
            trio_sharing=trio_sharing,
            pedigree_depths=pedigree_depths,
        )
    else:
        resolved_share_key_map = {tuple(trio): _normalise_share_key(value) for trio, value in share_key_map.items()}
        _resolved, share_keys = build_trio_share_key_map(
            trio_list,
            trio_sharing=trio_sharing,
            pedigree_depths=pedigree_depths,
            share_key_entries=(
                {"trio": list(trio), "share_key": resolved_share_key_map[trio]}
                for trio in trio_list
            ),
        )
        resolved_share_key_map = _resolved

    if not trio_list:
        return _empty_bundle(
            evidence_state=state_key,
            pedigree_trios=trio_list,
            genes=gene_list,
            gen_states=gen_states,
            trio_sharing=trio_sharing,
            share_key_map=resolved_share_key_map,
            share_keys=share_keys,
            materialization_sec=0.0,
            cache_hits=0,
            cache_misses=0,
        )

    trio_pairs = []
    trio_pair_seen = set()
    for parent1, parent2, child in trio_list:
        for pair in ((parent1, parent2), (parent1, child), (parent2, child)):
            if pair in trio_pair_seen:
                continue
            trio_pair_seen.add(pair)
            trio_pairs.append(pair)

    cache_key = (
        "state_feature_bundle",
        state_key,
        tuple(individuals),
        tuple(gen_states),
        trio_list,
        tuple(trio_pairs),
        gene_list,
        trio_sharing,
        tuple(resolved_share_key_map[trio] for trio in trio_list),
    )
    if isinstance(feature_cache, dict) and cache_key in feature_cache:
        cached = feature_cache[cache_key]
        payload = dict(cached)
        payload["cache"] = {"hits": 1, "misses": 0}
        payload["telemetry"] = {
            **dict(cached.get("telemetry", {})),
            "materialization_sec": 0.0,
        }
        return payload

    started_at = time.perf_counter()
    evidence = dict(state_key)
    marginals = _posterior_marginals(posterior_entry)
    per_gene = _per_gene_probs(posterior_entry, gene_list)

    trio_marginals = belief_mod.get_trio_marginals(
        infer,
        list(individuals),
        list(gen_states),
        evidence,
        list(trio_list),
        genes=list(gene_list) if gene_list else None,
    )
    pairwise_marginals = belief_mod.get_pairwise_marginals(
        infer,
        list(individuals),
        list(gen_states),
        evidence,
        trio_pairs,
        genes=list(gene_list) if gene_list else None,
    )

    if gene_list:
        pure3: Dict[str, Dict[Trio, Dict[GenotypeTriple, float]]] = {}
        for gene in gene_list:
            gene_trios = trio_marginals.get(gene, {}) if isinstance(trio_marginals, Mapping) else {}
            gene_pairs = pairwise_marginals.get(gene, {}) if isinstance(pairwise_marginals, Mapping) else {}
            gene_probs = per_gene.get(gene, {})
            gene_pure3: Dict[Trio, Dict[GenotypeTriple, float]] = {}
            for trio in trio_list:
                parent1, parent2, child = trio
                trio_dist = gene_trios.get(trio, {}) if isinstance(gene_trios, Mapping) else {}
                fm_probs = gene_pairs.get((parent1, parent2), {}) if isinstance(gene_pairs, Mapping) else {}
                fc_probs = gene_pairs.get((parent1, child), {}) if isinstance(gene_pairs, Mapping) else {}
                mc_probs = gene_pairs.get((parent2, child), {}) if isinstance(gene_pairs, Mapping) else {}
                parent1_probs = gene_probs.get(parent1, marginals.get(parent1, {}))
                parent2_probs = gene_probs.get(parent2, marginals.get(parent2, {}))
                child_probs = gene_probs.get(child, marginals.get(child, {}))
                residuals: Dict[GenotypeTriple, float] = {}
                for g_parent1 in gen_states:
                    for g_parent2 in gen_states:
                        for g_child in gen_states:
                            triple = (g_parent1, g_parent2, g_child)
                            residuals[triple] = compute_trio_pure3_value(
                                joint_prob=float(trio_dist.get(triple, 0.0)),
                                fm_prob=float(fm_probs.get((g_parent1, g_parent2), 0.0)),
                                fc_prob=float(fc_probs.get((g_parent1, g_child), 0.0)),
                                mc_prob=float(mc_probs.get((g_parent2, g_child), 0.0)),
                                parent1_prob=float(parent1_probs.get(g_parent1, 0.0)),
                                parent2_prob=float(parent2_probs.get(g_parent2, 0.0)),
                                child_prob=float(child_probs.get(g_child, 0.0)),
                            )
                gene_pure3[trio] = residuals
            pure3[gene] = gene_pure3
    else:
        pure3 = {}
        for trio in trio_list:
            parent1, parent2, child = trio
            trio_dist = trio_marginals.get(trio, {}) if isinstance(trio_marginals, Mapping) else {}
            fm_probs = pairwise_marginals.get((parent1, parent2), {}) if isinstance(pairwise_marginals, Mapping) else {}
            fc_probs = pairwise_marginals.get((parent1, child), {}) if isinstance(pairwise_marginals, Mapping) else {}
            mc_probs = pairwise_marginals.get((parent2, child), {}) if isinstance(pairwise_marginals, Mapping) else {}
            parent1_probs = marginals.get(parent1, {})
            parent2_probs = marginals.get(parent2, {})
            child_probs = marginals.get(child, {})
            residuals: Dict[GenotypeTriple, float] = {}
            for g_parent1 in gen_states:
                for g_parent2 in gen_states:
                    for g_child in gen_states:
                        triple = (g_parent1, g_parent2, g_child)
                        residuals[triple] = compute_trio_pure3_value(
                            joint_prob=float(trio_dist.get(triple, 0.0)),
                            fm_prob=float(fm_probs.get((g_parent1, g_parent2), 0.0)),
                            fc_prob=float(fc_probs.get((g_parent1, g_child), 0.0)),
                            mc_prob=float(mc_probs.get((g_parent2, g_child), 0.0)),
                            parent1_prob=float(parent1_probs.get(g_parent1, 0.0)),
                            parent2_prob=float(parent2_probs.get(g_parent2, 0.0)),
                            child_prob=float(child_probs.get(g_child, 0.0)),
                        )
            pure3[trio] = residuals

    payload = {
        "evidence_state": state_key,
        "trios": trio_list,
        "pairs": tuple(trio_pairs),
        "genes": gene_list,
        "gen_states": tuple(gen_states),
        "sharing": trio_sharing,
        "share_key_map": resolved_share_key_map,
        "share_keys": share_keys,
        "pairwise_marginals": pairwise_marginals,
        "trio_marginals": trio_marginals,
        "pure3": pure3,
        "cache": {"hits": 0, "misses": 1},
        "telemetry": {
            "materialization_sec": float(time.perf_counter() - started_at),
        },
    }
    if isinstance(feature_cache, dict):
        feature_cache[cache_key] = payload
    return payload


def iter_trio_feature_rows(
    bundle: Mapping[str, object],
) -> Iterator[Tuple[Optional[str], Trio, object, Mapping[GenotypeTriple, float], Mapping[GenotypeTriple, float]]]:
    trios = tuple(bundle.get("trios", ()))
    share_key_map = bundle.get("share_key_map", {})
    genes = tuple(bundle.get("genes", ()))
    trio_marginals = bundle.get("trio_marginals", {})
    pure3 = bundle.get("pure3", {})

    if genes:
        for gene in genes:
            gene_trios = trio_marginals.get(gene, {}) if isinstance(trio_marginals, Mapping) else {}
            gene_pure3 = pure3.get(gene, {}) if isinstance(pure3, Mapping) else {}
            for trio in trios:
                yield (
                    gene,
                    trio,
                    share_key_map.get(trio, trio) if isinstance(share_key_map, Mapping) else trio,
                    gene_trios.get(trio, {}) if isinstance(gene_trios, Mapping) else {},
                    gene_pure3.get(trio, {}) if isinstance(gene_pure3, Mapping) else {},
                )
        return

    for trio in trios:
        yield (
            None,
            trio,
            share_key_map.get(trio, trio) if isinstance(share_key_map, Mapping) else trio,
            trio_marginals.get(trio, {}) if isinstance(trio_marginals, Mapping) else {},
            pure3.get(trio, {}) if isinstance(pure3, Mapping) else {},
        )
