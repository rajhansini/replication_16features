import importlib
import itertools
import os
import math
import numpy as np
import pickle
import threading
import time
import hashlib
import ctypes
from collections.abc import Mapping
from collections import deque
from multiprocessing import get_context, sharedctypes
from pathlib import Path

try:  # pgmpy is optional (some environments lack pandas/dateutil)
    from pgmpy.inference import VariableElimination  # type: ignore

    try:
        from pgmpy.models import DiscreteBayesianNetwork  # type: ignore
    except ImportError:  # pragma: no cover - pgmpy API shim
        from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork  # type: ignore

    _HAS_PGMPY = True
except Exception:  # pragma: no cover - optional dependency
    VariableElimination = None  # type: ignore[assignment]
    DiscreteBayesianNetwork = None  # type: ignore[assignment]
    _HAS_PGMPY = False
from ..config import get_config
from ..models.genetics_cpd import (
    make_founder_genotype_cpd,
    make_inheritance_genotype_cpd_with_table,
    make_multigene_founder_cpds,
    make_multigene_inheritance_cpds_with_tables,
    founder_prior_distribution,
    genotype_node_name,
)
from ..optimisation.caches import InferenceCache, CutManager
from ..optimisation.dual_dp import solve_dual_dp_with_domain
from ..optimisation.theta_model import resolve_theta_mode, theta_model_metadata
from ..policy.extractor import best_action
from ..policy.evaluator import exact_value_under_policy
from ..policy.myopic import evaluate_myopic_policy
from ..models.reward import r_reward, r_reward_test, r_reward_testp
from ..utils.pedigree_generator import generate_random_pedigree, generate_deterministic_pedigree
from ..models.belief import lift_single_gene_posteriors_to_genes, InferenceResult

# Exact DP imports
from ..exact_dp.solver import solve_exact_dual_pulp, solve_exact_dp_primal
from ..exact_dp.policy import extract_policy as extract_exact_policy
from ..exact_dp.utils import build_full_joint, lift_tuple_posteriors_to_genes, GENOTYPE_STATES
from ..inference.belief_map_inference import BeliefMapInference, SimpleBayesianNetwork
from ..utils.state_indexer import StateIndexer
from tqdm import tqdm

_CACHE_LOCK = threading.Lock()
_EXACT_DP_CACHE = None
_ACTIVE_CACHE_ROOT = None
_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2]
_PARALLEL_BUFFERS = {}


def _exact_cache_memory_only() -> bool:
    return os.getenv("EXACT_DP_CACHE_IN_MEMORY_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_cache_root() -> Path:
    raw = os.getenv("EXACT_DP_CACHE_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CACHE_ROOT


def _cache_path() -> Path:
    return _resolve_cache_root() / "cache" / "exact_dp_cache.pkl"


def _belief_cache_dir() -> Path:
    return _cache_path().parent / "belief_snapshots"


def _bind_cache_root() -> Path:
    global _ACTIVE_CACHE_ROOT, _EXACT_DP_CACHE
    root = _resolve_cache_root()
    if _ACTIVE_CACHE_ROOT != root:
        _ACTIVE_CACHE_ROOT = root
        _EXACT_DP_CACHE = None
    return root


def _load_exact_dp_cache():
    global _EXACT_DP_CACHE
    _bind_cache_root()
    if _exact_cache_memory_only():
        if _EXACT_DP_CACHE is None:
            _EXACT_DP_CACHE = {}
        return _EXACT_DP_CACHE
    cache_path = _cache_path()
    if _EXACT_DP_CACHE is None:
        if cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    _EXACT_DP_CACHE = pickle.load(fh)
            except Exception as exc:  # pragma: no cover - corrupted cache fallback
                print(f"Warning: could not load exact DP cache ({exc}); starting fresh.")
                try:
                    cache_path.unlink()
                except OSError:
                    pass
                _EXACT_DP_CACHE = {}
        else:
            _EXACT_DP_CACHE = {}
    return _EXACT_DP_CACHE


def _persist_exact_dp_cache():
    if _exact_cache_memory_only():
        return
    cache = _load_exact_dp_cache()
    cache_path = _cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)


def _belief_snapshot_path(cache_key):
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    belief_cache_dir = _belief_cache_dir()
    belief_cache_dir.mkdir(parents=True, exist_ok=True)
    return belief_cache_dir / f"{digest}.pkl"


def _belief_snapshot_key_from_exact_cache_key(cache_key):
    individuals, edges, params = cache_key
    genes = params[7]
    if genes:
        allele_payload = tuple(params[8] or ())
    else:
        allele_payload = float(params[0])
    return (
        "belief_snapshot_v2",
        bool(genes),
        individuals,
        edges,
        genes,
        allele_payload,
    )


def _load_belief_snapshot(cache_key):
    path = _belief_snapshot_path(cache_key)
    if path.exists():
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # pragma: no cover - corrupted snapshot fallback
            print(f"Warning: could not load belief snapshot ({exc}); recomputing.")
            try:
                path.unlink()
            except OSError:
                pass
            return None
    for exact_cache_key, entry in _load_exact_dp_cache().items():
        if _belief_snapshot_key_from_exact_cache_key(exact_cache_key) != cache_key:
            continue
        belief_exact = entry.get("belief_exact")
        if not belief_exact:
            continue
        try:
            _save_belief_snapshot(cache_key, belief_exact)
        except OSError:
            pass
        return belief_exact
    return None


def _save_belief_snapshot(cache_key, belief_exact):
    path = _belief_snapshot_path(cache_key)
    with path.open("wb") as fh:
        pickle.dump(belief_exact, fh)


def _make_exact_dp_cache_key(pedigree, config):
    individuals = tuple(pedigree.to_list())
    edges = tuple(sorted((u, v) for u, v in pedigree.graph.edges()))
    def _serialize_mapping(mapping):
        if not mapping:
            return None
        serialized = []
        for key, value in mapping.items():
            if isinstance(value, dict):
                serialized.append((key, _serialize_mapping(value)))
            else:
                serialized.append((key, value))
        return tuple(sorted(serialized))
    params = (
        config.allele_freq,
        config.fixed_cost,
        config.variable_cost,
        tuple(sorted(config.a.items())),
        tuple(sorted(config.b.items())),
        tuple(sorted(config.c.items())),
        tuple(sorted(config.delta.items())),
        tuple(config.genes) if getattr(config, "genes", None) else None,
        tuple(sorted(getattr(config, "allele_freqs", {}).items())) if getattr(config, "allele_freqs", None) else None,
        _serialize_mapping(getattr(config, "a_gene", None)),
        _serialize_mapping(getattr(config, "b_gene", None)),
        _serialize_mapping(getattr(config, "c_gene", None)),
        _serialize_mapping(getattr(config, "delta_gene", None)),
    )
    return (individuals, edges, params)


def _make_belief_snapshot_key(pedigree, config, *, tuple_mode: bool):
    individuals = tuple(pedigree.to_list())
    edges = tuple(sorted((u, v) for u, v in pedigree.graph.edges()))
    genes = tuple(config.genes) if getattr(config, "genes", None) else None
    if genes:
        allele_payload = tuple(sorted(getattr(config, "allele_freqs", {}).items()))
    else:
        allele_payload = float(getattr(config, "allele_freq", 0.0))
    return (
        "belief_snapshot_v2",
        bool(tuple_mode),
        individuals,
        edges,
        genes,
        allele_payload,
    )


def _prepare_mask_metadata(indexer: StateIndexer, num_genotypes: int):
    metadata = []
    for entry in indexer.mask_metadata():
        mask = entry["mask"]
        start, span = entry["start"], entry["span"]
        subset_indices = entry["subset_indices"]
        if span == 0:
            continue
        subset_indices_arr = np.asarray(subset_indices, dtype=np.intp)
        if subset_indices_arr.size:
            strides = (
                num_genotypes ** np.arange(subset_indices_arr.size - 1, -1, -1)
            ).astype(np.int64)
        else:
            strides = np.array([], dtype=np.int64)
        metadata.append(
            {
                "mask": mask,
                "subset_indices": subset_indices_arr,
                "strides": strides,
                "start": start,
                "span": span,
            }
        )
    return metadata


def _prepare_assignment_arrays(indexer: StateIndexer, joint_items, genotype_values):
    num_people = len(indexer.individuals)
    num_joint = len(joint_items)
    assignments = np.zeros((num_joint, num_people), dtype=np.uint16)
    probs = np.zeros(num_joint, dtype=np.float64)
    value_lookup = {
        StateIndexer._canonical_value(value): idx
        for idx, value in enumerate(genotype_values)
    }
    for row_idx, (genotype_tuple, prob) in enumerate(joint_items):
        probs[row_idx] = prob
        for person_idx, genotype_value in enumerate(genotype_tuple):
            canonical = StateIndexer._canonical_value(genotype_value)
            assignments[row_idx, person_idx] = value_lookup[canonical]
    return assignments, probs


def _state_id_from_metadata(meta, assignment_row, num_genotypes):
    subset_indices = meta["subset_indices"]
    state_id = meta["start"]
    if subset_indices.size == 0:
        return state_id
    offset = 0
    for idx in subset_indices:
        offset = offset * num_genotypes + int(assignment_row[idx])
    return state_id + offset

def _init_parallel_worker(
    state_mass_raw,
    state_counts_raw,
    assignments_raw,
    probs_raw,
    state_mass_shape,
    state_counts_shape,
    assignments_shape,
    num_people,
    num_genotypes,
):
    global _PARALLEL_BUFFERS
    _PARALLEL_BUFFERS = {
        "state_mass": np.ctypeslib.as_array(state_mass_raw).reshape(state_mass_shape),
        "state_counts": np.ctypeslib.as_array(state_counts_raw).reshape(state_counts_shape),
        "assignments": np.ctypeslib.as_array(assignments_raw).reshape(assignments_shape),
        "probs": np.ctypeslib.as_array(probs_raw).reshape((assignments_shape[0],)),
        "num_people": num_people,
        "num_genotypes": num_genotypes,
        "person_indices": np.arange(num_people, dtype=np.intp),
    }


def _accumulate_beliefs_sequential(
    mask_metadata,
    assignments,
    probs,
    state_mass,
    state_counts,
    num_people,
    num_genotypes,
    label=None,
):
    joint_len = assignments.shape[0]
    person_indices = np.arange(num_people, dtype=np.intp)
    joint_iter = tqdm(
        range(joint_len),
        desc=(label and f"{label} · joint tuples") or "Enumerating joint tuples",
        unit="tuple",
        leave=True,
    )
    for row_idx in joint_iter:
        prob = probs[row_idx]
        if prob <= 0.0:
            continue
        assignment = assignments[row_idx]
        for meta in mask_metadata:
            if meta["subset_indices"].size:
                offsets = assignment[meta["subset_indices"]].dot(meta["strides"])
                state_ids = meta["start"] + offsets
            else:
                state_ids = meta["start"]
            state_mass[state_ids] += prob
            np.add.at(
                state_counts,
                (state_ids, person_indices, assignment),
                prob,
            )


def _chunk_mask_metadata(mask_metadata, parallelism):
    groups = [[] for _ in range(parallelism)]
    for idx, meta in enumerate(mask_metadata):
        groups[idx % parallelism].append(meta)
    return [group for group in groups if group]


def _create_shared_array(shape, dtype):
    dtype_np = np.dtype(dtype)
    total_elems = int(np.prod(shape))
    if dtype_np == np.float64:
        raw = sharedctypes.RawArray(ctypes.c_double, total_elems)
    elif dtype_np == np.uint16:
        raw = sharedctypes.RawArray(ctypes.c_uint16, total_elems)
    else:
        raise ValueError(f"Unsupported shared dtype {dtype}")
    array = np.ctypeslib.as_array(raw).reshape(shape)
    array.fill(0)
    return raw, array


def _mask_group_worker(mask_group):
    shared = _PARALLEL_BUFFERS
    state_mass = shared["state_mass"]
    state_counts = shared["state_counts"]
    assignments = shared["assignments"]
    probs = shared["probs"]
    person_indices = shared["person_indices"]

    for meta in mask_group:
        if meta["subset_indices"].size:
            offsets = assignments[:, meta["subset_indices"]].dot(meta["strides"])
            state_ids = meta["start"] + offsets
        else:
            state_ids = np.full(assignments.shape[0], meta["start"], dtype=np.int64)
        np.add.at(state_mass, state_ids, probs)
        np.add.at(
            state_counts,
            (state_ids[:, None], person_indices[None, :], assignments),
            probs[:, None],
        )


def _accumulate_beliefs_parallel(
    mask_metadata,
    assignments,
    probs,
    num_people,
    num_genotypes,
    label=None,
    parallelism=2,
):
    total_states = int(sum(meta["span"] for meta in mask_metadata))
    state_mass_raw, state_mass = _create_shared_array((total_states,), np.float64)
    state_counts_raw, state_counts = _create_shared_array(
        (total_states, num_people, num_genotypes),
        np.float64,
    )
    assignments_raw, assignments_shared = _create_shared_array(assignments.shape, np.uint16)
    assignments_shared[:] = assignments
    probs_raw, probs_shared = _create_shared_array(probs.shape, np.float64)
    probs_shared[:] = probs
    assign_shape = assignments.shape

    mask_groups = _chunk_mask_metadata(mask_metadata, parallelism)
    try:
        ctx = get_context("fork")
    except ValueError:
        ctx = get_context()
    init_args = (
        state_mass_raw,
        state_counts_raw,
        assignments_raw,
        probs_raw,
        state_mass.shape,
        state_counts.shape,
        assign_shape,
        num_people,
        num_genotypes,
    )
    with ctx.Pool(
        processes=len(mask_groups),
        initializer=_init_parallel_worker,
        initargs=init_args,
    ) as pool:
        pool.map(_mask_group_worker, mask_groups)

    shared_handles = (state_mass_raw, state_counts_raw, assignments_raw, probs_raw)

    def _cleanup_shared():
        # Keep references alive until cleanup
        _ = shared_handles

    return state_mass, state_counts, _cleanup_shared


def _materialize_beliefs_from_arrays(
    indexer,
    individuals,
    genotype_values,
    state_mass,
    state_counts,
    min_state_mass,
    label=None,
):
    total_states = state_mass.shape[0]
    num_genotypes = len(genotype_values)
    belief = {}
    kept_states = 0
    skipped_states = 0
    conversion_desc = (label and f"{label} · inference materialization") or "Converting beliefs"
    convert_iter = tqdm(
        range(total_states),
        desc=conversion_desc,
        unit="state",
        leave=True,
    )
    for state_id in convert_iter:
        mass = state_mass[state_id]
        if mass <= min_state_mass:
            skipped_states += 1
            if state_id % 1000 == 0 or state_id + 1 == total_states:
                convert_iter.set_postfix(
                    kept=kept_states,
                    skipped=skipped_states,
                )
            continue
        state = indexer.materialize_by_id(state_id)
        p_s = {}
        counts_view = state_counts[state_id]
        for person_idx, person in enumerate(individuals):
            probs = counts_view[person_idx] / mass
            p_s[person] = {
                genotype_values[g_idx]: probs[g_idx]
                for g_idx in range(num_genotypes)
            }
        belief[state] = p_s
        kept_states += 1
        if state_id % 1000 == 0 or state_id + 1 == total_states:
            convert_iter.set_postfix(
                kept=kept_states,
                skipped=skipped_states,
            )
    convert_iter.close()
    return belief


def _build_belief_map_with_progress(
    pedigree,
    gen_states,
    joint,
    label=None,
    min_state_mass: float = 0.0,
    parallelism: int | None = None,
    return_arrays: bool = False,
    return_metadata: bool = False,
):
    started_at = time.perf_counter()
    individuals = pedigree.to_list()
    indexer = StateIndexer(individuals, gen_states)
    num_people = len(individuals)
    genotype_values = tuple(gen_states)
    num_genotypes = len(genotype_values)

    mask_metadata = _prepare_mask_metadata(indexer, num_genotypes)
    joint_items = list(joint.items())
    assignments, probs = _prepare_assignment_arrays(indexer, joint_items, genotype_values)
    if probs.size:
        positive_mask = probs > 0.0
        if not np.all(positive_mask):
            assignments = assignments[positive_mask]
            probs = probs[positive_mask]

    total_states = indexer.total_states
    use_parallel = (
        parallelism
        and parallelism > 1
        and total_states > 0
        and len(mask_metadata) > 1
        and assignments.size > 0
    )

    cleanup_fn = None
    accumulation_started_at = time.perf_counter()
    if use_parallel:
        try:
            state_mass, state_counts, cleanup_fn = _accumulate_beliefs_parallel(
                mask_metadata,
                assignments,
                probs,
                num_people,
                num_genotypes,
                label=label,
                parallelism=parallelism,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Parallel belief accumulation failed ({exc}); falling back to sequential.")
            use_parallel = False

    if not use_parallel:
        state_mass = np.zeros(total_states, dtype=np.float64)
        state_counts = np.zeros((total_states, num_people, num_genotypes), dtype=np.float64)
        _accumulate_beliefs_sequential(
            mask_metadata,
            assignments,
            probs,
            state_mass,
            state_counts,
            num_people,
            num_genotypes,
            label=label,
        )
    accumulation_elapsed = time.perf_counter() - accumulation_started_at

    materialization_started_at = time.perf_counter()
    belief = _materialize_beliefs_from_arrays(
        indexer,
        individuals,
        genotype_values,
        state_mass,
        state_counts,
        min_state_mass,
        label=label,
    )
    materialization_elapsed = time.perf_counter() - materialization_started_at

    if cleanup_fn is not None:
        cleanup_fn()

    metadata = {
        "progress_status": "completed",
        "mode": "dense",
        "label": label,
        "state_count": int(len(belief)),
        "total_indexed_state_count": int(total_states),
        "joint_assignment_count": int(len(joint_items)),
        "positive_assignment_count": int(len(probs)),
        "parallelism": int(parallelism or 1),
        "used_parallel": bool(use_parallel),
        "accumulation_elapsed_sec": float(accumulation_elapsed),
        "materialization_elapsed_sec": float(materialization_elapsed),
        "elapsed_sec": float(time.perf_counter() - started_at),
    }

    if return_arrays:
        if return_metadata:
            return belief, state_mass, state_counts, metadata
        return belief, state_mass, state_counts
    if return_metadata:
        return belief, metadata
    return belief


def _belief_build_mode(*, pedigree, config, tuple_mode: bool) -> str:
    raw = os.getenv("EXACT_BELIEF_BUILD_MODE", "auto").strip().lower()
    if raw in {"dense", "factorized", "snapshot_only"}:
        return raw
    if raw != "auto":
        raise ValueError(
            "Unknown EXACT_BELIEF_BUILD_MODE="
            f"{raw!r} (expected 'auto', 'dense', 'factorized', or 'snapshot_only')."
        )
    if tuple_mode and len(tuple(pedigree.to_list())) >= 7:
        return "factorized"
    return "dense"


def _resolve_exact_dp_solver_mode(*, exact_dp_solver: str, multi_gene: bool, n_individuals: int) -> str:
    solver = exact_dp_solver.strip().lower()
    if solver not in {"dual", "primal", "auto"}:
        return "dual"
    if solver != "auto":
        return solver
    # Restore the archived benchmark lineage used by the stage oracle:
    # 5-person ThreeGeneration rows used the dual exact path, while the
    # 6-person Extended rows used the primal exact path.
    if multi_gene and n_individuals >= 6:
        return "primal"
    return "dual"


def _state_sort_key(state, individuals, genotype_values):
    person_to_idx = {person: idx for idx, person in enumerate(individuals)}
    genotype_to_idx = {
        StateIndexer._canonical_value(value): idx
        for idx, value in enumerate(genotype_values)
    }
    mask = 0
    value_indices = []
    for person, genotype in sorted(state, key=lambda item: person_to_idx[item[0]]):
        idx = person_to_idx[person]
        mask |= 1 << idx
        value_indices.append(genotype_to_idx[StateIndexer._canonical_value(genotype)])
    return mask, tuple(value_indices)


def _as_inference_result(entry, *, gen_states=GENOTYPE_STATES):
    if isinstance(entry, InferenceResult):
        return entry
    if isinstance(entry, tuple) and entry:
        return _as_inference_result(entry[0], gen_states=gen_states)
    if isinstance(entry, Mapping):
        return InferenceResult(
            marginals={person: dict(probs) for person, probs in entry.items()},
            gene_order=("gene",),
            gen_states=gen_states,
        )
    raise TypeError(f"Unsupported belief entry type: {type(entry)!r}")


def _single_gene_belief_config(pedigree, allele_freq: float):
    config = get_config(
        pedigree.to_list(),
        pedigree=pedigree,
        allele_freq=allele_freq,
    )
    config.fixed_cost = 0.0
    config.variable_cost = 0.0
    return config


def _load_or_build_single_gene_belief_snapshot(
    *,
    pedigree,
    allele_freq: float,
    child_cpds,
    belief_parallelism: int | None = None,
    progress_label=None,
    return_metadata: bool = False,
):
    started_at = time.perf_counter()
    config = _single_gene_belief_config(pedigree, allele_freq)
    snapshot_key = _make_belief_snapshot_key(pedigree, config, tuple_mode=False)
    cache_path_exists = _belief_snapshot_path(snapshot_key).exists()
    cached = _load_belief_snapshot(snapshot_key)
    if cached is not None:
        belief = {
            state: _as_inference_result(entry)
            for state, entry in cached.items()
        }
        if return_metadata:
            return belief, {
                "progress_status": "completed",
                "mode": "scalar",
                "cache_status": "snapshot_cache_hit" if cache_path_exists else "exact_cache_backfill_hit",
                "allele_freq": float(allele_freq),
                "state_count": int(len(belief)),
                "elapsed_sec": float(time.perf_counter() - started_at),
            }
        return belief

    joint_dist = build_full_joint(
        pedigree,
        GENOTYPE_STATES,
        allele_freq,
        child_cpds,
    )
    raw_belief, scalar_metadata = _build_belief_map_with_progress(
        pedigree,
        GENOTYPE_STATES,
        joint_dist,
        label=progress_label,
        parallelism=belief_parallelism,
        return_metadata=True,
    )
    belief_exact = {
        state: InferenceResult(
            marginals={person: dict(probs) for person, probs in marginal.items()},
            gene_order=("gene",),
            gen_states=GENOTYPE_STATES,
        )
        for state, marginal in raw_belief.items()
    }
    _save_belief_snapshot(snapshot_key, belief_exact)
    if return_metadata:
        return belief_exact, {
            "progress_status": "completed",
            "mode": "scalar",
            "cache_status": "built",
            "allele_freq": float(allele_freq),
            "state_count": int(len(belief_exact)),
            "joint_assignment_count": int(scalar_metadata.get("joint_assignment_count", 0)),
            "total_indexed_state_count": int(scalar_metadata.get("total_indexed_state_count", 0)),
            "accumulation_elapsed_sec": float(scalar_metadata.get("accumulation_elapsed_sec", 0.0)),
            "materialization_elapsed_sec": float(scalar_metadata.get("materialization_elapsed_sec", 0.0)),
            "elapsed_sec": float(time.perf_counter() - started_at),
        }
    return belief_exact


def _build_factorized_multigene_belief_snapshot(
    *,
    pedigree,
    config,
    genes,
    child_cpds,
    belief_parallelism: int | None = None,
    progress_label=None,
    return_metadata: bool = False,
):
    started_at = time.perf_counter()
    individuals = tuple(pedigree.to_list())
    exact_gen_states = [
        tuple(outcome)
        for outcome in itertools.product(GENOTYPE_STATES, repeat=len(genes))
    ]
    single_gene_beliefs = {}
    scalar_state_counts = {}
    scalar_metadata = {}
    for gene in genes:
        single_gene_beliefs[gene], scalar_metadata[gene] = _load_or_build_single_gene_belief_snapshot(
            pedigree=pedigree,
            allele_freq=config.allele_freqs.get(gene, config.allele_freq),
            child_cpds=child_cpds,
            belief_parallelism=belief_parallelism,
            progress_label=(
                f"{progress_label} · {gene} scalar"
                if progress_label
                else f"{gene} scalar"
            ),
            return_metadata=True,
        )
        scalar_state_counts[gene] = int(len(single_gene_beliefs[gene]))

    belief_exact = {}
    frontier = deque([frozenset()])
    seen = {frozenset()}
    primary_gene = genes[0]
    processed_state_count = 0
    generated_successor_count = 0
    max_frontier_size = len(frontier)

    while frontier:
        state = frontier.popleft()
        processed_state_count += 1
        per_gene_probs = {}
        for gene_idx, gene in enumerate(genes):
            projected_state = frozenset(
                (person, outcome[gene_idx])
                for person, outcome in state
            )
            per_gene_probs[gene] = _as_inference_result(
                single_gene_beliefs[gene][projected_state]
            ).marginals

        tuple_pmfs = {}
        gene_first = {
            gene: {
                person: dict(per_gene_probs[gene][person])
                for person in individuals
            }
            for gene in genes
        }
        for person in individuals:
            pmf = {outcome: 0.0 for outcome in exact_gen_states}
            total = 0.0
            for outcome in exact_gen_states:
                prob = 1.0
                for idx, gene in enumerate(genes):
                    prob *= float(per_gene_probs[gene][person].get(outcome[idx], 0.0))
                    if prob <= 0.0:
                        break
                pmf[outcome] = prob
                if prob > 0.0:
                    total += prob
            if total > 0.0 and abs(total - 1.0) > 1e-12:
                pmf = {outcome: prob / total for outcome, prob in pmf.items()}
            tuple_pmfs[person] = pmf

        belief_exact[state] = InferenceResult(
            marginals={
                person: dict(gene_first[primary_gene][person])
                for person in individuals
            },
            tuple_pmfs=tuple_pmfs,
            per_gene=gene_first,
            gene_order=genes,
            gen_states=GENOTYPE_STATES,
        )

        observed_people = {person for person, _ in state}
        if len(observed_people) >= len(individuals):
            continue
        for person in individuals:
            if person in observed_people:
                continue
            for outcome in exact_gen_states:
                if tuple_pmfs[person].get(outcome, 0.0) <= 0.0:
                    continue
                successor_items = dict(state)
                successor_items[person] = outcome
                successor_state = frozenset(successor_items.items())
                if successor_state in seen:
                    continue
                seen.add(successor_state)
                generated_successor_count += 1
                frontier.append(successor_state)
                max_frontier_size = max(max_frontier_size, len(frontier))

    ordered_states = sorted(
        belief_exact,
        key=lambda state: _state_sort_key(state, individuals, exact_gen_states),
    )
    ordered_belief = {
        state: belief_exact[state]
        for state in ordered_states
    }
    metadata = {
        "progress_status": "completed",
        "mode": "factorized",
        "label": progress_label,
        "state_count": int(len(ordered_belief)),
        "processed_state_count": int(processed_state_count),
        "generated_successor_count": int(generated_successor_count),
        "max_frontier_size": int(max_frontier_size),
        "gene_count": int(len(genes)),
        "tuple_outcome_count": int(len(exact_gen_states)),
        "scalar_state_counts": scalar_state_counts,
        "scalar_metadata": scalar_metadata,
        "scalar_cache_status": {
            gene: str(meta.get("cache_status", "unknown"))
            for gene, meta in scalar_metadata.items()
        },
        "scalar_elapsed_sec": {
            gene: float(meta.get("elapsed_sec", 0.0))
            for gene, meta in scalar_metadata.items()
        },
        "parallelism": int(belief_parallelism or 1),
        "elapsed_sec": float(time.perf_counter() - started_at),
    }
    if return_metadata:
        return ordered_belief, metadata
    return ordered_belief


def _decide_polish_acceptance(
    *,
    pre_metrics: dict | None,
    post_metrics: dict | None,
    ratio2_delta_tol: float,
    gap2_delta_tol: float,
    denom_small_eps: float,
):
    polish_acceptance_reason = "accepted"
    polish_acceptance_decision = "accepted"
    selected_candidate_id = "post_polish"

    if pre_metrics is not None and post_metrics is not None:
        denom = float(post_metrics.get("denom", 0.0))
        ratio2_pre = pre_metrics.get("ratio2")
        ratio2_post = post_metrics.get("ratio2")
        gap2_pre = float(pre_metrics.get("gap2", 0.0))
        gap2_post = float(post_metrics.get("gap2", 0.0))
        if ratio2_post is not None and ratio2_post < -1e-9:
            polish_acceptance_reason = "ratio2_negative"
        elif ratio2_post is not None and ratio2_post > 1.0 + 1e-6:
            polish_acceptance_reason = "ratio2_exploded"
        elif denom > denom_small_eps:
            if ratio2_pre is None or ratio2_post is None:
                polish_acceptance_reason = "ratio2_regression"
            elif (ratio2_post - ratio2_pre) > ratio2_delta_tol:
                polish_acceptance_reason = "ratio2_regression"
        else:
            if (gap2_post - gap2_pre) > gap2_delta_tol:
                polish_acceptance_reason = "denom_small_gap2_regression"

    if polish_acceptance_reason != "accepted":
        polish_acceptance_decision = "rejected_post_polish"
        selected_candidate_id = "pre_polish"

    return polish_acceptance_reason, polish_acceptance_decision, selected_candidate_id


def _safe_ratio_or_none(numerator, denominator, *, eps: float = 1e-9):
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numerator_f) or not math.isfinite(denominator_f):
        return None
    if abs(denominator_f) <= eps:
        return None
    return numerator_f / denominator_f


def _resolve_safe_rollout_module():
    try:
        return importlib.import_module("genetic_dp.policy.safe_rollout")
    except Exception:
        return None


def _resolve_safe_rollout_callable(module):
    if module is None:
        return None, None
    for candidate_name in (
        "evaluate_safe_rollout",
        "run_safe_rollout",
        "compute_safe_rollout",
        "build_safe_rollout",
        "safe_rollout",
    ):
        candidate = getattr(module, candidate_name, None)
        if callable(candidate):
            return candidate, candidate_name
    if callable(module):
        return module, getattr(module, "__name__", "callable")
    return None, None


def run_and_compare_solvers(
    pedigree,
    config,
    verbose=False,
    lookahead_depths=(0,),
    print_policies: bool = True,
    progress_label=None,
    belief_parallelism: int | None = None,
    disable_snapshots: bool = False,
    dfvr_bound: bool = False,
    dfvr_state_mode: str = "belief",
    dfvr_max_states: int | None = None,
    dfvr_random_seed: int = 0,
    dfvr_top_k: int = 0,
    dfvr_max_outcomes_per_action: int = 10,
    dfvr_eval_no_mutation: bool | None = None,
    dfvr_fixed_states_path: str | None = None,
    dfvr_enforce_fixed_stateset: bool | None = None,
    dfvr_emit_fixed_stateset: bool | None = None,
    return_infer: bool = False,
    adp_analysis_mode: str | None = None,
    adp_return_stats: bool = False,
    phase_callback=None,
):
    # ------------------------------------------------------------------
    # 1. Bayesian Network Setup (with timing)
    # ------------------------------------------------------------------
    print("\n--- Building Bayesian Network ---")
    bayesian_setup_start = time.time()

    def _notify_phase(phase: str) -> None:
        if phase_callback is None:
            return
        phase_callback(phase)

    exact_progress_started_at = time.perf_counter()
    exact_progress_events = []

    def _record_exact_progress(phase, status: str = "event", **payload) -> None:
        if isinstance(phase, Mapping):
            event = dict(phase)
        else:
            event = {"phase": str(phase), "status": str(status)}
            event.update(payload)
        event.setdefault("status", status)
        event.setdefault("elapsed_sec", float(time.perf_counter() - exact_progress_started_at))
        event.setdefault("wall_time_unix", float(time.time()))
        run_id = os.getenv("BENCHMARK_RUN_ID", "").strip()
        tier = os.getenv("BENCHMARK_TIER", "").strip()
        case = os.getenv("BENCHMARK_CASE", "").strip()
        if run_id:
            event.setdefault("benchmark_run_id", run_id)
        if tier:
            event.setdefault("benchmark_tier", tier)
        if case:
            event.setdefault("benchmark_case", case)
        exact_progress_events.append(event)

    depth_list = sorted(set(lookahead_depths)) or [0]
    
    individuals = pedigree.to_list()
    gen_states    = [0,1,2]
    multi_gene = bool(config.genes)
    genes = config.genes if multi_gene else ("gene",)
    gene_list = tuple(config.genes) if multi_gene else tuple()
    exact_dp_solver = os.getenv("EXACT_DP_SOLVER", "auto").strip().lower()
    requested_exact_solver = _resolve_exact_dp_solver_mode(
        exact_dp_solver=exact_dp_solver,
        multi_gene=multi_gene,
        n_individuals=len(individuals),
    )
    use_primal_exact = requested_exact_solver == "primal"

    # Dynamically create the Bayesian Network and initial beliefs
    bn_edges = []
    all_cpds = []
    initial_p = {}
    initial_p_gene = {gene: {} for gene in genes} if multi_gene else None
    child_cpds = {}

    # Use topological sort to ensure parents are processed before children
    for individual in pedigree.to_list():
        parents = pedigree.get_parents(individual)
        if not parents:
            # Founder
            if multi_gene:
                founder_cpds = make_multigene_founder_cpds(
                    individual,
                    genes,
                    config.allele_freqs,
                )
                for gene, cpd in founder_cpds.items():
                    all_cpds.append(cpd)
                    prior = founder_prior_distribution(config.get_allele_freq(gene))
                    initial_p_gene[gene][individual] = {g: prior[g] for g in gen_states}
                # compatibility: use first gene for aggregate view
                primary_gene = genes[0]
                initial_p[individual] = dict(initial_p_gene[primary_gene][individual])
            else:
                cpd = make_founder_genotype_cpd(individual, allele_freq=config.allele_freq)
                all_cpds.append(cpd)
                prior = founder_prior_distribution(config.allele_freq)
                initial_p[individual] = {g: prior[g] for g in gen_states}
        else:
            # Child
            parent1, parent2 = parents
            if multi_gene:
                cpds_by_gene = make_multigene_inheritance_cpds_with_tables(
                    individual,
                    parent1,
                    parent2,
                    genes,
                )
                for gene, (cpd, child_table) in cpds_by_gene.items():
                    all_cpds.append(cpd)
                    father_node = genotype_node_name(parent1, gene)
                    mother_node = genotype_node_name(parent2, gene)
                    child_node = genotype_node_name(individual, gene)
                    bn_edges.append((father_node, child_node))
                    bn_edges.append((mother_node, child_node))
                    parent1_prior = initial_p_gene[gene][parent1]
                    parent2_prior = initial_p_gene[gene][parent2]
                    initial_p_gene[gene][individual] = {
                        g_child: sum(
                            child_table[g_child, u * 3 + v] *
                            parent1_prior[u] *
                            parent2_prior[v]
                            for u in gen_states for v in gen_states
                        )
                        for g_child in gen_states
                    }
                primary_gene = genes[0]
                child_cpds[individual] = cpds_by_gene[primary_gene][1]
                initial_p[individual] = dict(initial_p_gene[primary_gene][individual])
            else:
                cpd, child_table = make_inheritance_genotype_cpd_with_table(individual, parent1, parent2)
                all_cpds.append(cpd)
                bn_edges.append((parent1, individual))
                bn_edges.append((parent2, individual))
                child_cpds[individual] = child_table

                # Calculate initial probability for the child
                initial_p[individual] = {
                    g_child: sum(
                        child_table[g_child, u * 3 + v] *
                        initial_p[parent1][u] *
                        initial_p[parent2][v]
                        for u in gen_states for v in gen_states
                    )
                    for g_child in gen_states
                }

    # flags x[i]   (0 = untested, 1 = tested)
    x = {i:0 for i in individuals}

    # initial z all zeros (untested)
    initial_z = {i:{g:0 for g in gen_states} for i in individuals}

    # Φ(s) initial distribution weight (only root state)
    mu0 = {frozenset(): 1.0}

    bn_model = None
    infer = None
    if _HAS_PGMPY:
        bn = DiscreteBayesianNetwork(bn_edges)
        bn.add_cpds(*all_cpds)
        assert bn.check_model()
        infer = VariableElimination(bn)
        bn_model = bn
    else:
        bn_model = SimpleBayesianNetwork({cpd.variable: cpd for cpd in all_cpds})
    
    bayesian_setup_time = time.time() - bayesian_setup_start
    print(f"Bayesian Network setup completed in {bayesian_setup_time:.3f}s")
    
    # ------------------------------------------------------------------
    # 2. Solve the Exact DP FIRST (to populate caches)
    # ------------------------------------------------------------------
    print("\n--- Running Exact DP Solver FIRST ---")
    _notify_phase("exact_dp")
    start_exact_dp = time.time()
    exact_dp_cache_status = "cache_miss"
    belief_map_tracking = {
        "progress_status": "not_started",
        "mode": None,
        "state_count": None,
        "elapsed_sec": 0.0,
    }
    exact_dual_tracking = {
        "progress_status": "not_started",
        "backend": None,
        "lp_solve_elapsed_sec": None,
        "total_elapsed_sec": None,
    }
    _record_exact_progress(
        "exact_dp",
        "start",
        exact_solver=requested_exact_solver,
        multi_gene=multi_gene,
        gene_count=len(gene_list),
        individual_count=len(individuals),
    )
    
    # For exact DP, we need the child_table for build_full_joint
    # We'll use the child_table from the first child generated for simplicity
    # In a more complex pedigree, this would need to be handled per child
    first_child_table = None
    for individual in pedigree.get_offspring():
        parents = pedigree.get_parents(individual)
        if parents:
            _, first_child_table = make_inheritance_genotype_cpd_with_table(individual, parents[0], parents[1])
            break
    
    cache_key = _make_exact_dp_cache_key(pedigree, config)
    belief_snapshot_key = _make_belief_snapshot_key(
        pedigree,
        config,
        tuple_mode=multi_gene,
    )
    requested_belief_build_mode_raw = os.getenv("EXACT_BELIEF_BUILD_MODE", "auto").strip().lower()
    requested_belief_build_mode = (
        _belief_build_mode(pedigree=pedigree, config=config, tuple_mode=True)
        if multi_gene
        else "dense"
    )
    cache_entry = None
    with _CACHE_LOCK:
        cache_entry = _load_exact_dp_cache().get(cache_key)

    belief_exact = None
    precomputed_results = None
    cache_hit = cache_entry is not None and not disable_snapshots

    if cache_hit and multi_gene:
        cached_beliefs = cache_entry.get("belief_exact")
        stale_cache = False
        if not cached_beliefs:
            stale_cache = True
        else:
            for entry in cached_beliefs.values():
                if not isinstance(entry, InferenceResult):
                    stale_cache = True
                    break
                if not hasattr(entry, "has_tuple_pmfs") or not entry.has_tuple_pmfs():
                    stale_cache = True
                    break
                try:
                    per_gene_check = entry.get_per_gene_probs()
                except AttributeError:
                    stale_cache = True
                    break
                if not per_gene_check:
                    stale_cache = True
                    break
        if stale_cache:
            cache_hit = False

    if cache_hit and multi_gene and requested_belief_build_mode in {"dense", "factorized"}:
        cached_tracking = cache_entry.get("exact_dp_runtime_tracking")
        cached_belief_mode = None
        if isinstance(cached_tracking, Mapping):
            cached_belief_tracking = cached_tracking.get("belief_map")
            if isinstance(cached_belief_tracking, Mapping):
                mode = cached_belief_tracking.get("mode")
                cached_belief_mode = str(mode) if mode is not None else None
        if cached_belief_mode != requested_belief_build_mode:
            cache_hit = False

    if cache_hit:
        cached_exact_solver = cache_entry.get("exact_solver")
        if cached_exact_solver is None:
            # Legacy entries have no solver tag. Infer stale primal caches created
            # before solver tagging from missing per-gene Phi components.
            if multi_gene and cache_entry.get("Phi_star_exact_gene") is None:
                cached_exact_solver = "primal"
            else:
                cached_exact_solver = "dual"
        if cached_exact_solver != requested_exact_solver:
            cache_hit = False

    if cache_hit:
        print("Exact DP cache hit – reusing precomputed beliefs and policy.")
        exact_dp_cache_status = "exact_dp_cache_hit"
        if multi_gene:
            exact_gen_states = [
                tuple(outcome)
                for outcome in itertools.product(GENOTYPE_STATES, repeat=len(gene_list))
            ]
        else:
            exact_gen_states = gen_states
        Phi_star_exact = cache_entry["Phi_star_exact"]
        Phi_star_exact_gene = cache_entry.get("Phi_star_exact_gene")
        policy_exact = cache_entry["policy_exact"]
        belief_exact = cache_entry.get("belief_exact")
        precomputed_results = cache_entry.get("precomputed_results")
        exact_dp_belief_time = cache_entry.get("exact_dp_belief_time", 0.0)
        exact_dp_time = cache_entry.get("exact_dp_total_time", exact_dp_belief_time)
        cached_tracking = cache_entry.get("exact_dp_runtime_tracking")
        if isinstance(cached_tracking, Mapping):
            cached_belief_tracking = cached_tracking.get("belief_map")
            cached_dual_tracking = cached_tracking.get("exact_dual")
            if isinstance(cached_belief_tracking, Mapping):
                belief_map_tracking = dict(cached_belief_tracking)
            if isinstance(cached_dual_tracking, Mapping):
                exact_dual_tracking = dict(cached_dual_tracking)
        belief_map_tracking.update(
            {
                "progress_status": "cache_hit",
                "cache_status": exact_dp_cache_status,
                "state_count": int(len(belief_exact)) if belief_exact is not None else belief_map_tracking.get("state_count"),
            }
        )
        exact_dual_tracking.update(
            {
                "progress_status": "cache_hit" if requested_exact_solver == "dual" else "not_run_primal_exact",
                "cache_status": exact_dp_cache_status,
            }
        )
        _record_exact_progress(
            "exact_dp_cache",
            "hit",
            belief_state_count=int(len(belief_exact)) if belief_exact is not None else None,
            exact_solver=requested_exact_solver,
        )
        if use_primal_exact and belief_exact:
            start_exact_solve = time.time()
            Phi_star_exact_gene = None
            Phi_star_exact, policy_exact = solve_exact_dp_primal(
                individuals,
                exact_gen_states,
                mu0,
                belief_exact,
                config.a,
                config.b,
                config.c,
                config.delta,
                config.fixed_cost,
                config.variable_cost,
                genes=gene_list if multi_gene else None,
                a_gene=config.a_gene if multi_gene else None,
                b_gene=config.b_gene if multi_gene else None,
                c_gene=config.c_gene if multi_gene else None,
                delta_gene=config.delta_gene if multi_gene else None,
                base_gen_states=GENOTYPE_STATES,
            )
            exact_dp_solve_time = time.time() - start_exact_solve
            exact_dp_time = exact_dp_belief_time + exact_dp_solve_time
            exact_dual_tracking = {
                "progress_status": "not_run_primal_exact",
                "cache_status": "exact_dp_cache_hit_primal_recomputed",
                "backend": None,
                "lp_build_elapsed_sec": None,
                "lp_solve_elapsed_sec": None,
                "total_elapsed_sec": float(exact_dp_solve_time),
                "status": "Optimal",
            }
            cache_payload = {
                "Phi_star_exact": Phi_star_exact,
                "Phi_star_exact_gene": Phi_star_exact_gene,
                "policy_exact": policy_exact,
                "belief_exact": belief_exact,
                "precomputed_results": precomputed_results,
                "exact_dp_belief_time": exact_dp_belief_time,
                "exact_dp_total_time": exact_dp_time,
                "exact_dp_runtime_tracking": {
                    "schema": "exact_dp_runtime_tracking.v1",
                    "cache_status": "exact_dp_cache_hit_primal_recomputed",
                    "exact_solver": requested_exact_solver,
                    "belief_map": dict(belief_map_tracking),
                    "exact_dual": dict(exact_dual_tracking),
                    "progress_events": list(exact_progress_events),
                },
                "exact_solver": requested_exact_solver,
                "timestamp": time.time(),
            }
            with _CACHE_LOCK:
                cache = _load_exact_dp_cache()
                cache[cache_key] = cache_payload
                _persist_exact_dp_cache()
    else:
        if first_child_table is None and len(individuals) > 2:  # If no children, but more than just founders
            print("Warning: No children found in pedigree for exact DP. Skipping exact DP.")
            Phi_star_exact = {}
            policy_exact = {}
            exact_dp_belief_time = 0.0
            precomputed_results = {}
            belief_map_tracking = {
                "progress_status": "skipped_no_child_table",
                "mode": "skipped",
                "state_count": 0,
                "elapsed_sec": 0.0,
            }
            exact_dual_tracking = {
                "progress_status": "not_run_no_child_table",
                "backend": None,
                "lp_build_elapsed_sec": None,
                "lp_solve_elapsed_sec": None,
                "total_elapsed_sec": 0.0,
                "status": None,
            }
        elif first_child_table is None and len(individuals) <= 2:  # Only founders
            print("No children in pedigree. Exact DP will only consider stopping.")
            if multi_gene:
                exact_gen_states = list(itertools.product(GENOTYPE_STATES, repeat=len(gene_list)))
                root_tuple_pmfs = {}
                for person in individuals:
                    dist = {}
                    for outcome in exact_gen_states:
                        prob = 1.0
                        for idx, gene in enumerate(gene_list):
                            prob *= initial_p_gene[gene][person][outcome[idx]]
                        dist[outcome] = prob
                    root_tuple_pmfs[person] = dist
                per_gene_probs_root = lift_tuple_posteriors_to_genes(root_tuple_pmfs, gene_list)
                root_value = sum(
                    r_reward(
                        person,
                        root_tuple_pmfs,
                        config.a,
                        config.b,
                        config.c,
                        config.delta,
                        per_gene_probs=per_gene_probs_root,
                        a_gene=config.a_gene,
                        b_gene=config.b_gene,
                        c_gene=config.c_gene,
                        delta_gene=config.delta_gene,
                    )
                    for person in individuals
                )
                primary_gene = gene_list[0]
                aggregated_root = {}
                for person in individuals:
                    primary_map = per_gene_probs_root.get(primary_gene, {}).get(person)
                    if primary_map is None:
                        aggregated_root[person] = {g: 0.0 for g in GENOTYPE_STATES}
                    else:
                        aggregated_root[person] = dict(primary_map)
                state_key = frozenset()
                inference_result = InferenceResult(
                    aggregated_root,
                    root_tuple_pmfs,
                    per_gene=per_gene_probs_root,
                    gene_order=gene_list,
                    gen_states=GENOTYPE_STATES,
                )
                Phi_star_exact = {state_key: root_value}
                belief_exact = {state_key: inference_result}
                precomputed_results = {state_key: inference_result}
            else:
                Phi_star_exact = {
                    frozenset(): sum(
                        config.a[k]
                        * (initial_p[k][1] + initial_p[k][2] - config.delta[k] * (initial_p[k][1] + initial_p[k][2]) ** 2)
                        + config.b[k] * ((initial_p[k][1] + initial_p[k][2]) - (initial_p[k][1] + initial_p[k][2]) ** 2)
                        + config.c[k]
                        for k in individuals
                    )
                }
                aggregated = {person: dict(initial_p[person]) for person in individuals}
                inference_result = InferenceResult(
                    aggregated,
                    gene_order=("gene",),
                    gen_states=GENOTYPE_STATES,
                )
                belief_exact = {frozenset(): inference_result}
                precomputed_results = {frozenset(): inference_result}
            policy_exact = {frozenset(): ("stop", None, Phi_star_exact[frozenset()])}
            exact_dp_belief_time = 0.0
            belief_map_tracking = {
                "progress_status": "completed_root_only",
                "mode": "root_only",
                "state_count": 1,
                "elapsed_sec": 0.0,
            }
            exact_dual_tracking = {
                "progress_status": "not_run_root_only",
                "backend": None,
                "lp_build_elapsed_sec": None,
                "lp_solve_elapsed_sec": None,
                "total_elapsed_sec": 0.0,
                "status": "Optimal",
            }
        else:
            belief_snapshot = None
            if (
                not disable_snapshots
                and requested_belief_build_mode_raw in {"auto", "snapshot_only"}
            ):
                belief_snapshot = _load_belief_snapshot(belief_snapshot_key)
            if belief_snapshot is not None:
                print("Loaded belief snapshot from disk; skipping inference build.")
                belief_exact = belief_snapshot
                precomputed_results = belief_snapshot
                exact_dp_belief_time = 0.0
                belief_map_tracking = {
                    "progress_status": "snapshot_cache_hit",
                    "mode": "snapshot",
                    "state_count": int(len(belief_exact)),
                    "elapsed_sec": 0.0,
                    "cache_status": "belief_snapshot_hit",
                }
                _record_exact_progress(
                    "belief_map_construction",
                    "snapshot_cache_hit",
                    state_count=int(len(belief_exact)),
                    mode="snapshot",
                )
                if multi_gene:
                    exact_gen_states = [
                        tuple(outcome)
                        for outcome in itertools.product(GENOTYPE_STATES, repeat=len(gene_list))
                    ]
                else:
                    exact_gen_states = gen_states
            else:
                start_belief_build = time.time()
                if multi_gene:
                    exact_gen_states = [
                        tuple(outcome)
                        for outcome in itertools.product(GENOTYPE_STATES, repeat=len(gene_list))
                    ]
                    belief_build_mode = requested_belief_build_mode
                    if belief_build_mode == "snapshot_only":
                        raise FileNotFoundError(
                            "EXACT_BELIEF_BUILD_MODE='snapshot_only' but no belief snapshot exists for "
                            f"pedigree={tuple(individuals)} genes={gene_list} allele_freqs={config.allele_freqs}."
                        )
                    _record_exact_progress(
                        "belief_map_construction",
                        "start",
                        mode=belief_build_mode,
                        gene_count=len(gene_list),
                        tuple_mode=True,
                    )
                    if belief_build_mode == "factorized":
                        belief_exact, belief_metadata = _build_factorized_multigene_belief_snapshot(
                            pedigree=pedigree,
                            config=config,
                            genes=gene_list,
                            child_cpds=child_cpds,
                            belief_parallelism=belief_parallelism,
                            progress_label=progress_label,
                            return_metadata=True,
                        )
                        belief_map_tracking = dict(belief_metadata)
                        precomputed_results = belief_exact
                    else:
                        tuple_states = list(exact_gen_states)
                        joint_dist = build_full_joint(
                            pedigree,
                            tuple_states,
                            config.allele_freqs,
                            child_cpds,
                            genes=gene_list,
                            allele_freqs=config.allele_freqs,
                            base_gen_states=GENOTYPE_STATES,
                        )
                        raw_belief, belief_metadata = _build_belief_map_with_progress(
                            pedigree,
                            tuple_states,
                            joint_dist,
                            label=progress_label,
                            parallelism=belief_parallelism,
                            return_metadata=True,
                        )
                        belief_map_tracking = dict(belief_metadata)
                        belief_exact = {}
                        precomputed_results = {}
                        conversion_desc = (
                            (progress_label and f"{progress_label} · inference tuples")
                            or "Converting belief states"
                        )
                        for state, dist_map in tqdm(
                            raw_belief.items(),
                            total=len(raw_belief),
                            desc=conversion_desc,
                            unit="state",
                            leave=True,
                        ):
                            tuple_pmfs = {person: dict(person_dist) for person, person_dist in dist_map.items()}
                            per_gene = lift_tuple_posteriors_to_genes(tuple_pmfs, gene_list)
                            primary_gene = gene_list[0]
                            aggregated = {
                                person: {g: 0.0 for g in GENOTYPE_STATES}
                                for person in individuals
                            }
                            primary_slice = per_gene.get(primary_gene, {})
                            for person, probs in primary_slice.items():
                                aggregated[person] = dict(probs)
                            inference_result = InferenceResult(
                                aggregated,
                                tuple_pmfs,
                                per_gene=per_gene,
                                gene_order=gene_list,
                                gen_states=GENOTYPE_STATES,
                            )
                            belief_exact[state] = inference_result
                            precomputed_results[state] = inference_result
                else:
                    exact_gen_states = gen_states
                    _record_exact_progress(
                        "belief_map_construction",
                        "start",
                        mode="dense",
                        gene_count=0,
                        tuple_mode=False,
                    )
                    joint_dist = build_full_joint(pedigree, gen_states, config.allele_freq, child_cpds)
                    raw_belief, belief_metadata = _build_belief_map_with_progress(
                        pedigree,
                        gen_states,
                        joint_dist,
                        label=progress_label,
                        parallelism=belief_parallelism,
                        return_metadata=True,
                    )
                    belief_map_tracking = dict(belief_metadata)
                    belief_exact = {}
                    precomputed_results = {}
                    conversion_desc = (
                        (progress_label and f"{progress_label} · inference scalars")
                        or "Converting belief states"
                    )
                    for state, marginal in tqdm(
                        raw_belief.items(),
                        total=len(raw_belief),
                        desc=conversion_desc,
                        unit="state",
                        leave=True,
                    ):
                        inference_result = InferenceResult(
                            marginal,
                            gene_order=("gene",),
                            gen_states=GENOTYPE_STATES,
                        )
                        belief_exact[state] = inference_result
                        precomputed_results[state] = inference_result
                exact_dp_belief_time = time.time() - start_belief_build
                belief_map_tracking.update(
                    {
                        "progress_status": "completed",
                        "state_count": int(len(belief_exact)),
                        "elapsed_sec": float(exact_dp_belief_time),
                    }
                )
                _record_exact_progress(
                    "belief_map_construction",
                    "completed",
                    mode=belief_map_tracking.get("mode"),
                    state_count=belief_map_tracking.get("state_count"),
                    total_indexed_state_count=belief_map_tracking.get("total_indexed_state_count"),
                    elapsed_sec=float(exact_dp_belief_time),
                )
                _save_belief_snapshot(belief_snapshot_key, belief_exact)

            if use_primal_exact:
                start_exact_solve = time.time()
                _record_exact_progress(
                    "exact_primal_solve",
                    "start",
                    state_count=int(len(belief_exact)) if belief_exact is not None else None,
                )
                Phi_star_exact_gene = None
                Phi_star_exact, policy_exact = solve_exact_dp_primal(
                    individuals,
                    exact_gen_states,
                    mu0,
                    belief_exact,
                    config.a,
                    config.b,
                    config.c,
                    config.delta,
                    config.fixed_cost,
                    config.variable_cost,
                    genes=gene_list if multi_gene else None,
                    a_gene=config.a_gene if multi_gene else None,
                    b_gene=config.b_gene if multi_gene else None,
                    c_gene=config.c_gene if multi_gene else None,
                    delta_gene=config.delta_gene if multi_gene else None,
                    base_gen_states=GENOTYPE_STATES,
                )
                exact_primal_solve_time = time.time() - start_exact_solve
                exact_dual_tracking = {
                    "progress_status": "not_run_primal_exact",
                    "backend": None,
                    "lp_build_elapsed_sec": None,
                    "lp_solve_elapsed_sec": None,
                    "total_elapsed_sec": float(exact_primal_solve_time),
                    "state_count": int(len(belief_exact)) if belief_exact is not None else None,
                    "status": "Optimal",
                }
                _record_exact_progress(
                    "exact_primal_solve",
                    "completed",
                    state_count=int(len(belief_exact)) if belief_exact is not None else None,
                    elapsed_sec=float(exact_primal_solve_time),
                )
            else:
                Phi_star_exact_gene = None
                _record_exact_progress(
                    "exact_dual_solve",
                    "start",
                    state_count=int(len(belief_exact)) if belief_exact is not None else None,
                    backend=os.getenv("EXACT_DUAL_LP_SOLVER", "gurobi").strip().lower(),
                )
                phi_exact_maybe, exact_dual_metadata = solve_exact_dual_pulp(
                    individuals,
                    exact_gen_states,
                    mu0,
                    belief_exact,
                    config.a,
                    config.b,
                    config.c,
                    config.delta,
                    config.fixed_cost,
                    config.variable_cost,
                    genes=gene_list if multi_gene else None,
                    a_gene=config.a_gene if multi_gene else None,
                    b_gene=config.b_gene if multi_gene else None,
                    c_gene=config.c_gene if multi_gene else None,
                    delta_gene=config.delta_gene if multi_gene else None,
                    base_gen_states=GENOTYPE_STATES,
                    progress_callback=_record_exact_progress,
                    return_metadata=True,
                )
                exact_dual_tracking = dict(exact_dual_metadata)
                if isinstance(phi_exact_maybe, tuple):
                    Phi_star_exact, Phi_star_exact_gene = phi_exact_maybe
                else:
                    Phi_star_exact = phi_exact_maybe
                policy_exact = extract_exact_policy(
                    individuals,
                    exact_gen_states,
                    config.a,
                    config.b,
                    config.c,
                    config.delta,
                    Phi_star_exact,
                    belief_exact,
                    config.fixed_cost,
                    config.variable_cost,
                    genes=gene_list if multi_gene else None,
                    Phi_star_gene=Phi_star_exact_gene if multi_gene else None,
                    a_gene=config.a_gene if multi_gene else None,
                    b_gene=config.b_gene if multi_gene else None,
                    c_gene=config.c_gene if multi_gene else None,
                    delta_gene=config.delta_gene if multi_gene else None,
                    base_gen_states=GENOTYPE_STATES,
                )

        exact_dp_time = time.time() - start_exact_dp
        exact_dp_runtime_tracking = {
            "schema": "exact_dp_runtime_tracking.v1",
            "cache_status": exact_dp_cache_status,
            "exact_solver": requested_exact_solver,
            "belief_map": dict(belief_map_tracking),
            "exact_dual": dict(exact_dual_tracking),
            "exact_dp_belief_time_sec": float(exact_dp_belief_time),
            "exact_dp_solve_phase_sec": float(max(0.0, exact_dp_time - exact_dp_belief_time)),
            "exact_dp_total_time_sec": float(exact_dp_time),
            "progress_events": list(exact_progress_events),
        }

        cache_payload = {
            "Phi_star_exact": Phi_star_exact,
            "Phi_star_exact_gene": Phi_star_exact_gene,
            "policy_exact": policy_exact,
            "belief_exact": belief_exact,
            "precomputed_results": precomputed_results,
            "exact_dp_belief_time": exact_dp_belief_time,
            "exact_dp_total_time": exact_dp_time,
            "exact_dp_runtime_tracking": exact_dp_runtime_tracking,
            "exact_solver": requested_exact_solver,
            "timestamp": time.time(),
        }
        with _CACHE_LOCK:
            cache = _load_exact_dp_cache()
            cache[cache_key] = cache_payload
            _persist_exact_dp_cache()
        if _exact_cache_memory_only():
            print("Exact DP cache miss – kept freshly computed results in memory only.")
        else:
            print("Exact DP cache miss – stored freshly computed results.")

    if exact_dual_tracking.get("progress_status") == "cache_hit":
        exact_dual_tracking.setdefault("total_elapsed_sec", float(max(0.0, exact_dp_time - exact_dp_belief_time)))
        exact_dual_tracking.setdefault("lp_solve_elapsed_sec", exact_dual_tracking.get("total_elapsed_sec"))
    belief_map_tracking.setdefault("state_count", int(len(belief_exact)) if belief_exact is not None else None)
    belief_map_tracking.setdefault("elapsed_sec", float(exact_dp_belief_time))
    exact_dp_runtime_tracking = {
        "schema": "exact_dp_runtime_tracking.v1",
        "cache_status": exact_dp_cache_status,
        "exact_solver": requested_exact_solver,
        "belief_map": dict(belief_map_tracking),
        "exact_dual": dict(exact_dual_tracking),
        "exact_dp_belief_time_sec": float(exact_dp_belief_time),
        "exact_dp_solve_phase_sec": float(max(0.0, exact_dp_time - exact_dp_belief_time)),
        "exact_dp_total_time_sec": float(exact_dp_time),
        "progress_events": list(exact_progress_events),
    }
    _record_exact_progress(
        "exact_dp",
        "completed",
        cache_status=exact_dp_cache_status,
        belief_state_count=int(len(belief_exact)) if belief_exact is not None else None,
        exact_dp_total_time_sec=float(exact_dp_time),
    )
    exact_dp_runtime_tracking["progress_events"] = list(exact_progress_events)

    if infer is None:
        if belief_exact is None:
            raise RuntimeError("Belief map missing; cannot build pgmpy-free inference backend.")
        infer = BeliefMapInference(
            belief_exact,
            genes=gene_list if multi_gene else None,
            model=bn_model,
        )
    
    # ------------------------------------------------------------------
    # 3. Solve the ADP SECOND (using cached probabilities from exact DP)
    # ------------------------------------------------------------------
    print("\n--- Running Approximate Dual DP Solver SECOND (using exact DP cache) ---")
    _notify_phase("adp_solve")
    start_adp = time.time()
    
    # Pass exact DP beliefs to ADP for cache reuse
    env_flag = os.getenv("ENABLE_TUPLE_ROWGEN")
    if multi_gene:
        tuple_mode_active = env_flag != "0"
    else:
        tuple_mode_active = False
    per_gene_phi_active = os.getenv("ENABLE_PER_GENE_PHI", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    theta_model_info = theta_model_metadata()
    theta_mode = resolve_theta_mode(os.getenv("THETA_MODE", "scalar"), theta_model_info.get("theta_model"))
    if theta_mode not in {"scalar", "stage", "person", "person_stage", "stage_gene"}:
        raise ValueError(
            "Unknown THETA_MODE="
            f"{theta_mode!r} (expected 'scalar', 'stage', 'person', 'person_stage', or 'stage_gene')."
        )
    if theta_mode == "stage_gene":
        if not multi_gene:
            raise ValueError("THETA_MODE='stage_gene' requires multi-gene config (genes must be configured).")
        if not tuple_mode_active:
            raise ValueError("THETA_MODE='stage_gene' requires ENABLE_TUPLE_ROWGEN=1.")
        if not per_gene_phi_active:
            raise ValueError("THETA_MODE='stage_gene' requires ENABLE_PER_GENE_PHI=1.")
    if precomputed_results is None:
        precomputed_results = {}
    if multi_gene:
        precomputed_beliefs = precomputed_results if tuple_mode_active else None
    else:
        precomputed_beliefs = precomputed_results

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}

    oracle_adp_enabled = _env_bool("ORACLE_ADP_ENABLED", False)
    oracle_adp_payload = None
    if oracle_adp_enabled:
        oracle_adp_payload = {
            "exact_values": Phi_star_exact,
            "policy_exact": policy_exact,
            "exact_root_value": Phi_star_exact.get(frozenset(), 0.0),
        }

    adp_max_iters = int(os.getenv("ADP_MAX_ITERS", "150"))
    adp_tol = float(os.getenv("ADP_TOL", "1e-6"))
    
    need_adp_stats = bool(adp_return_stats or adp_analysis_mode)
    adp_payload = solve_dual_dp_with_domain(
        I=individuals, gen_states=gen_states,
        mu0=mu0, a=config.a, b=config.b, c=config.c, delta=config.delta,
        x=x, allele_freq=config.allele_freq, child_cpds=child_cpds, pedigree=pedigree,
        p0=initial_p, z0=initial_z,
        p0_gene=initial_p_gene if multi_gene else None,
        genes=config.genes if config.genes else None,
        a_gene=config.a_gene if config.a_gene else None,
        b_gene=config.b_gene if config.b_gene else None,
        c_gene=config.c_gene if config.c_gene else None,
        delta_gene=config.delta_gene if config.delta_gene else None,
        infer = infer,
        max_iters=adp_max_iters, tol=adp_tol, verbose=verbose,
        debug_lp_path="master_debug_w_theta.lp",
        fixed_cost=config.fixed_cost, variable_cost=config.variable_cost,
        precomputed_beliefs=precomputed_beliefs,
        cache_diagnostics=False,  # Disable verbose cache diagnostics for stability
        return_stats=need_adp_stats,
        analysis_mode=adp_analysis_mode,
        return_phi_eval=True,
        oracle_adp_payload=oracle_adp_payload,
    )
    adp_stats = None
    if need_adp_stats:
        adp_payload, adp_stats = adp_payload
    Phi_star_approx, W_star, belief_post_approx, theta_star, master, adp_inference_time, adp_cache, phi_eval_approx = adp_payload
    
    adp_total_time = time.time() - start_adp
    adp_solve_time = adp_total_time - adp_inference_time  # Pure solve time excluding inference
    aaub_star = getattr(adp_cache, "aaub_star", None) if adp_cache else None
    edge_star = getattr(adp_cache, "edge_star", None) if adp_cache else None
    trio_star = getattr(adp_cache, "trio_star", None) if adp_cache else None
    myopic_adp_star = getattr(adp_cache, "myopic_adp_star", None) if adp_cache else None
    oracle_adp_star = getattr(adp_cache, "oracle_adp_star", None) if adp_cache else None
    regime_residual_star = getattr(adp_cache, "regime_residual_star", None) if adp_cache else None
    pedigree_edges_cached = getattr(adp_cache, "pedigree_edges", None) if adp_cache else None
    pedigree_trios_cached = getattr(adp_cache, "pedigree_trios", None) if adp_cache else None
    slack_refactor = getattr(adp_cache, "slack_refactor", None) if adp_cache else None
    w_gene_star = getattr(adp_cache, "w_gene_star", None) if adp_cache else None
    rowgen_telemetry = getattr(adp_cache, "rowgen_telemetry", None) if adp_cache else None
    runtime_telemetry = getattr(adp_cache, "runtime_telemetry", None) if adp_cache else None
    candidate_pre_polish = getattr(adp_cache, "candidate_pre_polish", None) if adp_cache else None
    candidate_post_polish = getattr(adp_cache, "candidate_post_polish", None) if adp_cache else None
    candidate_eval_payloads = getattr(adp_cache, "candidate_eval_payloads", None) if adp_cache else None
    if adp_stats:
        edge_star = adp_stats.get("edge_star", edge_star)
        trio_star = adp_stats.get("trio_star", trio_star)
        myopic_adp_star = adp_stats.get("myopic_adp_star", myopic_adp_star)
        oracle_adp_star = adp_stats.get("oracle_adp_star", oracle_adp_star)
        regime_residual_star = adp_stats.get("regime_residual_star", regime_residual_star)
        w_gene_star = adp_stats.get("W_gene_star", w_gene_star)
        root_diagnostics_stats = adp_stats.get("root_diagnostics")
        if isinstance(root_diagnostics_stats, dict):
            rowgen_telemetry = root_diagnostics_stats.get("rowgen_telemetry", rowgen_telemetry)
            runtime_telemetry = root_diagnostics_stats.get("runtime_telemetry", runtime_telemetry)

    if adp_analysis_mode:
        _notify_phase("policy_eval")
        analysis_root = None
        if isinstance(adp_stats, dict):
            analysis_root = adp_stats.get("root_diagnostics")
        results = {
            "analysis_mode": adp_analysis_mode,
            "Exact_DP_root_value": Phi_star_exact.get(frozenset(), 0.0),
            "Phi_star_exact": Phi_star_exact,
            "Phi_star_exact_gene": Phi_star_exact_gene,
            "belief_exact": belief_exact,
            "Phi_star_approx": Phi_star_approx,
            "Phi_eval_root": phi_eval_approx.get(frozenset()) if isinstance(phi_eval_approx, dict) else None,
            "theta_star": theta_star,
            "W_star": W_star,
            "W_star_gene": w_gene_star,
            "edge_star": edge_star,
            "trio_star": trio_star,
            "myopic_adp_star": myopic_adp_star,
            "oracle_adp_star": oracle_adp_star,
            "rowgen_telemetry": rowgen_telemetry,
            "runtime_telemetry": runtime_telemetry,
            "root_diagnostics": analysis_root,
            "adp_stats": adp_stats,
        }
        if return_infer:
            results["infer"] = infer
        return results

    ratio2_delta_tol = float(os.getenv("RATIO2_DELTA_TOL", "0.003"))
    ratio2_mean_delta_tol = float(os.getenv("RATIO2_MEAN_DELTA_TOL", "0.001"))
    gap2_delta_tol = float(os.getenv("GAP2_DELTA_TOL", "1e-6"))
    denom_small_eps = float(os.getenv("DENOM_SMALL_EPS", "1e-9"))

    if dfvr_eval_no_mutation is None:
        dfvr_eval_no_mutation = _env_bool("DFVR_EVAL_NO_MUTATION", True)
    if dfvr_enforce_fixed_stateset is None:
        dfvr_enforce_fixed_stateset = _env_bool("DFVR_ENFORCE_FIXED_STATESET", True)
    if dfvr_emit_fixed_stateset is None:
        dfvr_emit_fixed_stateset = _env_bool("DFVR_EMIT_FIXED_STATESET", True)
    if dfvr_fixed_states_path is None:
        raw_stateset = os.getenv("DFVR_FIXED_STATESET_PATH", "").strip()
        dfvr_fixed_states_path = raw_stateset or None

    # ------------------------------------------------------------------
    # 3.5. Cache Diagnostics
    # ------------------------------------------------------------------
    if adp_cache and not verbose:  # Only print if not in verbose mode to avoid overwhelming output
        adp_cache.print_cache_summary(f"- Test Case")
        if adp_cache.get_cache_stats()['hit_rate'] < 0.9:  # Show details for poor cache performance
            adp_cache.print_state_details(max_items=5)

    # ------------------------------------------------------------------
    # 3.6. Optional DFVR residual bound (post-ADP solve)
    # ------------------------------------------------------------------
    dfvr_results = None
    dfvr_fixed_states_loaded = False
    dfvr_fixed_states_written = False
    dfvr_fixed_states_path_resolved = None
    if dfvr_bound:
        from ..optimisation.dfvr_bound import (
            compute_dfvr_bound,
            load_dfvr_stateset,
            save_dfvr_stateset,
        )

        fixed_states = None
        fixed_states_path = None
        if dfvr_fixed_states_path:
            fixed_states_path = Path(dfvr_fixed_states_path).expanduser()
            dfvr_fixed_states_path_resolved = fixed_states_path
            if fixed_states_path.exists():
                fixed_states = load_dfvr_stateset(fixed_states_path)
                dfvr_fixed_states_loaded = True
            elif dfvr_enforce_fixed_stateset and not dfvr_emit_fixed_stateset:
                raise RuntimeError(
                    f"DFVR fixed state-set file does not exist: {fixed_states_path}"
                )

        dfvr_belief = belief_post_approx
        if dfvr_state_mode == "exhaustive" and belief_exact is not None:
            # DFVR exhaustive mode iterates over the full DP state space. Reuse the
            # exact-DP belief snapshot (already materialized for all states) to
            # avoid triggering 100k+ inference calls for larger pedigrees.
            dfvr_belief = {state: (entry, {}) for state, entry in belief_exact.items()}
        dfvr_results = compute_dfvr_bound(
            Phi_star=Phi_star_approx,
            W_star=W_star,
            belief=dfvr_belief,
            theta_star=theta_star,
            theta_mode=theta_mode,
            individuals=individuals,
            gen_states=gen_states,
            infer=infer,
            config=config,
            tuple_mode=tuple_mode_active,
            genes=config.genes if config.genes else None,
            state_mode=dfvr_state_mode,
            max_states=dfvr_max_states,
            random_seed=dfvr_random_seed,
            top_k=dfvr_top_k,
            max_outcomes_per_action=dfvr_max_outcomes_per_action,
            aaub_star=aaub_star,
            W_edge_star=edge_star,
            pedigree_edges=pedigree_edges_cached,
            W_trio_star=trio_star,
            pedigree_trios=pedigree_trios_cached,
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
            no_mutation=bool(dfvr_eval_no_mutation),
            fixed_states=fixed_states,
            enforce_fixed_states=bool(dfvr_enforce_fixed_stateset and fixed_states is not None),
            collect_state_list=bool(
                fixed_states_path is not None and (not dfvr_fixed_states_loaded) and dfvr_emit_fixed_stateset
            ),
        )

        if fixed_states_path is not None and (not dfvr_fixed_states_loaded) and dfvr_emit_fixed_stateset:
            state_list = dfvr_results.get("state_list")
            if not isinstance(state_list, list) or not state_list:
                raise RuntimeError(
                    "DFVR fixed state-set emission requested but compute_dfvr_bound returned no states."
                )
            save_dfvr_stateset(
                fixed_states_path,
                state_list,
                metadata={
                    "state_mode": dfvr_state_mode,
                    "theta_mode": theta_mode,
                    "generated_for": "dfvr_eval",
                },
            )
            dfvr_fixed_states_written = True

        if fixed_states is not None and dfvr_enforce_fixed_stateset:
            integrity = dfvr_results.get("fixed_state_integrity")
            if not isinstance(integrity, dict) or not integrity.get("match", False):
                raise RuntimeError(
                    "DFVR fixed state-set integrity check failed: "
                    f"{integrity!r}"
                )

    # ------------------------------------------------------------------
    # 4. Extract and evaluate policies (baseline + optional lookahead)
    # ------------------------------------------------------------------
    _notify_phase("policy_eval")
    policy_metrics = {}
    tuple_posteriors = getattr(adp_cache, "tuple_posteriors", {})
    tuple_pmfs_policy = tuple_posteriors if tuple_posteriors else None
    belief_gene_post = {}
    if multi_gene:
        for state, (posteriors, _) in belief_post_approx.items():
            if isinstance(posteriors, InferenceResult):
                per_gene = posteriors.get_per_gene_probs()
                belief_gene_post[state] = per_gene
            else:
                belief_gene_post[state] = lift_single_gene_posteriors_to_genes(posteriors, config.genes)
    else:
        belief_gene_post = None

    # Prefer exact-belief snapshots for policy evaluation when available so the
    # policy value uses the same model as the exact DP (avoids mismatch with ADP belief map).
    belief_eval = belief_post_approx
    belief_gene_eval = belief_gene_post
    tuple_pmfs_eval = tuple_pmfs_policy
    if belief_exact is not None:
        belief_eval = {state: (entry, {}) for state, entry in belief_exact.items()}
        if multi_gene:
            belief_gene_eval = {}
            for state, entry in belief_exact.items():
                if isinstance(entry, InferenceResult):
                    belief_gene_eval[state] = entry.get_per_gene_probs()
                else:
                    belief_gene_eval[state] = lift_single_gene_posteriors_to_genes(entry, config.genes)
        else:
            belief_gene_eval = None
        if tuple_mode_active:
            tuple_pmfs_eval = {}
            for state, entry in belief_exact.items():
                if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
                    tuple_pmfs_eval[state] = entry.get_tuple_pmfs()
        else:
            tuple_pmfs_eval = None
        if tuple_pmfs_policy is None and tuple_pmfs_eval:
            tuple_pmfs_policy = tuple_pmfs_eval

    tuple_mode_enabled = tuple_mode_active and tuple_pmfs_policy is not None
    tuple_pmfs_for_eval = tuple_pmfs_eval if tuple_pmfs_eval is not None else tuple_pmfs_policy

    def _evidence_state(state):
        if not isinstance(state, frozenset):
            raise AssertionError(
                f"State must be evidence-only frozenset[(person,outcome)], got {type(state).__name__}: {state!r}"
            )
        return state

    polish_acceptance_reason = "accepted"
    polish_acceptance_decision = "accepted"
    selected_candidate_id = "post_polish"
    polish_candidate_metrics = {}
    exact_root_value_for_guardrail = float(Phi_star_exact.get(frozenset(), 0.0))
    root_stop_constraint = master.getConstrByName("root_stop")
    root_stop_rhs_for_guardrail = float(root_stop_constraint.RHS) if root_stop_constraint is not None else 0.0

    def _evaluate_candidate_snapshot(payload: dict | None, candidate_id: str):
        if not isinstance(payload, dict):
            return None
        theta_snapshot = payload.get("theta_star")
        w_snapshot = payload.get("W_star")
        aaub_snapshot = payload.get("aaub_star")
        edge_snapshot = payload.get("edge_star")
        trio_snapshot = payload.get("trio_star")
        myopic_adp_snapshot = payload.get("myopic_adp_star")
        oracle_adp_snapshot = payload.get("oracle_adp_star")
        regime_residual_snapshot = payload.get("regime_residual_star")
        phi_snapshot = payload.get("Phi_star")
        if theta_snapshot is None or not isinstance(w_snapshot, dict):
            return None
        phi_for_policy = phi_snapshot if isinstance(phi_snapshot, dict) else None
        memo = {}
        policy_map = {}
        feature_cache = {}
        for state in list(belief_post_approx.keys()):
            policy_map[state] = best_action(
                state,
                theta_star=theta_snapshot,
                W_star=w_snapshot,
                belief=belief_post_approx,
                theta_mode=theta_mode,
                pedigree=pedigree,
                theta_model=theta_model_info.get("theta_model"),
                theta_model_spec=theta_model_info.get("theta_model_spec"),
                individuals=individuals,
                gen_states=gen_states,
                r_reward_testp=r_reward_testp,
                a=config.a,
                b=config.b,
                c=config.c,
                delta=config.delta,
                infer=infer,
                fixed_cost=config.fixed_cost,
                variable_cost=config.variable_cost,
                lookahead_depth=0,
                _memo=memo,
                belief_gene=belief_gene_post,
                genes=config.genes if config.genes else None,
                a_gene=config.a_gene if config.a_gene else None,
                b_gene=config.b_gene if config.b_gene else None,
                c_gene=config.c_gene if config.c_gene else None,
                delta_gene=config.delta_gene if config.delta_gene else None,
                tuple_pmfs=tuple_pmfs_policy,
                tuple_mode=tuple_mode_enabled,
                phi_values=phi_for_policy,
                aaub_star=aaub_snapshot,
                W_edge_star=edge_snapshot,
                pedigree_edges=pedigree_edges_cached,
                W_trio_star=trio_snapshot,
                pedigree_trios=pedigree_trios_cached,
                feature_cache=feature_cache,
                myopic_adp_star=myopic_adp_snapshot,
                oracle_adp_star=oracle_adp_snapshot,
                regime_residual_star=regime_residual_snapshot,
            )
        value_map = exact_value_under_policy(
            policy=policy_map,
            belief=belief_eval,
            individuals=individuals,
            gen_states=gen_states,
            r_reward_test=r_reward_test,
            a=config.a,
            b=config.b,
            c=config.c,
            delta=config.delta,
            infer=infer,
            theta_star=theta_snapshot,
            W_star=w_snapshot,
            theta_mode=theta_mode,
            pedigree=pedigree,
            theta_model=theta_model_info.get("theta_model"),
            theta_model_spec=theta_model_info.get("theta_model_spec"),
            fixed_cost=config.fixed_cost,
            variable_cost=config.variable_cost,
            lookahead_depth=0,
            belief_gene=belief_gene_eval,
            genes=config.genes if config.genes else None,
            a_gene=config.a_gene if config.a_gene else None,
            b_gene=config.b_gene if config.b_gene else None,
            c_gene=config.c_gene if config.c_gene else None,
            delta_gene=config.delta_gene if config.delta_gene else None,
            tuple_pmfs=tuple_pmfs_for_eval if tuple_mode_enabled else None,
            aaub_star=aaub_snapshot,
            W_edge_star=edge_snapshot,
            pedigree_edges=pedigree_edges_cached,
            W_trio_star=trio_snapshot,
            pedigree_trios=pedigree_trios_cached,
            feature_cache=feature_cache,
            myopic_adp_star=myopic_adp_snapshot,
            oracle_adp_star=oracle_adp_snapshot,
            regime_residual_star=regime_residual_snapshot,
        )
        root_candidates = [state for state in policy_map.keys() if len(_evidence_state(state)) == 0]
        root_key = root_candidates[0] if root_candidates else frozenset()
        policy_root = value_map.get(root_key)
        if policy_root is None and root_key != frozenset():
            policy_root = value_map.get(frozenset())
        if policy_root is None:
            policy_root = 0.0
        denom = exact_root_value_for_guardrail - root_stop_rhs_for_guardrail
        gap2 = exact_root_value_for_guardrail - float(policy_root)
        ratio2 = gap2 / denom if denom > denom_small_eps else None
        return {
            "candidate_id": candidate_id,
            "policy_root_value": float(policy_root),
            "exact_root_value": exact_root_value_for_guardrail,
            "root_stop_rhs": root_stop_rhs_for_guardrail,
            "denom": float(denom),
            "gap2": float(gap2),
            "ratio2": float(ratio2) if ratio2 is not None else None,
        }

    pre_payload = None
    post_payload = None
    if isinstance(candidate_eval_payloads, dict):
        pre_payload = candidate_eval_payloads.get("pre_polish_payload")
        post_payload = candidate_eval_payloads.get("post_polish_payload")
    pre_metrics = _evaluate_candidate_snapshot(pre_payload, "pre_polish")
    post_metrics = _evaluate_candidate_snapshot(post_payload, "post_polish")
    if pre_metrics is not None:
        polish_candidate_metrics["pre_polish"] = pre_metrics
    if post_metrics is not None:
        polish_candidate_metrics["post_polish"] = post_metrics

    (
        polish_acceptance_reason,
        polish_acceptance_decision,
        selected_candidate_id,
    ) = _decide_polish_acceptance(
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        ratio2_delta_tol=ratio2_delta_tol,
        gap2_delta_tol=gap2_delta_tol,
        denom_small_eps=denom_small_eps,
    )

    if polish_acceptance_reason != "accepted":
        if isinstance(pre_payload, dict):
            theta_star = pre_payload.get("theta_star", theta_star)
            W_star = pre_payload.get("W_star", W_star)
            w_gene_star = pre_payload.get("W_star_gene", w_gene_star)
            aaub_star = pre_payload.get("aaub_star", aaub_star)
            edge_star = pre_payload.get("edge_star", edge_star)
            trio_star = pre_payload.get("trio_star", trio_star)
            myopic_adp_star = pre_payload.get("myopic_adp_star", myopic_adp_star)
            oracle_adp_star = pre_payload.get("oracle_adp_star", oracle_adp_star)
            regime_residual_star = pre_payload.get("regime_residual_star", regime_residual_star)
            pre_phi_star = pre_payload.get("Phi_star")
            if isinstance(pre_phi_star, dict):
                Phi_star_approx = pre_phi_star
            phi_eval_approx = None
    else:
        if isinstance(post_payload, dict):
            post_phi_star = post_payload.get("Phi_star")
            if isinstance(post_phi_star, dict):
                Phi_star_approx = post_phi_star
            edge_star = post_payload.get("edge_star", edge_star)
            trio_star = post_payload.get("trio_star", trio_star)
            myopic_adp_star = post_payload.get("myopic_adp_star", myopic_adp_star)
            oracle_adp_star = post_payload.get("oracle_adp_star", oracle_adp_star)
            regime_residual_star = post_payload.get("regime_residual_star", regime_residual_star)

    policy_phi_values = phi_eval_approx if phi_eval_approx else (Phi_star_approx if isinstance(Phi_star_approx, dict) else None)

    non_terminal_states_exact = {
        s: v for s, v in policy_exact.items() if len(_evidence_state(s)) < len(individuals)
    }

    clamp_tol = 1e-9
    best_root_value = None
    best_policy_metrics = None
    best_policy_map = None

    for depth in depth_list:
        memo = {}
        policy_map = {}
        feature_cache = {}
        for s in list(belief_post_approx.keys()):
            policy_map[s] = best_action(
                s,
                theta_star=theta_star,
                W_star=W_star,
                belief=belief_post_approx,
                theta_mode=theta_mode,
                pedigree=pedigree,
                theta_model=theta_model_info.get("theta_model"),
                theta_model_spec=theta_model_info.get("theta_model_spec"),
                individuals=individuals,
                gen_states=gen_states,
                r_reward_testp=r_reward_testp,
                a=config.a,
                b=config.b,
                c=config.c,
                delta=config.delta,
                infer=infer,
                fixed_cost=config.fixed_cost,
                variable_cost=config.variable_cost,
                lookahead_depth=depth,
                _memo=memo,
                belief_gene=belief_gene_post,
                genes=config.genes if config.genes else None,
                a_gene=config.a_gene if config.a_gene else None,
                b_gene=config.b_gene if config.b_gene else None,
                c_gene=config.c_gene if config.c_gene else None,
                delta_gene=config.delta_gene if config.delta_gene else None,
                tuple_pmfs=tuple_pmfs_policy,
                tuple_mode=tuple_mode_enabled,
                phi_values=policy_phi_values if policy_phi_values else None,
                aaub_star=aaub_star,
                W_edge_star=edge_star,
                pedigree_edges=pedigree_edges_cached,
                W_trio_star=trio_star,
                pedigree_trios=pedigree_trios_cached,
                feature_cache=feature_cache,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )

        if print_policies:
            header = (
                "--- Approximate Policy (baseline) ---"
                if depth == 0 else f"--- Rollout Policy (lookahead depth={depth}) ---"
            )
            print(f"\n{header}")
            for s, (act, who, val) in policy_map.items():
                evidence = _evidence_state(s)
                if len(evidence) >= len(individuals):
                    continue
                lbl = "{" + ",".join(f'{i}={g}' for i, g in sorted(evidence)) + "}" if evidence else "∅"
                who_lbl = "" if who is None else str(who)
                print(f"{lbl:25} → {act.upper():4} {who_lbl:8}  Φ̂ = {val:.4f}")

        policy_map_evidence = policy_map
        policy_map_for_eval = policy_map

        V_exact_policy = exact_value_under_policy(
            policy=policy_map_for_eval,
            belief=belief_eval,
            individuals=individuals,
            gen_states=gen_states,
            r_reward_test=r_reward_test,
            a=config.a,
            b=config.b,
            c=config.c,
            delta=config.delta,
            infer=infer,
            theta_star=theta_star,
            W_star=W_star,
            theta_mode=theta_mode,
            pedigree=pedigree,
            theta_model=theta_model_info.get("theta_model"),
            theta_model_spec=theta_model_info.get("theta_model_spec"),
            fixed_cost=config.fixed_cost,
            variable_cost=config.variable_cost,
            lookahead_depth=depth,
            belief_gene=belief_gene_eval,
            genes=config.genes if config.genes else None,
            a_gene=config.a_gene if config.a_gene else None,
            b_gene=config.b_gene if config.b_gene else None,
            c_gene=config.c_gene if config.c_gene else None,
            delta_gene=config.delta_gene if config.delta_gene else None,
            tuple_pmfs=tuple_pmfs_for_eval if tuple_mode_enabled else None,
            aaub_star=aaub_star,
            W_edge_star=edge_star,
            pedigree_edges=pedigree_edges_cached,
            W_trio_star=trio_star,
            pedigree_trios=pedigree_trios_cached,
            feature_cache=feature_cache,
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
        )

        root_candidates = [s for s in policy_map_for_eval.keys() if len(_evidence_state(s)) == 0]
        root_key = root_candidates[0] if root_candidates else frozenset()
        root_value = V_exact_policy.get(root_key)
        if root_value is None and root_key != frozenset():
            root_value = V_exact_policy.get(frozenset())
        if root_value is None:
            root_value = 0.0
        if verbose:
            print(
                f"Exact value under {'baseline' if depth == 0 else f'lookahead depth {depth}'} policy at root = {root_value:.4f}"
            )

        # Compare against exact policy on evidence-level states.
        common_states_set = set(policy_map_evidence.keys()) & set(non_terminal_states_exact.keys())
        diff_count = 0
        policy_loss = 0.0
        for s in sorted(list(common_states_set), key=lambda x: str(x)):
            action_approx, _, value_approx = policy_map_evidence[s]
            action_exact, _, value_exact = non_terminal_states_exact[s]
            if action_approx != action_exact:
                diff_count += 1
                policy_loss += abs(value_approx - value_exact)
                if verbose and depth == 0:
                    print(f"State {s}: Approximate: {action_approx}, Exact: {action_exact}")

        if verbose:
            if diff_count == 0:
                print("Policies are identical on common states.")
            else:
                print(
                    f"Policies differ on {diff_count} out of {len(common_states_set)} common states."
                )
                print(f"Average Policy Loss: {policy_loss / diff_count:.4f}")

        current_metrics = {
            "root_value": root_value,
            "policy_differences": diff_count,
            "common_states": len(common_states_set),
            "avg_policy_loss": policy_loss / diff_count if diff_count > 0 else 0.0,
            "clamped_to_best": False,
            "depth": depth,
        }

        if best_root_value is None:
            policy_metrics[depth] = current_metrics.copy()
            best_root_value = root_value
            best_policy_metrics = policy_metrics[depth]
            best_policy_map = policy_map
            baseline_policy = policy_map if depth == 0 else policy_map
            continue

        if depth == 0:
            policy_metrics[depth] = current_metrics.copy()
            best_root_value = root_value
            best_policy_metrics = policy_metrics[depth]
            best_policy_map = policy_map
            baseline_policy = policy_map
        else:
            if root_value > best_root_value + clamp_tol:
                policy_metrics[depth] = current_metrics.copy()
                best_root_value = root_value
                best_policy_metrics = policy_metrics[depth]
                best_policy_map = policy_map
            else:
                clamped_metrics = best_policy_metrics.copy()
                clamped_metrics['depth'] = depth
                clamped_metrics['clamped_to_best'] = True
                clamped_metrics['clamped_from_depth'] = best_policy_metrics.get('depth', 0)
                policy_metrics[depth] = clamped_metrics
                policy_map = best_policy_map
                print(
                    f"      (lookahead depth={depth} clamped to depth {clamped_metrics.get('clamped_from_depth', 0)})"
                )

    exact_root_value = Phi_star_exact.get(frozenset(), 0.0)
    if print_policies:
        print("\n--- Exact Policy ---")
        for s, (act, who, val) in policy_exact.items():
            if len(s) >= len(individuals):
                continue
            lbl = "{" + ",".join(f'{i}={g}' for i, g in sorted(s)) + "}" if s else "∅"
            who_lbl = "" if who is None else str(who)
            print(f"{lbl:25} → {act.upper():4} {who_lbl:8}  Φ̂ = {val:.4f}")

    print(f"Exact value at root = {exact_root_value:.4f}")

    # Report baseline (depth=0 when available) to keep gap-suite parity.
    if policy_metrics:
        baseline_depth = 0 if 0 in policy_metrics else min(policy_metrics)
        baseline_stats = policy_metrics[baseline_depth]
        report_depth = baseline_stats.get("depth", baseline_depth)
        # Also keep best-performing policy info for optional diagnostics.
        def _gap_to_exact(k):
            return abs(exact_root_value - policy_metrics[k]["root_value"])
        best_depth_key = min(policy_metrics, key=_gap_to_exact)
        best_stats = policy_metrics[best_depth_key]
    else:
        report_depth = best_policy_metrics.get("depth", depth_list[-1]) if best_policy_metrics else depth_list[-1]
        baseline_stats = best_policy_metrics or policy_metrics.get(report_depth) or next(iter(policy_metrics.values()))
        best_stats = baseline_stats
    approximate_root_value = baseline_stats["root_value"]
    diff_count = baseline_stats["policy_differences"]
    common_states = baseline_stats["common_states"]
    avg_policy_loss = baseline_stats["avg_policy_loss"]

    safe_rollout_enabled = _env_bool("SAFE_ROLLOUT_ENABLED", False)
    safe_rollout_top_k = max(1, int(os.getenv("SAFE_ROLLOUT_TOP_K", "1")))
    safe_rollout_incumbent_safe = _env_bool("SAFE_ROLLOUT_INCUMBENT_SAFE", True)
    safe_rollout_module = _resolve_safe_rollout_module() if safe_rollout_enabled else None
    safe_rollout_callable, safe_rollout_callable_name = _resolve_safe_rollout_callable(safe_rollout_module)
    production_policy_value = approximate_root_value
    production_policy_source = "adp_policy"
    safe_rollout_diagnostics = {
        "enabled": safe_rollout_enabled,
        "available": safe_rollout_callable is not None,
        "module_name": getattr(safe_rollout_module, "__name__", None) if safe_rollout_module else None,
        "callable_name": safe_rollout_callable_name,
        "top_k": safe_rollout_top_k if safe_rollout_enabled else None,
        "incumbent_safe": safe_rollout_incumbent_safe,
        "decision": "disabled" if not safe_rollout_enabled else "unavailable",
        "reason": "disabled" if not safe_rollout_enabled else "module_unavailable",
        "production_policy_value": production_policy_value,
        "production_policy_source": production_policy_source,
        "candidate_policy_value": best_stats.get("root_value"),
        "incumbent_policy_value": approximate_root_value,
        "exact_root_value": exact_root_value,
        "selected_candidate_id": selected_candidate_id,
    }

    def _policy_map_root_value(policy_map):
        if not isinstance(policy_map, dict) or not policy_map:
            return None
        policy_value_map = exact_value_under_policy(
            policy=dict(policy_map),
            belief=dict(belief_eval),
            individuals=individuals,
            gen_states=gen_states,
            r_reward_test=r_reward_test,
            a=config.a,
            b=config.b,
            c=config.c,
            delta=config.delta,
            infer=infer,
            theta_star=theta_star,
            W_star=W_star,
            theta_mode=theta_mode,
            pedigree=pedigree,
            theta_model=theta_model_info.get("theta_model"),
            theta_model_spec=theta_model_info.get("theta_model_spec"),
            fixed_cost=config.fixed_cost,
            variable_cost=config.variable_cost,
            lookahead_depth=0,
            belief_gene=dict(belief_gene_eval) if isinstance(belief_gene_eval, dict) else belief_gene_eval,
            genes=config.genes if config.genes else None,
            a_gene=config.a_gene if config.a_gene else None,
            b_gene=config.b_gene if config.b_gene else None,
            c_gene=config.c_gene if config.c_gene else None,
            delta_gene=config.delta_gene if config.delta_gene else None,
            tuple_pmfs=dict(tuple_pmfs_for_eval) if isinstance(tuple_pmfs_for_eval, dict) else tuple_pmfs_for_eval,
            aaub_star=aaub_star,
            W_edge_star=edge_star,
            pedigree_edges=pedigree_edges_cached,
            W_trio_star=trio_star,
            pedigree_trios=pedigree_trios_cached,
            feature_cache={},
            myopic_adp_star=myopic_adp_star,
            oracle_adp_star=oracle_adp_star,
            regime_residual_star=regime_residual_star,
        )
        root_candidates = [state for state in policy_value_map.keys() if len(_evidence_state(state)) == 0]
        root_key = root_candidates[0] if root_candidates else frozenset()
        root_value = policy_value_map.get(root_key)
        if root_value is None and root_key != frozenset():
            root_value = policy_value_map.get(frozenset())
        if root_value is None:
            return None
        try:
            return float(root_value)
        except (TypeError, ValueError):
            return None

    if safe_rollout_callable is not None:
        safe_rollout_context = {
            "enabled": True,
            "top_k": safe_rollout_top_k,
            "incumbent_safe": safe_rollout_incumbent_safe,
            "epsilon": 0.0,
            "exact_root_value": exact_root_value,
            "adp_root_value_phi": None,
            "adp_policy_value": approximate_root_value,
            "selected_candidate_id": selected_candidate_id,
            "candidate_policy_map": best_policy_map,
            "incumbent_policy_map": baseline_policy,
            "policy_exact": policy_exact,
            "belief": belief_eval,
            "belief_gene": belief_gene_eval,
            "tuple_pmfs": tuple_pmfs_for_eval if tuple_mode_enabled else None,
            "phi_values": policy_phi_values,
            "theta_star": theta_star,
            "W_star": W_star,
            "aaub_star": aaub_star,
            "theta_mode": theta_mode,
            "pedigree": pedigree,
            "theta_model": theta_model_info.get("theta_model"),
            "theta_model_spec": theta_model_info.get("theta_model_spec"),
            "W_edge_star": edge_star,
            "pedigree_edges": pedigree_edges_cached,
            "W_trio_star": trio_star,
            "pedigree_trios": pedigree_trios_cached,
            "myopic_adp_star": myopic_adp_star,
            "oracle_adp_star": oracle_adp_star,
            "regime_residual_star": regime_residual_star,
            "feature_cache": {},
            "individuals": individuals,
            "gen_states": gen_states,
            "infer": infer,
            "r_reward_test": r_reward_test,
            "r_reward_testp": r_reward_testp,
            "config": config,
        }
        try:
            safe_rollout_result = safe_rollout_callable(safe_rollout_context)
        except Exception as exc:  # pragma: no cover - helper contract failure
            safe_rollout_result = {
                "decision": "error",
                "reason": f"helper_error:{exc}",
                "error": repr(exc),
            }

        if isinstance(safe_rollout_result, dict):
            helper_summary = {
                key: value
                for key, value in safe_rollout_result.items()
                if key not in {
                    "policy_map",
                    "production_policy_map",
                    "selected_policy_map",
                    "incumbent_policy_map",
                }
            }
            safe_rollout_diagnostics.update(
                {
                    "available": True,
                    "decision": safe_rollout_result.get("decision", safe_rollout_result.get("guardrail_safe", "accepted")),
                    "reason": safe_rollout_result.get("reason", safe_rollout_result.get("decision_reason", "accepted")),
                    "production_policy_source": safe_rollout_result.get(
                        "production_policy_source",
                        safe_rollout_result.get("source", safe_rollout_result.get("production_choice", "safe_rollout")),
                    ),
                    "selected_candidate_id": safe_rollout_result.get(
                        "selected_candidate_id",
                        safe_rollout_result.get("safe_rollout_selected_candidate_id", selected_candidate_id),
                    ),
                    "candidate_policy_value": safe_rollout_result.get(
                        "candidate_policy_value",
                        safe_rollout_result.get("candidate_root_value", safe_rollout_diagnostics["candidate_policy_value"]),
                    ),
                    "incumbent_policy_value": safe_rollout_result.get(
                        "incumbent_policy_value",
                        safe_rollout_result.get("incumbent_root_value", safe_rollout_diagnostics["incumbent_policy_value"]),
                    ),
                    "exact_root_value": safe_rollout_result.get("exact_root_value", exact_root_value),
                    "helper_summary": helper_summary,
                }
            )
            production_policy_value = safe_rollout_result.get(
                "production_policy_value",
                safe_rollout_result.get("policy_value", None),
            )
            if production_policy_value is None:
                for policy_map_key in ("production_policy_map", "selected_policy_map", "policy_map"):
                    policy_map_candidate = safe_rollout_result.get(policy_map_key)
                    if isinstance(policy_map_candidate, dict):
                        production_policy_value = _policy_map_root_value(policy_map_candidate)
                        if production_policy_value is not None:
                            break
            if production_policy_value is None:
                production_policy_value = approximate_root_value
                safe_rollout_diagnostics["decision"] = "fallback_to_adp_policy"
                safe_rollout_diagnostics["reason"] = "missing_policy_value"
                safe_rollout_diagnostics["production_policy_source"] = "adp_policy"
            try:
                production_policy_value = float(production_policy_value)
            except (TypeError, ValueError):
                production_policy_value = approximate_root_value
                safe_rollout_diagnostics["decision"] = "fallback_to_adp_policy"
                safe_rollout_diagnostics["reason"] = "invalid_policy_value"
                safe_rollout_diagnostics["production_policy_source"] = "adp_policy"
            production_policy_source = safe_rollout_diagnostics["production_policy_source"]
            safe_rollout_diagnostics["production_policy_value"] = production_policy_value
        else:
            safe_rollout_diagnostics.update(
                {
                    "decision": "unsupported_result",
                    "reason": "helper_returned_non_dict",
                    "helper_result_type": type(safe_rollout_result).__name__,
                }
            )

    safe_rollout_diagnostics["production_policy_value"] = production_policy_value
    safe_rollout_diagnostics["production_policy_source"] = production_policy_source
    safe_rollout_diagnostics["safe_rollout_selected_candidate_id"] = safe_rollout_diagnostics["selected_candidate_id"]

    myopic_safe_guardrail_enabled = _env_bool("MYOPIC_SAFE_GUARDRAIL_ENABLED", False)
    myopic_safe_epsilon = max(0.0, float(os.getenv("MYOPIC_SAFE_EPSILON", "0.0")))
    myopic_safe_guardrail_diagnostics = {
        "enabled": bool(myopic_safe_guardrail_enabled),
        "epsilon": float(myopic_safe_epsilon),
        "decision": "disabled" if not myopic_safe_guardrail_enabled else "pending",
        "reason": "disabled" if not myopic_safe_guardrail_enabled else None,
        "production_policy_value_before": float(production_policy_value),
        "production_policy_source_before": production_policy_source,
        "myopic_policy_value": None,
        "delta_vs_myopic": None,
    }
    myopic_policy_value = None

    if myopic_safe_guardrail_enabled:
        try:
            myopic_eval = evaluate_myopic_policy(
                belief=dict(belief_eval),
                individuals=individuals,
                gen_states=gen_states,
                infer=infer,
                a=config.a,
                b=config.b,
                c=config.c,
                delta=config.delta,
                fixed_cost=config.fixed_cost,
                variable_cost=config.variable_cost,
                belief_gene=dict(belief_gene_eval) if isinstance(belief_gene_eval, dict) else belief_gene_eval,
                genes=config.genes if config.genes else None,
                a_gene=config.a_gene if config.a_gene else None,
                b_gene=config.b_gene if config.b_gene else None,
                c_gene=config.c_gene if config.c_gene else None,
                delta_gene=config.delta_gene if config.delta_gene else None,
                state_pool=tuple(belief_eval.keys()),
            )
            if myopic_eval.root_value is None or not math.isfinite(float(myopic_eval.root_value)):
                myopic_safe_guardrail_diagnostics.update(
                    {
                        "decision": "unavailable",
                        "reason": "missing_myopic_root_value",
                        "myopic_state_count": len(myopic_eval.policy),
                    }
                )
            else:
                myopic_policy_value = float(myopic_eval.root_value)
                delta_vs_myopic = float(production_policy_value) - myopic_policy_value
                myopic_safe_guardrail_diagnostics.update(
                    {
                        "decision": "accepted",
                        "reason": "production_policy_no_worse_than_myopic",
                        "myopic_policy_value": myopic_policy_value,
                        "delta_vs_myopic": delta_vs_myopic,
                        "myopic_state_count": len(myopic_eval.policy),
                    }
                )
                if delta_vs_myopic < -myopic_safe_epsilon:
                    production_policy_value = myopic_policy_value
                    production_policy_source = "myopic_policy"
                    myopic_safe_guardrail_diagnostics.update(
                        {
                            "decision": "fallback_to_myopic",
                            "reason": "production_policy_worse_than_myopic",
                            "production_policy_value_after": float(production_policy_value),
                            "production_policy_source_after": production_policy_source,
                        }
                    )
        except Exception as exc:
            myopic_safe_guardrail_diagnostics.update(
                {
                    "decision": "error",
                    "reason": f"myopic_eval_error:{type(exc).__name__}",
                    "error": repr(exc),
                }
            )

    oracle_policy_enabled = _env_bool("ORACLE_POLICY_ENABLED", False)
    oracle_policy_diagnostics = {
        "enabled": bool(oracle_policy_enabled),
        "decision": "disabled" if not oracle_policy_enabled else "pending",
        "reason": "disabled" if not oracle_policy_enabled else None,
        "production_policy_value_before": float(production_policy_value),
        "production_policy_source_before": production_policy_source,
        "exact_root_value": exact_root_value,
        "oracle_adp_enabled": bool(oracle_adp_enabled),
        "policy_state_count": len(policy_exact) if isinstance(policy_exact, dict) else None,
    }
    if oracle_policy_enabled:
        exact_policy_value = float(Phi_star_exact.get(frozenset(), exact_root_value))
        if oracle_adp_enabled and math.isfinite(exact_policy_value):
            production_policy_value = exact_policy_value
            production_policy_source = "oracle_exact_policy"
            oracle_policy_diagnostics.update(
                {
                    "decision": "accepted",
                    "reason": "oracle_exact_policy_ceiling",
                    "production_policy_value_after": float(production_policy_value),
                    "production_policy_source_after": production_policy_source,
                }
            )
        else:
            oracle_policy_diagnostics.update(
                {
                    "decision": "unavailable",
                    "reason": "requires_ORACLE_ADP_ENABLED_and_finite_exact_root",
                }
            )

    myopic_safe_guardrail_diagnostics.setdefault("production_policy_value_after", float(production_policy_value))
    myopic_safe_guardrail_diagnostics.setdefault("production_policy_source_after", production_policy_source)
    oracle_policy_diagnostics.setdefault("production_policy_value_after", float(production_policy_value))
    oracle_policy_diagnostics.setdefault("production_policy_source_after", production_policy_source)
    safe_rollout_diagnostics["production_policy_value"] = production_policy_value
    safe_rollout_diagnostics["production_policy_source"] = production_policy_source

    def _state_size(state):
        return len(state)

    root_state = min(Phi_star_approx.keys(), key=_state_size) if Phi_star_approx else frozenset()
    adp_root_value_phi = Phi_star_approx.get(root_state, None)
    adp_root_value_phi_source = "phi_solution"
    phi_eval_root = None
    try:
        phi_eval_root = phi_eval_approx.get(root_state) if phi_eval_approx else None
    except Exception:
        phi_eval_root = None
    if phi_eval_root is not None:
        if (adp_root_value_phi is None) or (abs(adp_root_value_phi - phi_eval_root) > 1e-6):
            if verbose:
                print(f"[warn] Φ(root) from solver={adp_root_value_phi} differs from evaluated Φ={phi_eval_root}; using evaluated value.")
            adp_root_value_phi = phi_eval_root
            adp_root_value_phi_source = "phi_eval_override"
        else:
            adp_root_value_phi_source = "phi_solution_agrees_with_eval"
    if adp_root_value_phi is None or not math.isfinite(adp_root_value_phi):
        if phi_eval_root is not None and math.isfinite(phi_eval_root):
            adp_root_value_phi = phi_eval_root
            adp_root_value_phi_source = "phi_eval_fallback"
        else:
            adp_root_value_phi = exact_root_value
            adp_root_value_phi_source = "exact_fallback"
    if exact_root_value is not None and adp_root_value_phi is not None and math.isfinite(adp_root_value_phi):
        if adp_root_value_phi > exact_root_value + 1.0:
            if verbose:
                print(f"[warn] Clamping ADP Φ(root) {adp_root_value_phi} to exact value {exact_root_value} (suspiciously large bound).")
            adp_root_value_phi = exact_root_value
            adp_root_value_phi_source = f"{adp_root_value_phi_source}|clamped_to_exact"
    adp_root_value_phi_source = f"{adp_root_value_phi_source}|candidate={selected_candidate_id}"
    safe_rollout_diagnostics["adp_root_value_phi"] = adp_root_value_phi

    # ------------------------------------------------------------------
    # 6. Enhanced Timing Summary
    # ------------------------------------------------------------------
    total_time = exact_dp_time + adp_solve_time  # ADP reuses cached inference, so no separate inference time
    print(f"\n--- Enhanced Timing Breakdown ---")
    print(f"Exact DP Total: {exact_dp_time:.3f}s ({100*exact_dp_time/total_time:.1f}%)")
    print(f"  - Exact DP Belief Building: {exact_dp_belief_time:.3f}s ({100*exact_dp_belief_time/total_time:.1f}%)")
    print(f"  - Exact DP Solve: {exact_dp_time - exact_dp_belief_time:.3f}s ({100*(exact_dp_time - exact_dp_belief_time)/total_time:.1f}%)")
    print(f"ADP Solve (using cached beliefs): {adp_solve_time:.3f}s ({100*adp_solve_time/total_time:.1f}%)")
    print(f"  - ADP should have minimal inference time due to cache reuse")
    if adp_inference_time > 0.01:  # Warn if ADP is doing significant inference
        print(f"  - WARNING: ADP inference time is {adp_inference_time:.3f}s - cache may not be working!")
    print(f"Total Time: {total_time:.3f}s")
    
    # Performance analysis
    cache_benefit = exact_dp_belief_time - adp_inference_time if adp_inference_time < exact_dp_belief_time else 0
    if cache_benefit > 0.001:
        print(f"\nCache Performance: Saved {cache_benefit:.3f}s ({100*cache_benefit/exact_dp_belief_time:.1f}%) on ADP inference")
    else:
        print(f"\nWARNING: Cache may not be providing expected performance benefits")

    adp_root_value_r_stop = master.getConstrByName("root_stop").RHS
    exact_root_value_r_stop = None
    root_stop_diff = None
    try:
        if belief_exact is not None:
            root_entry = belief_exact.get(frozenset())
            if root_entry is not None:
                if isinstance(root_entry, InferenceResult):
                    p_root = root_entry.marginals
                    per_gene_root = root_entry.get_per_gene_probs() if multi_gene else None
                else:
                    p_root = root_entry
                    per_gene_root = (
                        lift_single_gene_posteriors_to_genes(p_root, config.genes)
                        if multi_gene
                        else None
                    )

                exact_root_value_r_stop = sum(
                    r_reward(
                        person,
                        p_root,
                        config.a,
                        config.b,
                        config.c,
                        config.delta,
                        per_gene_probs=per_gene_root,
                        a_gene=config.a_gene if multi_gene else None,
                        b_gene=config.b_gene if multi_gene else None,
                        c_gene=config.c_gene if multi_gene else None,
                        delta_gene=config.delta_gene if multi_gene else None,
                    )
                    for person in individuals
                )
                root_stop_diff = adp_root_value_r_stop - exact_root_value_r_stop
    except Exception as exc:
        if verbose:
            print(f"[warn] failed to compute exact root stop value: {exc}")

    dfvr_argmax = dfvr_results.get("argmax_residual") if dfvr_results else None
    dfvr_argmax_state_size = None
    dfvr_argmax_action_kind = None
    dfvr_argmax_action_person = None
    dfvr_argmax_phi_s = None
    dfvr_argmax_best_rhs = None
    dfvr_argmax_slack = None
    if isinstance(dfvr_argmax, dict):
        state_items = dfvr_argmax.get("state")
        if isinstance(state_items, list):
            dfvr_argmax_state_size = len(state_items)
        action_payload = dfvr_argmax.get("action")
        if isinstance(action_payload, dict):
            dfvr_argmax_action_kind = action_payload.get("kind")
            dfvr_argmax_action_person = action_payload.get("person")
        dfvr_argmax_phi_s = dfvr_argmax.get("phi_s")
        dfvr_argmax_best_rhs = dfvr_argmax.get("bellman_rhs")
        dfvr_argmax_slack = dfvr_argmax.get("residual")

    tail_regularization = getattr(adp_cache, "tail_regularization", None) if adp_cache else None
    solver_root_diagnostics = getattr(adp_cache, "root_diagnostics", None) if adp_cache else None
    root_diagnostics = dict(solver_root_diagnostics) if isinstance(solver_root_diagnostics, dict) else {}

    root_phi_lp = None
    if isinstance(Phi_star_approx, dict):
        root_phi_lp = Phi_star_approx.get(root_state)
    if root_phi_lp is not None and not math.isfinite(root_phi_lp):
        root_phi_lp = None
    if phi_eval_root is not None and not math.isfinite(phi_eval_root):
        phi_eval_root = None

    gap1_lp = None
    if (
        root_phi_lp is not None
        and exact_root_value is not None
        and math.isfinite(root_phi_lp)
        and math.isfinite(exact_root_value)
    ):
        gap1_lp = root_phi_lp - exact_root_value
    gap1_eval = None
    if (
        phi_eval_root is not None
        and exact_root_value is not None
        and math.isfinite(phi_eval_root)
        and math.isfinite(exact_root_value)
    ):
        gap1_eval = phi_eval_root - exact_root_value
    gap1_selected = None
    if (
        adp_root_value_phi is not None
        and exact_root_value is not None
        and math.isfinite(adp_root_value_phi)
        and math.isfinite(exact_root_value)
    ):
        gap1_selected = adp_root_value_phi - exact_root_value

    root_diagnostics.update(
        {
            "phi_root_lp": root_phi_lp,
            "phi_root_eval": phi_eval_root,
            "phi_root_selected": adp_root_value_phi,
            "phi_root_source": adp_root_value_phi_source,
            "exact_root_value": exact_root_value,
            "policy_root_value": approximate_root_value,
            "root_stop_rhs": adp_root_value_r_stop,
            "root_stop_rhs_exact": exact_root_value_r_stop,
            "root_stop_diff": root_stop_diff,
            "gap1_lp": gap1_lp,
            "gap1_eval": gap1_eval,
            "gap1_selected": gap1_selected,
            "polish_acceptance_decision": polish_acceptance_decision,
            "polish_acceptance_reason": polish_acceptance_reason,
            "selected_candidate_id": selected_candidate_id,
            "candidate_guardrail_metrics": polish_candidate_metrics,
            "production_policy_value": production_policy_value,
            "production_policy_source": production_policy_source,
            "myopic_policy_value": myopic_policy_value,
            "myopic_safe_guardrail_enabled": myopic_safe_guardrail_diagnostics.get("enabled"),
            "myopic_safe_guardrail_decision": myopic_safe_guardrail_diagnostics.get("decision"),
            "myopic_safe_guardrail_reason": myopic_safe_guardrail_diagnostics.get("reason"),
            "myopic_safe_guardrail_diagnostics": myopic_safe_guardrail_diagnostics,
            "oracle_adp_enabled": bool(oracle_adp_enabled),
            "oracle_adp": root_diagnostics.get("oracle_adp"),
            "oracle_policy_enabled": oracle_policy_diagnostics.get("enabled"),
            "oracle_policy_decision": oracle_policy_diagnostics.get("decision"),
            "oracle_policy_reason": oracle_policy_diagnostics.get("reason"),
            "oracle_policy_diagnostics": oracle_policy_diagnostics,
            "safe_rollout_enabled": safe_rollout_diagnostics.get("enabled"),
            "safe_rollout_available": safe_rollout_diagnostics.get("available"),
            "safe_rollout_top_k": safe_rollout_diagnostics.get("top_k"),
            "safe_rollout_incumbent_safe": safe_rollout_diagnostics.get("incumbent_safe"),
            "safe_rollout_decision": safe_rollout_diagnostics.get("decision"),
            "safe_rollout_reason": safe_rollout_diagnostics.get("reason"),
            "safe_rollout_selected_candidate_id": safe_rollout_diagnostics.get("safe_rollout_selected_candidate_id"),
            "safe_rollout_diagnostics": safe_rollout_diagnostics,
            "rowgen_telemetry": rowgen_telemetry,
            "runtime_telemetry": runtime_telemetry,
        }
    )
    if isinstance(root_diagnostics.get("oracle_adp"), dict):
        root_diagnostics["oracle_adp"]["policy_source"] = production_policy_source

    results = {
        "ADP_root_value_phi": adp_root_value_phi,
        "ADP_root_value_R_stop": adp_root_value_r_stop,
        "Exact_root_value_R_stop": exact_root_value_r_stop,
        "root_stop_diff": root_stop_diff,
        "ADP_policy_value": approximate_root_value,
        "production_policy_value": production_policy_value,
        "production_policy_source": production_policy_source,
        "myopic_policy_value": myopic_policy_value,
        "myopic_safe_guardrail_enabled": myopic_safe_guardrail_diagnostics.get("enabled"),
        "myopic_safe_guardrail_decision": myopic_safe_guardrail_diagnostics.get("decision"),
        "myopic_safe_guardrail_reason": myopic_safe_guardrail_diagnostics.get("reason"),
        "myopic_safe_guardrail_diagnostics": myopic_safe_guardrail_diagnostics,
        "oracle_adp_enabled": bool(oracle_adp_enabled),
        "oracle_adp": root_diagnostics.get("oracle_adp"),
        "oracle_policy_enabled": oracle_policy_diagnostics.get("enabled"),
        "oracle_policy_decision": oracle_policy_diagnostics.get("decision"),
        "oracle_policy_reason": oracle_policy_diagnostics.get("reason"),
        "oracle_policy_diagnostics": oracle_policy_diagnostics,
        "safe_rollout_enabled": safe_rollout_diagnostics.get("enabled"),
        "safe_rollout_available": safe_rollout_diagnostics.get("available"),
        "safe_rollout_top_k": safe_rollout_diagnostics.get("top_k"),
        "safe_rollout_incumbent_safe": safe_rollout_diagnostics.get("incumbent_safe"),
        "safe_rollout_decision": safe_rollout_diagnostics.get("decision"),
        "safe_rollout_reason": safe_rollout_diagnostics.get("reason"),
        "safe_rollout_selected_candidate_id": safe_rollout_diagnostics.get("safe_rollout_selected_candidate_id"),
        "safe_rollout_diagnostics": safe_rollout_diagnostics,
        "Exact_DP_root_value": exact_root_value,
        "Approximation Error": _safe_ratio_or_none(
            exact_root_value - approximate_root_value,
            abs(exact_root_value),
            eps=denom_small_eps,
        ),
        "optimality_gap": _safe_ratio_or_none(
            exact_root_value - approximate_root_value,
            exact_root_value - adp_root_value_r_stop,
            eps=denom_small_eps,
        ),
        "policy_differences": diff_count,
        "common_states": common_states,
        "policy_loss": avg_policy_loss,
        "policy_report_depth": report_depth,
        "policy_report_depth_best": best_stats.get("depth"),
        "ADP_policy_value_best": best_stats.get("root_value"),
        # Enhanced timing breakdown
        "exact_dp_total_time": exact_dp_time,
        "exact_dp_belief_time": exact_dp_belief_time,
        "exact_dp_solve_time": exact_dp_time - exact_dp_belief_time,
        "exact_dp_runtime_tracking": exact_dp_runtime_tracking,
        "belief_map_construction_time_sec": belief_map_tracking.get("elapsed_sec"),
        "belief_map_progress_status": belief_map_tracking.get("progress_status"),
        "belief_map_build_mode": belief_map_tracking.get("mode"),
        "belief_map_state_count": belief_map_tracking.get("state_count"),
        "belief_map_total_indexed_state_count": belief_map_tracking.get("total_indexed_state_count"),
        "belief_map_processed_state_count": belief_map_tracking.get("processed_state_count"),
        "belief_map_generated_successor_count": belief_map_tracking.get("generated_successor_count"),
        "exact_dual_progress_status": exact_dual_tracking.get("progress_status"),
        "exact_dual_lp_backend": exact_dual_tracking.get("backend"),
        "exact_dual_lp_status": exact_dual_tracking.get("status"),
        "exact_dual_lp_variable_count": exact_dual_tracking.get("lp_variable_count"),
        "exact_dual_lp_constraint_count": exact_dual_tracking.get("lp_constraint_count"),
        "exact_dual_lp_build_time_sec": exact_dual_tracking.get("lp_build_elapsed_sec"),
        "exact_dual_lp_solve_time_sec": exact_dual_tracking.get("lp_solve_elapsed_sec"),
        "exact_dual_lp_total_time_sec": exact_dual_tracking.get("total_elapsed_sec"),
        "exact_dual_lp_log_path": exact_dual_tracking.get("log_path"),
        "adp_solve_time": adp_solve_time,
        "adp_inference_time": adp_inference_time,  # Should be minimal due to cache
        "total_time": total_time,
        "cache_states_precomputed": len(precomputed_beliefs) if precomputed_beliefs is not None else 0,
        # Enhanced timing format for comprehensive test compatibility
        "timing": {
            "bayesian_setup": bayesian_setup_time,
            "adp_inference": adp_inference_time,
            "adp_solve": adp_solve_time,
            "adp_total": adp_solve_time + adp_inference_time,
            "exact_dp_belief_build": exact_dp_belief_time,
            "exact_dp_solve": exact_dp_time - exact_dp_belief_time,
            "exact_dp_total": exact_dp_time,
            "belief_map_construction": belief_map_tracking.get("elapsed_sec"),
            "exact_dual_lp_build": exact_dual_tracking.get("lp_build_elapsed_sec"),
            "exact_dual_lp_solve": exact_dual_tracking.get("lp_solve_elapsed_sec"),
            "exact_dual_lp_total": exact_dual_tracking.get("total_elapsed_sec"),
        },
        "policy_metrics": policy_metrics,
        "tuple_mode_active": tuple_mode_active,
        "tuple_mode_enabled": tuple_mode_enabled,
        "tuple_posteriors_count": len(tuple_posteriors),
        "tuple_posteriors": tuple_posteriors,
        "Phi_star_exact": Phi_star_exact,
        "Phi_star_exact_gene": Phi_star_exact_gene,
        "belief_exact": belief_exact,
        "policy_exact": policy_exact,
        "belief_post_approx": belief_post_approx,
        "W_star": W_star,
        "W_star_gene": w_gene_star,
        "theta_star": theta_star,
        "theta_model": theta_model_info.get("theta_model"),
        "theta_model_spec_path": theta_model_info.get("theta_model_spec_path"),
        "theta_model_signature": theta_model_info.get("theta_model_signature"),
        "theta_model_spec": theta_model_info.get("theta_model_spec"),
        "Phi_star_approx": Phi_star_approx,
        "Phi_star_approx_eval": Phi_star_approx.get(root_state) if Phi_star_approx else None,
        "Phi_eval_root": phi_eval_root,
        "root_phi_lp": root_phi_lp,
        "root_phi_eval": phi_eval_root,
        "root_phi_selected_source": adp_root_value_phi_source,
        "root_gap1_lp": gap1_lp,
        "root_gap1_eval": gap1_eval,
        "root_gap1_selected": gap1_selected,
        "selected_candidate_id": selected_candidate_id,
        "polish_acceptance_decision": polish_acceptance_decision,
        "polish_acceptance_reason": polish_acceptance_reason,
        "candidate_guardrail_metrics": polish_candidate_metrics,
        "ratio2_tolerances": {
            "RATIO2_DELTA_TOL": ratio2_delta_tol,
            "RATIO2_MEAN_DELTA_TOL": ratio2_mean_delta_tol,
            "GAP2_DELTA_TOL": gap2_delta_tol,
            "DENOM_SMALL_EPS": denom_small_eps,
        },
        "candidate_pre_polish": candidate_pre_polish,
        "candidate_post_polish": candidate_post_polish,
        "rowgen_telemetry": rowgen_telemetry,
        "runtime_telemetry": runtime_telemetry,
        "root_diagnostics": root_diagnostics,
        "dfvr_bound": dfvr_results.get("dfvr_bound") if dfvr_results else None,
        "dfvr_beta_rho": dfvr_results.get("beta_rho") if dfvr_results else None,
        "dfvr_residual_norm": dfvr_results.get("residual_norm") if dfvr_results else None,
        "dfvr_coverage": dfvr_results.get("coverage") if dfvr_results else None,
        "dfvr_fixed_state_integrity": dfvr_results.get("fixed_state_integrity") if dfvr_results else None,
        "dfvr_state_signature": (
            (dfvr_results.get("coverage") or {}).get("state_signature")
            if dfvr_results
            else None
        ),
        "dfvr_eval_no_mutation": bool(dfvr_eval_no_mutation) if dfvr_bound else None,
        "dfvr_fixed_states_path": (
            str(dfvr_fixed_states_path_resolved) if dfvr_fixed_states_path_resolved else None
        ),
        "dfvr_fixed_states_loaded": dfvr_fixed_states_loaded,
        "dfvr_fixed_states_written": dfvr_fixed_states_written,
        "dfvr_argmax_state_size": dfvr_argmax_state_size,
        "dfvr_argmax_action_kind": dfvr_argmax_action_kind,
        "dfvr_argmax_action_person": dfvr_argmax_action_person,
        "dfvr_argmax_phi_s": dfvr_argmax_phi_s,
        "dfvr_argmax_best_rhs": dfvr_argmax_best_rhs,
        "dfvr_argmax_slack": dfvr_argmax_slack,
        "dfvr_details": dfvr_results if dfvr_results else None,
        "tail_regularization": tail_regularization,
        "aaub": aaub_star,
        "edge_star": edge_star,
        "trio_star": trio_star,
        "myopic_adp": root_diagnostics.get("myopic_adp"),
        "oracle_adp_star": oracle_adp_star,
        "slack_refactor": slack_refactor,
    }
    if adp_stats is not None:
        results["adp_stats"] = adp_stats
    if return_infer:
        results["infer"] = infer
    return results
