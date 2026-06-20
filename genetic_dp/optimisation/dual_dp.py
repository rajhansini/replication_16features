import gurobipy as gp
import os
import json
import time
import math
import hashlib
from itertools import product
from gurobipy import GRB
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
try:
    import resource  # Unix-only, used for RSS watchdog.
except Exception:  # pragma: no cover - platform-dependent
    resource = None  # type: ignore[assignment]
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
from ..models.belief import (
    propagate_all_marginals,
    lift_single_gene_posteriors_to_genes,
    propagate_multigene_marginals,
    InferenceResult,
)
from ..models.reward import r_reward, r_reward_testp
from ..models.outcomes import project_state_by_gene, project_successor_by_gene
from ..optimisation.bellman_rowgen import BellmanRowGenerator, log_bellman_cut, select_successors, select_tuple_successors, bellman_violation
from ..optimisation.state_features import build_state_feature_bundle, iter_trio_feature_rows
from ..optimisation.theta_model import (
    evaluate_theta_model_correction,
    resolve_theta_mode,
    theta_model_metadata,
)
from ..optimisation.myopic_adp import (
    ABCD16_DIRECT_CANONICAL14_FEATURES,
    ABCD16_DIRECT_CANONICAL14_MYOPIC_EMBEDDING_ORDER,
    ABCD16_DIRECT_CANONICAL14_MYOPIC_FEATURES,
    ABCD16_DIRECT_CANONICAL14_REGIME_FEATURES,
    ABCD16_DIRECT_FEATURES,
    ABCD16_DIRECT_MYOPIC_EMBEDDING_ORDER,
    ABCD16_DIRECT_MYOPIC_FEATURES,
    ABCD16_DIRECT_REGIME_EMBEDDING_ORDER,
    ABCD16_DIRECT_REGIME_FEATURES,
    FEATURE_BANK_ABCD_HAND,
    build_state_features,
    feature_semantics_for_bank,
    regime_parameter_gates,
    regime_residual_candidate_features,
    regime_residual_feature_values,
    regime_residual_term_value,
    regime_residual_v2_candidate_features,
    resolve_feature_bank,
    resolve_feature_semantics,
    select_signature_features,
    serializable_summary,
)
from ..optimisation.oracle_adp import (
    build_oracle_feature_payload,
    oracle_adp_term_value,
    oracle_feature_values,
    serializable_summary as oracle_serializable_summary,
)
from ..policy.myopic import evaluate_myopic_policy

from ..optimisation.caches import InferenceCache, CutManager
from ..optimisation.utils import (
    compute_phi,
    canonicalize_state,
    reward_signature_fn,
    validate_role_groups,
    discover_role_groups,
)



# --------------------------
# Tunables (can be env-driven)
# --------------------------
MAX_STATES_PER_ITER = int(os.getenv("MAX_STATES_PER_ITER", "200"))
MAX_CUTS_PER_ITER   = int(os.getenv("MAX_CUTS_PER_ITER", "500"))
TOPK_SUCCESSORS     = int(os.getenv("TOPK_SUCCESSORS", "2"))
PMIN_SUCCESSOR      = float(os.getenv("PMIN_SUCCESSOR", "1e-3"))
GAP_TOL             = float(os.getenv("GAP_TOL", "1e-9"))

def solve_dual_dp_with_domain(
        I, gen_states,
        mu0,                    # {state: prob}
        a, b, c, delta,            # dicts over I
        x, allele_freq, child_cpds, pedigree, # domain data
        p0, z0,                 # nested‑dict priors
        infer,                        # infer
        p0_gene=None,
        genes=None,
        a_gene=None,
        b_gene=None,
        c_gene=None,
        delta_gene=None,
        role_groups=None,             # exchangeable cohorts (probability-only)
        value_canon_mode: str = 'identity',  # 'identity' (default), 'role', 'cohort' (optional)
        max_iters=200, tol=1e-6, verbose=False,
        debug_lp_path=None,      # e.g. "master_debug.lp"
        fixed_cost=0.01, variable_cost=0.02,
        cut_validator=None,       # optional hook: fn(kind, state, rhs_const, successors)
        return_stats: bool = False,
        return_phi_eval: bool = False,
        analysis_mode: str | None = None,
        oracle_adp_payload=None,
        precomputed_beliefs=None, # Optional: {state: belief_dict} from exact DP
        cache_diagnostics: bool = False): # Enable detailed cache diagnostics
    # Always use Bellman-consistent row generation
    use_bellman_rowgen = True
    if analysis_mode not in {None, "seeded", "first_pass"}:
        raise ValueError(
            f"Unknown analysis_mode={analysis_mode!r} "
            "(expected None, 'seeded', or 'first_pass')."
        )

    # --- Master Dual LP ---
    master = gp.Model("dual_DP")
    master.Params.OutputFlag = 0 # Set to 1 for Gurobi output
    master.Params.Method = 1        # dual simplex = fast reopt
    master.Params.Presolve = 1      # light presolve; try 2 if helpful
    def _apply_adp_gurobi_env_params() -> None:
        param_specs = (
            ("ADP_GUROBI_METHOD", "Method", int),
            ("ADP_GUROBI_PRESOLVE", "Presolve", int),
            ("ADP_GUROBI_CROSSOVER", "Crossover", int),
            ("ADP_GUROBI_NUMERIC_FOCUS", "NumericFocus", int),
            ("ADP_GUROBI_FEASIBILITY_TOL", "FeasibilityTol", float),
            ("ADP_GUROBI_OPTIMALITY_TOL", "OptimalityTol", float),
        )
        for env_name, param_name, caster in param_specs:
            raw = os.getenv(env_name, "").strip()
            if raw:
                setattr(master.Params, param_name, caster(raw))

    _apply_adp_gurobi_env_params()

    gene_list = tuple(genes) if genes else tuple()
    multi_gene = bool(gene_list)
    env_flag = os.getenv("ENABLE_TUPLE_ROWGEN")
    per_gene_phi_env_flag = os.getenv("ENABLE_PER_GENE_PHI", "1")
    per_gene_phi_enabled = per_gene_phi_env_flag.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if multi_gene:
        tuple_mode = env_flag != "0"
    else:
        tuple_mode = False
    tuple_successor_K = None if tuple_mode else TOPK_SUCCESSORS
    # Per-gene Φ/W is the only supported multigene mode.
    per_gene_phi_active = multi_gene and tuple_mode and per_gene_phi_enabled
    theta_model_info = theta_model_metadata()
    theta_model = theta_model_info.get("theta_model")
    theta_mode = resolve_theta_mode(os.getenv("THETA_MODE", "scalar"), theta_model)
    if theta_mode not in {"scalar", "stage", "person", "person_stage", "stage_gene"}:
        raise ValueError(
            "Unknown THETA_MODE="
            f"{theta_mode!r} (expected 'scalar', 'stage', 'person', 'person_stage', or 'stage_gene')."
        )
    stage_theta_active = theta_mode == "stage"
    person_theta_active = theta_mode == "person"
    person_stage_theta_active = theta_mode == "person_stage"
    stage_gene_theta_active = theta_mode == "stage_gene"
    if stage_gene_theta_active:
        if not multi_gene:
            raise ValueError("THETA_MODE='stage_gene' requires multi-gene config (genes must be configured).")
        if not tuple_mode:
            raise ValueError("THETA_MODE='stage_gene' requires ENABLE_TUPLE_ROWGEN=1.")
        if not per_gene_phi_enabled:
            raise ValueError("THETA_MODE='stage_gene' requires ENABLE_PER_GENE_PHI=1.")

    def _resolve_candidate_modes(raw_value: str, default_mode: str = "stage"):
        raw = raw_value.strip().lower()
        if raw == "*":
            return {"scalar", "stage", "person", "person_stage", "stage_gene"}
        modes = {mode.strip() for mode in raw.split(",") if mode.strip()}
        if not modes:
            modes = {default_mode}
        return modes

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}

    def _env_with_fallback(primary: str, fallback: str, default: str) -> str:
        raw = os.getenv(primary, "").strip()
        if raw:
            return raw
        return os.getenv(fallback, default)

    tail_phi_reg_enable = os.getenv("TAIL_PHI_REG_ENABLE", "0") == "1"
    tail_phi_reg_modes = _resolve_candidate_modes(os.getenv("TAIL_PHI_REG_THETA_MODES", "stage"))
    tail_phi_reg_active = tail_phi_reg_enable and theta_mode in tail_phi_reg_modes
    tail_phi_reg_weight = float(os.getenv("TAIL_PHI_REG_WEIGHT", "1e-5"))
    tail_phi_reg_max_root_delta = max(0.0, float(os.getenv("TAIL_PHI_REG_MAX_ROOT_DELTA", "0.0")))
    tail_phi_reg_max_states = int(os.getenv("TAIL_PHI_REG_MAX_STATES", "5000"))
    min_stage_default = max(1, len(I) - 1)
    tail_phi_reg_min_stage = int(os.getenv("TAIL_PHI_REG_MIN_STAGE", str(min_stage_default)))
    if tail_phi_reg_min_stage < 1:
        tail_phi_reg_min_stage = 1
    if tail_phi_reg_min_stage > len(I):
        tail_phi_reg_min_stage = len(I)

    hotspot_minmax_enable = _env_bool("HOTSPOT_MINMAX_ENABLE", False)
    slack_polish_enable = os.getenv("SLACK_POLISH_ENABLE", "0") == "1" or hotspot_minmax_enable
    slack_polish_modes = _resolve_candidate_modes(os.getenv("SLACK_POLISH_THETA_MODES", "stage"))
    slack_polish_active = slack_polish_enable and theta_mode in slack_polish_modes
    slack_polish_topk = max(1, int(_env_with_fallback("HOTSPOT_TOPK", "SLACK_POLISH_TOPK", "16")))
    slack_polish_rounds = max(1, int(_env_with_fallback("HOTSPOT_ROUNDS", "SLACK_POLISH_ROUNDS", "2")))
    slack_polish_max_root_delta = max(
        0.0,
        float(_env_with_fallback("HOTSPOT_MAX_ROOT_DELTA", "SLACK_POLISH_MAX_ROOT_DELTA", "0.0")),
    )
    slack_polish_phi_delta_max_raw = _env_with_fallback(
        "HOTSPOT_PHI_DELTA_MAX",
        "SLACK_POLISH_PHI_DELTA_MAX",
        "",
    ).strip()
    slack_polish_phi_delta_max = (
        float(slack_polish_phi_delta_max_raw)
        if slack_polish_phi_delta_max_raw
        else None
    )
    if slack_polish_phi_delta_max is not None and slack_polish_phi_delta_max < 0.0:
        slack_polish_phi_delta_max = None

    aaub_enable = os.getenv("AAUB_ENABLE", "0") == "1"
    aaub_modes = _resolve_candidate_modes(os.getenv("AAUB_THETA_MODES", "stage"))
    aaub_active = aaub_enable and theta_mode in aaub_modes
    aaub_fixed_enable = os.getenv("AAUB_FIXED_ENABLE", "1") == "1"
    aaub_p12_enable = os.getenv("AAUB_P12_ENABLE", "0") == "1"
    aaub_sign_safe_mode = _env_bool("AAUB_SIGN_SAFE_MODE", aaub_enable)
    aaub_max_root_delta = max(0.0, float(os.getenv("AAUB_MAX_ROOT_DELTA", "0.0")))
    aaub_coef_abs_cap = max(0.0, float(os.getenv("AAUB_COEF_ABS_MAX", "100.0")))
    aaub_apply = aaub_active and (aaub_fixed_enable or aaub_p12_enable)
    if (aaub_apply or slack_polish_active) and tail_phi_reg_active:
        tail_phi_reg_active = False

    abcd16_direct_active = _env_bool("ABCD16_DIRECT_ENABLED", False)
    abcd16_direct_selection = os.getenv("ABCD16_DIRECT_SELECTION", "fixed_all_16").strip()
    if abcd16_direct_active and abcd16_direct_selection not in {"fixed_all_16", "fixed_canonical_14"}:
        raise ValueError(
            "ABCD16_DIRECT_SELECTION must be 'fixed_all_16' or 'fixed_canonical_14' when "
            "ABCD16_DIRECT_ENABLED=1."
        )
    if abcd16_direct_selection == "fixed_canonical_14":
        abcd16_direct_features = ABCD16_DIRECT_CANONICAL14_FEATURES
        abcd16_direct_myopic_features = ABCD16_DIRECT_CANONICAL14_MYOPIC_FEATURES
        abcd16_direct_myopic_embedding_order = ABCD16_DIRECT_CANONICAL14_MYOPIC_EMBEDDING_ORDER
        abcd16_direct_regime_features = ABCD16_DIRECT_CANONICAL14_REGIME_FEATURES
    else:
        abcd16_direct_features = ABCD16_DIRECT_FEATURES
        abcd16_direct_myopic_features = ABCD16_DIRECT_MYOPIC_FEATURES
        abcd16_direct_myopic_embedding_order = ABCD16_DIRECT_MYOPIC_EMBEDDING_ORDER
        abcd16_direct_regime_features = ABCD16_DIRECT_REGIME_FEATURES

    myopic_control_variate_active = _env_bool("MYOPIC_CONTROL_VARIATE_ENABLED", False)
    myopic_residual_basis_active = _env_bool("MYOPIC_RESIDUAL_BASIS_ENABLED", False)
    myopic_piecewise_regions_active = _env_bool("MYOPIC_PIECEWISE_REGIONS_ENABLED", False)
    myopic_direction_count = sum(
        bool(flag)
        for flag in (
            myopic_control_variate_active,
            myopic_residual_basis_active,
            myopic_piecewise_regions_active,
            abcd16_direct_active,
        )
    )
    if myopic_direction_count > 1:
        raise ValueError(
            "Enable only one myopic-ADP direction at a time: "
            "MYOPIC_CONTROL_VARIATE_ENABLED, MYOPIC_RESIDUAL_BASIS_ENABLED, "
            "MYOPIC_PIECEWISE_REGIONS_ENABLED, or ABCD16_DIRECT_ENABLED."
        )
    myopic_adp_active = myopic_direction_count == 1
    myopic_residual_top_k = max(1, int(os.getenv("MYOPIC_RESIDUAL_BASIS_TOP_K", "6")))
    myopic_residual_extra_features = tuple(
        feature.strip()
        for feature in os.getenv("MYOPIC_RESIDUAL_BASIS_EXTRA_FEATURES", "").split(",")
        if feature.strip()
    )
    myopic_residual_force_features = tuple(
        feature.strip()
        for feature in os.getenv("MYOPIC_RESIDUAL_BASIS_FORCE_FEATURES", "").split(",")
        if feature.strip()
    )
    myopic_residual_only_extra = _env_bool("MYOPIC_RESIDUAL_BASIS_ONLY_EXTRA", False)
    regime_residual_v2_active = _env_bool("GAUGED_REGIME_RESIDUAL_V2_ENABLED", False)
    regime_residual_selector = os.getenv("GAUGED_REGIME_RESIDUAL_SELECTOR", "root_test").strip().lower()
    regime_residual_anchor = os.getenv("GAUGED_REGIME_RESIDUAL_ANCHOR", "v1").strip().lower()
    regime_residual_v2_payload_path = os.getenv("GAUGED_REGIME_RESIDUAL_V2_PAYLOAD_PATH", "").strip()
    regime_residual_active = (
        _env_bool("GAUGED_REGIME_RESIDUAL_ADP_ENABLED", False)
        or regime_residual_v2_active
        or abcd16_direct_active
    )
    if abcd16_direct_active:
        regime_feature_bank = FEATURE_BANK_ABCD_HAND
    elif regime_residual_active:
        regime_feature_bank = resolve_feature_bank(require=True)
    else:
        regime_feature_bank = resolve_feature_bank("FB0_PROXY")
    regime_feature_semantics = feature_semantics_for_bank(regime_feature_bank)
    legacy_feature_semantics_raw = os.getenv("GAUGED_REGIME_FEATURE_SEMANTICS", "").strip()
    if regime_residual_active and legacy_feature_semantics_raw:
        legacy_feature_semantics = resolve_feature_semantics(legacy_feature_semantics_raw)
        if legacy_feature_semantics != regime_feature_semantics:
            raise ValueError(
                "GAUGED_REGIME_FEATURE_SEMANTICS="
                f"{legacy_feature_semantics!r} does not match GAUGED_REGIME_FEATURE_BANK="
                f"{regime_feature_bank!r} ({regime_feature_semantics!r})."
            )
    regime_residual_top_k = max(1, int(os.getenv("GAUGED_REGIME_RESIDUAL_TOP_K", "5")))
    regime_residual_min_signature_ratio = max(
        0.0,
        float(os.getenv("GAUGED_REGIME_RESIDUAL_MIN_SIGNATURE_RATIO", "0.20")),
    )
    regime_residual_incremental_tol = max(
        0.0,
        float(os.getenv("GAUGED_REGIME_RESIDUAL_INCREMENTAL_TOL", "1e-8")),
    )
    regime_residual_v2_top_k = max(1, int(os.getenv("GAUGED_REGIME_RESIDUAL_V2_TOP_K", "5")))
    regime_residual_v2_min_signature_ratio = max(
        0.0,
        float(os.getenv("GAUGED_REGIME_RESIDUAL_V2_MIN_SIGNATURE_RATIO", "0.10")),
    )
    regime_residual_v2_incremental_tol = max(
        0.0,
        float(os.getenv("GAUGED_REGIME_RESIDUAL_V2_INCREMENTAL_TOL", "1e-8")),
    )
    disable_truncated_tuple_strengthening = _env_bool("DISABLE_TRUNCATED_TUPLE_STRENGTHENING", False)

    # ── Edge features ──────────────────────────────────────────────
    edge_features_enable = os.getenv("ENABLE_EDGE_FEATURES", "0") == "1"
    pedigree_edges = []
    if pedigree:
        graph_obj = getattr(pedigree, "graph", None)
        if graph_obj is not None and hasattr(graph_obj, "edges"):
            pedigree_edges = list(graph_obj.edges())
        elif hasattr(pedigree, "get_parents"):
            seen_edges = set()
            for child in I:
                parents = pedigree.get_parents(child) or ()
                for parent in parents:
                    edge = (parent, child)
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    pedigree_edges.append(edge)
    edge_features_active = edge_features_enable and len(pedigree_edges) > 0
    edge_feature_mode = os.getenv("EDGE_FEATURE_MODE", "raw").strip().lower() or "raw"
    if edge_feature_mode not in {"raw", "residual", "hybrid"}:
        raise ValueError(
            "Unknown EDGE_FEATURE_MODE="
            f"{edge_feature_mode!r} (expected 'raw', 'residual', or 'hybrid')."
        )
    edge_feature_blocks = ("raw", "residual") if edge_feature_mode == "hybrid" else (edge_feature_mode,)
    edge_seed_scope = os.getenv("EDGE_SEED_SCOPE", "root").strip().lower() or "root"
    if edge_seed_scope not in {"root", "root_plus_stage1"}:
        raise ValueError(
            "Unknown EDGE_SEED_SCOPE="
            f"{edge_seed_scope!r} (expected 'root' or 'root_plus_stage1')."
        )
    secondary_phi_objective = os.getenv("SECONDARY_PHI_OBJECTIVE", "off").strip().lower() or "off"
    if secondary_phi_objective not in {"off", "stage12_mean"}:
        raise ValueError(
            "Unknown SECONDARY_PHI_OBJECTIVE="
            f"{secondary_phi_objective!r} (expected 'off' or 'stage12_mean')."
        )
    secondary_phi_root_tol = max(0.0, float(os.getenv("SECONDARY_PHI_OBJECTIVE_ROOT_TOL", "1e-8")))
    edge_diagnostics_enable = _env_bool("EDGE_DIAGNOSTICS", edge_features_active)

    def _pedigree_trios():
        if not pedigree or not hasattr(pedigree, "get_parents"):
            return []
        trios = []
        seen = set()
        for child in I:
            parents = tuple(sorted(pedigree.get_parents(child) or ()))
            if len(parents) != 2:
                continue
            trio = (parents[0], parents[1], child)
            if trio in seen:
                continue
            seen.add(trio)
            trios.append(trio)
        return trios

    def _pedigree_depths():
        memo = {}

        def _depth(node):
            if node in memo:
                return memo[node]
            parents = tuple(pedigree.get_parents(node) or ()) if pedigree and hasattr(pedigree, "get_parents") else ()
            if not parents:
                memo[node] = 0
                return 0
            memo[node] = 1 + max(_depth(parent) for parent in parents)
            return memo[node]

        return {person: _depth(person) for person in I}

    pedigree_trios = _pedigree_trios()
    pedigree_depths = _pedigree_depths() if pedigree_trios else {}
    trio_features_enable = os.getenv("ENABLE_TRIO_FEATURES", "0") == "1"
    trio_features_active = trio_features_enable and len(pedigree_trios) > 0
    trio_feature_mode = os.getenv("TRIO_FEATURE_MODE", "raw").strip().lower() or "raw"
    if trio_feature_mode not in {"raw", "pure3"}:
        raise ValueError(
            "Unknown TRIO_FEATURE_MODE="
            f"{trio_feature_mode!r} (expected 'raw' or 'pure3')."
        )
    trio_feature_blocks = (trio_feature_mode,)
    trio_coef_sharing = os.getenv("TRIO_COEF_SHARING", "free").strip().lower() or "free"
    if trio_coef_sharing not in {"free", "child_depth"}:
        raise ValueError(
            "Unknown TRIO_COEF_SHARING="
            f"{trio_coef_sharing!r} (expected 'free' or 'child_depth')."
        )
    trio_seed_scope = os.getenv("TRIO_SEED_SCOPE", edge_seed_scope).strip().lower() or edge_seed_scope
    if trio_seed_scope not in {"root", "root_plus_stage1", "root_plus_stage1_clinical_stage2"}:
        raise ValueError(
            "Unknown TRIO_SEED_SCOPE="
            f"{trio_seed_scope!r} (expected 'root', 'root_plus_stage1', or "
            "'root_plus_stage1_clinical_stage2')."
        )
    trio_diagnostics_enable = _env_bool("TRIO_DIAGNOSTICS", trio_features_active)

    def _trio_share_key(trio):
        if trio_coef_sharing == "child_depth":
            return int(pedigree_depths.get(trio[2], 0))
        return trio

    trio_share_key_by_trio = {trio: _trio_share_key(trio) for trio in pedigree_trios}
    trio_share_keys = []
    seen_trio_share_keys = set()
    for trio in pedigree_trios:
        share_key = trio_share_key_by_trio[trio]
        if share_key in seen_trio_share_keys:
            continue
        seen_trio_share_keys.add(share_key)
        trio_share_keys.append(share_key)

    def _clinical_priority_order():
        if not pedigree or not hasattr(pedigree, "get_founders"):
            return list(I)
        founders = set(pedigree.get_founders())
        non_founders = sorted(
            [person for person in I if person not in founders],
            key=lambda person: (-pedigree_depths.get(person, 0), str(person)),
        )
        founder_list = sorted(
            [person for person in I if person in founders],
            key=lambda person: (-pedigree_depths.get(person, 0), str(person)),
        )
        return non_founders + founder_list

    clinical_priority_order = _clinical_priority_order()

    def _clinical_frontier_person(state):
        tested_people = {person for person, _ in _evidence(state)}
        for person in clinical_priority_order:
            if person not in tested_people:
                return person
        return None

    seed_stage1_enabled = (
        edge_seed_scope == "root_plus_stage1"
        or trio_seed_scope in {"root_plus_stage1", "root_plus_stage1_clinical_stage2"}
    )
    seed_clinical_stage2_enabled = trio_seed_scope == "root_plus_stage1_clinical_stage2"
    effective_seed_scope = (
        "root_plus_stage1_clinical_stage2"
        if seed_clinical_stage2_enabled
        else ("root_plus_stage1" if seed_stage1_enabled else "root")
    )

    dfvr_eval_no_mutation = _env_bool("DFVR_EVAL_NO_MUTATION", True)
    dfvr_enforce_fixed_stateset = _env_bool("DFVR_ENFORCE_FIXED_STATESET", True)
    dfvr_fixed_states_path_raw = os.getenv("DFVR_FIXED_STATESET_PATH", "").strip()
    dfvr_fixed_states = None
    if dfvr_fixed_states_path_raw:
        from ..optimisation.dfvr_bound import load_dfvr_stateset

        dfvr_fixed_states_path = Path(dfvr_fixed_states_path_raw).expanduser()
        if dfvr_fixed_states_path.exists():
            dfvr_fixed_states = load_dfvr_stateset(dfvr_fixed_states_path)
        elif dfvr_enforce_fixed_stateset and (slack_polish_active or aaub_apply):
            raise ValueError(
                "DFVR fixed state-set is required for candidate polishing but file is missing: "
                f"{dfvr_fixed_states_path}"
            )
    if (slack_polish_active or aaub_apply) and dfvr_enforce_fixed_stateset and dfvr_fixed_states is None:
        raise ValueError(
            "DFVR fixed state-set is required for candidate polishing. "
            "Set DFVR_FIXED_STATESET_PATH to a baseline-emitted stateset file."
        )

    exhaustive_bellman = os.getenv("EXHAUSTIVE_BELLMAN", "0") == "1"
    if exhaustive_bellman and not precomputed_beliefs:
        raise ValueError("EXHAUSTIVE_BELLMAN=1 requires precomputed_beliefs (full belief map).")
    strict_default = exhaustive_bellman
    exhaustive_strict = _env_bool("EXHAUSTIVE_STRICT", strict_default)
    telemetry_schema_version = "rowgen_v2"
    runtime_sidecar_path_raw = os.getenv("EXHAUSTIVE_RUNTIME_SIDECAR_PATH", "").strip()
    runtime_sidecar_path = Path(runtime_sidecar_path_raw).expanduser() if runtime_sidecar_path_raw else None
    runtime_sidecar_run_id = os.getenv("BENCHMARK_RUN_ID", "").strip()
    runtime_sidecar_tier = os.getenv("BENCHMARK_TIER", "").strip()
    runtime_sidecar_contract_profile = os.getenv("BENCHMARK_CONTRACT_PROFILE", "").strip()
    runtime_sidecar_case = os.getenv("BENCHMARK_CASE", "").strip()

    max_states_per_iter = int(os.getenv("MAX_STATES_PER_ITER", str(MAX_STATES_PER_ITER)))
    max_cuts_per_iter = int(os.getenv("MAX_CUTS_PER_ITER", str(MAX_CUTS_PER_ITER)))

    exhaustive_walltime_limit_sec = max(1e-9, float(os.getenv("EXHAUSTIVE_WALLTIME_LIMIT_SEC", "7200")))
    exhaustive_no_progress_limit_sec = max(1e-9, float(os.getenv("EXHAUSTIVE_NO_PROGRESS_LIMIT_SEC", "600")))
    exhaustive_progress_eps = max(0.0, float(os.getenv("EXHAUSTIVE_PROGRESS_EPS", "1e-8")))
    exhaustive_heartbeat_every_sec = max(1e-9, float(os.getenv("EXHAUSTIVE_HEARTBEAT_EVERY_SEC", "30")))
    exhaustive_heartbeat_every_iters = max(1, int(os.getenv("EXHAUSTIVE_HEARTBEAT_EVERY_ITERS", "1")))
    exhaustive_max_cut_queue = max(1, int(os.getenv("EXHAUSTIVE_MAX_CUT_QUEUE", "500000")))
    exhaustive_max_rss_raw = os.getenv("EXHAUSTIVE_MAX_RSS_MB", "").strip()
    exhaustive_max_rss_mb = float(exhaustive_max_rss_raw) if exhaustive_max_rss_raw else None

    deterministic_seed = int(os.getenv("GUROBI_SEED", os.getenv("EXHAUSTIVE_GUROBI_SEED", "0")))
    strict_deterministic_active = exhaustive_bellman and exhaustive_strict
    if strict_deterministic_active:
        master.Params.Threads = 1
        master.Params.Seed = deterministic_seed
        master.Params.Method = 1
        master.Params.Presolve = 1
        _apply_adp_gurobi_env_params()

    def _current_rss_mb():
        if resource is None:
            return None
        try:
            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return None
        # Linux reports KiB; macOS reports bytes.
        if rss > 1e7:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    def _deterministic_state_sort_key(state):
        evidence = _evidence(state)
        return (len(evidence), _state_label(evidence))

    def _rowgen_signature_payload(*, state_labels, cursor, mode_label):
        payload = {
            "signature_schema": "rowgen_pass_signature_v1",
            "theta_mode": theta_mode,
            "mode": mode_label,
            "strict_deterministic_active": strict_deterministic_active,
            "cursor": cursor,
            "settings": {
                "threads": int(master.Params.Threads),
                "seed": deterministic_seed,
                "method": int(master.Params.Method),
                "presolve": int(master.Params.Presolve),
            },
            "state_labels": list(state_labels),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    topk_successors = 0 if exhaustive_bellman else TOPK_SUCCESSORS
    successor_pmin = 0.0 if exhaustive_bellman else PMIN_SUCCESSOR
    successor_prob_cut = 0.0 if exhaustive_bellman else 1e-8

    def _keep_successor_prob(prob: float) -> bool:
        if exhaustive_bellman:
            return prob > 0.0
        return prob >= successor_pmin
    if verbose:
        path_label = "per-gene Φ_j" if per_gene_phi_active else "single-gene Φ"
        print(
            f"[config] Φ path: {path_label}; tuple_rowgen={'on' if tuple_mode else 'off'}; "
            f"theta_mode={theta_mode}"
        )
        if tail_phi_reg_active:
            print(
                "[config] tail_phi_reg=on; "
                f"modes={sorted(tail_phi_reg_modes)} "
                f"min_stage={tail_phi_reg_min_stage} "
                f"weight={tail_phi_reg_weight} "
                f"max_states={tail_phi_reg_max_states} "
                f"max_root_delta={tail_phi_reg_max_root_delta}"
            )
        if aaub_apply:
            print(
                "[config] aaub=on; "
                f"modes={sorted(aaub_modes)} "
                f"fixed={aaub_fixed_enable} "
                f"p12={aaub_p12_enable} "
                f"sign_safe={aaub_sign_safe_mode} "
                f"max_root_delta={aaub_max_root_delta} "
                f"coef_abs_cap={aaub_coef_abs_cap}"
            )
        if slack_polish_active:
            print(
                "[config] slack_polish=on; "
                f"modes={sorted(slack_polish_modes)} "
                f"topk={slack_polish_topk} "
                f"rounds={slack_polish_rounds} "
                f"max_root_delta={slack_polish_max_root_delta} "
                f"phi_delta_max={slack_polish_phi_delta_max} "
                f"minmax={hotspot_minmax_enable}"
            )
        if exhaustive_bellman:
            print(
                "[config] exhaustive_rowgen=on; "
                f"strict={exhaustive_strict} "
                f"max_states_per_iter={max_states_per_iter} "
                f"max_cuts_per_iter={max_cuts_per_iter} "
                f"walltime_limit_sec={exhaustive_walltime_limit_sec} "
                f"no_progress_limit_sec={exhaustive_no_progress_limit_sec}"
            )

    def _any_nonzero(container) -> bool:
        if container is None:
            return False
        if isinstance(container, Mapping):
            return any(_any_nonzero(v) for v in container.values())
        try:
            return abs(float(container)) > 1e-12
        except Exception:
            return False

    if multi_gene and not tuple_mode:
        bad_params = []
        if _any_nonzero(a_gene):
            bad_params.append("a_gene")
        if _any_nonzero(b_gene):
            bad_params.append("b_gene")
        if _any_nonzero(c_gene):
            bad_params.append("c_gene")
        if _any_nonzero(delta_gene):
            bad_params.append("delta_gene")
        if bad_params:
            raise ValueError(
                f"Per-gene parameters {bad_params} provided but tuple row generation is off; "
                "either set ENABLE_TUPLE_ROWGEN=1 or zero out per-gene inputs."
            )
    if per_gene_phi_active and not any(
        _any_nonzero(param) for param in (a_gene, b_gene, c_gene, delta_gene)
    ):
        print("[config] Per-gene parameters are empty/zero; proceeding with additive Φ_j using zero per-gene rewards.")

    if tuple_mode:
        outcome_space = [tuple(outcome) for outcome in product(gen_states, repeat=len(gene_list))]
    else:
        outcome_space = list(gen_states)
    available_outcomes = tuple(outcome_space)

    def _normalize_outcome(outcome):
        if tuple_mode:
            if isinstance(outcome, tuple):
                if len(outcome) != len(gene_list):
                    raise AssertionError(f"Outcome {outcome!r} length mismatch for genes {gene_list}")
                for component in outcome:
                    if component not in gen_states:
                        raise AssertionError(f"Outcome component {component!r} not in gen_states {gen_states}")
                return tuple(outcome)
            if outcome in gen_states:
                return tuple(outcome for _ in gene_list)
            raise AssertionError(f"Invalid scalar outcome {outcome!r} for tuple mode")
        else:
            if isinstance(outcome, tuple):
                if len(outcome) == 1 and outcome[0] in gen_states:
                    return outcome[0]
                raise AssertionError(f"Tuple outcome {outcome!r} unexpected in single-gene mode")
            if outcome not in gen_states:
                raise AssertionError(f"Outcome {outcome!r} not in gen_states {gen_states}")
            return outcome

    def _outcome_label(outcome):
        normalized = _normalize_outcome(outcome)
        if tuple_mode:
            return "-".join(str(val) for val in normalized)
        return str(normalized)

    def _make_z(state_items):
        z_map = {person: {outcome: 0.0 for outcome in available_outcomes} for person in I}
        for person, raw_outcome in state_items:
            normalized = _normalize_outcome(raw_outcome)
            z_map.setdefault(person, {outcome: 0.0 for outcome in available_outcomes})
            z_map[person][normalized] = 1.0
        return z_map

    def _state_from_z0(z0_dict):
        if not z0_dict:
            return frozenset()
        observed = []
        for person, dist in z0_dict.items():
            for outcome_key, val in dist.items():
                if val >= 0.5:
                    observed.append((person, outcome_key))
                    break
        return frozenset((person, _normalize_outcome(outcome)) for person, outcome in observed)

    def _evidence(state):
        if not isinstance(state, frozenset):
            raise AssertionError(
                f"State must be evidence-only frozenset[(person,outcome)], got {type(state).__name__}: {state!r}"
            )
        return state

    # 1) Shadow-price variables:
    # - Monolithic Φ path: W[i, outcome]
    # - Per-gene Φ path:   W_gene[gene, i, g] (and we aggregate W_sol at the end)
    W_var = None
    if not per_gene_phi_active:
        W_var = master.addVars(I, outcome_space, lb=-GRB.INFINITY, name="W")
    W_gene_var = (
        master.addVars(gene_list, I, gen_states, lb=-GRB.INFINITY, name="W_gene")
        if per_gene_phi_active
        else None
    )
    theta_var = None
    theta_vars = None
    theta_stage_base_vars = None
    stage_gene_shared_scale = 1.0 / float(len(gene_list)) if stage_gene_theta_active and gene_list else 0.0
    if stage_gene_theta_active:
        theta_stage_base_vars = master.addVars(
            range(len(I) + 1),
            lb=-GRB.INFINITY,
            name="theta_stage_base",
        )
        theta_vars = master.addVars(range(len(I) + 1), gene_list, lb=-GRB.INFINITY, name="theta_stage_gene")
        for k in range(len(I) + 1):
            master.addConstr(
                gp.quicksum(theta_vars[k, gene] for gene in gene_list) == 0.0,
                name=f"theta_stage_gene_anchor_{k}",
            )
        terminal_stage = len(I)
        theta_stage_base_vars[terminal_stage].LB = 0.0
        theta_stage_base_vars[terminal_stage].UB = 0.0
        for gene in gene_list:
            theta_vars[terminal_stage, gene].LB = 0.0
            theta_vars[terminal_stage, gene].UB = 0.0
    elif stage_theta_active:
        theta_vars = master.addVars(range(len(I) + 1), lb=-GRB.INFINITY, name="theta")
        # Anchor the terminal intercept for numerical stability.
        theta_vars[len(I)].LB = 0.0
        theta_vars[len(I)].UB = 0.0
    elif person_theta_active:
        theta_vars = master.addVars(I, lb=-GRB.INFINITY, name="theta_person")
        # Anchor to avoid arbitrary shifts.
        master.addConstr(
            gp.quicksum(theta_vars[i] for i in I) == 0.0,
            name="theta_person_anchor",
        )
    elif person_stage_theta_active:
        theta_vars = master.addVars(I, range(1, len(I) + 1), lb=-GRB.INFINITY, name="theta_person_stage")
        # Anchor each stage to avoid arbitrary shifts.
        for k in range(1, len(I) + 1):
            master.addConstr(
                gp.quicksum(theta_vars[i, k] for i in I) == 0.0,
                name=f"theta_person_stage_anchor_{k}",
            )
    else:
        theta_var = master.addVar(lb=-GRB.INFINITY, name="theta")

    # Note: do not pin θ in per-gene mode; it must remain free for THETA_MODE=stage to matter.

    aaub_u_vars = None
    aaub_v_vars = None
    aaub_lb = -aaub_coef_abs_cap
    aaub_ub = 0.0 if aaub_sign_safe_mode else aaub_coef_abs_cap
    if aaub_apply and aaub_fixed_enable:
        aaub_u_vars = master.addVars(
            I,
            lb=aaub_lb,
            ub=aaub_ub,
            name="aaub_u",
        )
    if aaub_apply and aaub_p12_enable:
        aaub_v_vars = master.addVars(
            I,
            lb=aaub_lb,
            ub=aaub_ub,
            name="aaub_v",
        )

    # ── Edge feature LP variables ────────────────────────────────
    # NOTE: Use par/ch (not p/c) to avoid shadowing the function parameter `c`
    # (reward coefficient dict) and `p_vals` patterns.
    W_edge_vars = None
    if edge_features_active:
        edge_keys = list(pedigree_edges)
        if per_gene_phi_active:
            W_edge_vars = {}
            for block in edge_feature_blocks:
                block_tag = "resid" if block == "residual" else "raw"
                for gene in gene_list:
                    for par, ch in edge_keys:
                        for g_p in gen_states:
                            for g_c in gen_states:
                                W_edge_vars[block, gene, (par, ch), g_p, g_c] = master.addVar(
                                    lb=-GRB.INFINITY,
                                    name=f"W_edge_{block_tag}_{gene}_{par}_{ch}_{g_p}_{g_c}",
                                )
        else:
            W_edge_vars = {}
            for block in edge_feature_blocks:
                block_tag = "resid" if block == "residual" else "raw"
                for par, ch in edge_keys:
                    for g_p in gen_states:
                        for g_c in gen_states:
                            W_edge_vars[block, (par, ch), g_p, g_c] = master.addVar(
                                lb=-GRB.INFINITY,
                                name=f"W_edge_{block_tag}_{par}_{ch}_{g_p}_{g_c}",
                            )
        if verbose:
            edge_var_count = len(edge_feature_blocks) * len(pedigree_edges) * len(gen_states) ** 2
            if per_gene_phi_active:
                edge_var_count *= len(gene_list)
            print(
                f"[config] edge_features=on; mode={edge_feature_mode}; "
                f"edges={len(pedigree_edges)} vars={edge_var_count}; seed_scope={edge_seed_scope}"
            )

    W_trio_vars = None
    if trio_features_active:
        W_trio_vars = {}
        if per_gene_phi_active:
            for block in trio_feature_blocks:
                block_tag = "pure3" if block == "pure3" else "raw"
                for gene in gene_list:
                    for share_key in trio_share_keys:
                        share_label = (
                            f"depth_{share_key}"
                            if trio_coef_sharing == "child_depth"
                            else f"{share_key[0]}_{share_key[1]}_{share_key[2]}"
                        )
                        for g_f in gen_states:
                            for g_m in gen_states:
                                for g_c in gen_states:
                                    W_trio_vars[block, gene, share_key, g_f, g_m, g_c] = master.addVar(
                                        lb=-GRB.INFINITY,
                                        name=f"W_trio_{block_tag}_{gene}_{share_label}_{g_f}_{g_m}_{g_c}",
                                    )
        else:
            for block in trio_feature_blocks:
                block_tag = "pure3" if block == "pure3" else "raw"
                for share_key in trio_share_keys:
                    share_label = (
                        f"depth_{share_key}"
                        if trio_coef_sharing == "child_depth"
                        else f"{share_key[0]}_{share_key[1]}_{share_key[2]}"
                    )
                    for g_f in gen_states:
                        for g_m in gen_states:
                            for g_c in gen_states:
                                W_trio_vars[block, share_key, g_f, g_m, g_c] = master.addVar(
                                    lb=-GRB.INFINITY,
                                    name=f"W_trio_{block_tag}_{share_label}_{g_f}_{g_m}_{g_c}",
                                )
        if verbose:
            trio_var_count = len(trio_feature_blocks) * len(trio_share_keys) * len(gen_states) ** 3
            if per_gene_phi_active:
                trio_var_count *= len(gene_list)
            print(
                f"[config] trio_features=on; mode={trio_feature_mode}; sharing={trio_coef_sharing}; "
                f"trios={len(pedigree_trios)} shared_keys={len(trio_share_keys)} vars={trio_var_count}; "
                f"seed_scope={trio_seed_scope}"
            )

    def _theta_for_state(state):
        if stage_gene_theta_active:
            stage = len(_evidence(state))
            return theta_stage_base_vars[stage]
        if stage_theta_active:
            return theta_vars[len(_evidence(state))]
        if person_theta_active:
            tested_people = {person for person, _ in _evidence(state)}
            return gp.quicksum(theta_vars[p] for p in tested_people)
        if person_stage_theta_active:
            tested_people = {person for person, _ in _evidence(state)}
            stage = len(_evidence(state))
            if stage <= 0:
                return gp.LinExpr(0.0)
            return gp.quicksum(theta_vars[person, stage] for person in tested_people)
        return theta_var

    def _theta_for_state_gene(state, gene):
        if stage_gene_theta_active:
            stage = len(_evidence(state))
            return theta_stage_base_vars[stage] * stage_gene_shared_scale + theta_vars[stage, gene]
        return _theta_for_state(state) * (1.0 / len(gene_list))

    def _theta_model_correction_expr(state):
        correction = evaluate_theta_model_correction(
            state,
            belief=belief,
            individuals=I,
            gen_states=gen_states,
            pedigree=pedigree,
            theta_model=theta_model_info.get("theta_model"),
            theta_model_spec=theta_model_info.get("theta_model_spec"),
        )
        return gp.LinExpr(float(correction))

    def _w_key(person, outcome):
        normalized = _normalize_outcome(outcome)
        if isinstance(normalized, tuple):
            return (person,) + normalized
        return (person, normalized)

    def _get_w(person, outcome):
        if W_var is None:
            raise RuntimeError("Monolithic W not initialised")
        return W_var[_w_key(person, outcome)]
    def _get_w_gene(gene, person, genotype):
        if W_gene_var is None:
            raise RuntimeError("Per-gene W not initialised")
        return W_gene_var[gene, person, genotype]

    def _safe_var_value(var, default: float = 0.0) -> float:
        try:
            return float(var.X)
        except Exception:
            return float(default)

    def _current_theta_star():
        if stage_gene_theta_active:
            return {
                (k, gene): _safe_var_value(theta_stage_base_vars[k]) * stage_gene_shared_scale
                + _safe_var_value(theta_vars[k, gene])
                for k in range(len(I) + 1)
                for gene in gene_list
            }
        if stage_theta_active:
            return [_safe_var_value(theta_vars[k]) for k in range(len(I) + 1)]
        if person_theta_active:
            return {person: _safe_var_value(theta_vars[person]) for person in I}
        if person_stage_theta_active:
            return {(person, k): _safe_var_value(theta_vars[person, k]) for person in I for k in range(1, len(I) + 1)}
        return _safe_var_value(theta_var)

    def _current_w_gene_solution():
        if not (per_gene_phi_active and W_gene_var is not None):
            return None
        return {
            gene: {i: {g: _safe_var_value(_get_w_gene(gene, i, g)) for g in gen_states} for i in I}
            for gene in gene_list
        }

    def _current_w_solution():
        if per_gene_phi_active and W_gene_var is not None:
            w_gene_sol = _current_w_gene_solution() or {}
            w_sol = {i: {} for i in I}
            for outcome in available_outcomes:
                for i in I:
                    total = 0.0
                    for idx, gene in enumerate(gene_list):
                        comp = outcome[idx]
                        total += w_gene_sol[gene][i][comp]
                    w_sol[i][outcome] = total
            return w_sol
        return {i: {g: _safe_var_value(_get_w(i, g)) for g in available_outcomes} for i in I}

    def _current_phi_solution():
        phi_sol = {}
        for state, phi_var in Phi.items():
            try:
                phi_sol[state] = phi_var.X
            except Exception:
                continue
        return phi_sol

    def _current_aaub_star():
        if not aaub_apply:
            return None
        return {
            "fixed_enabled": aaub_u_vars is not None,
            "p12_enabled": aaub_v_vars is not None,
            "u": {person: _safe_var_value(aaub_u_vars[person]) for person in I} if aaub_u_vars is not None else {},
            "v": {person: _safe_var_value(aaub_v_vars[person]) for person in I} if aaub_v_vars is not None else {},
        }

    def _current_edge_star():
        if not edge_features_active or W_edge_vars is None:
            return None
        block_payload = {}
        for block in edge_feature_blocks:
            if per_gene_phi_active:
                block_payload[block] = {
                    gene: {
                        (p, c): {
                            (gp, gc): _safe_var_value(W_edge_vars[block, gene, (p, c), gp, gc])
                            for gp in gen_states
                            for gc in gen_states
                        }
                        for p, c in pedigree_edges
                    }
                    for gene in gene_list
                }
            else:
                block_payload[block] = {
                    (p, c): {
                        (gp, gc): _safe_var_value(W_edge_vars[block, (p, c), gp, gc])
                        for gp in gen_states
                        for gc in gen_states
                    }
                    for p, c in pedigree_edges
                }

        if edge_feature_mode == "raw":
            return block_payload["raw"]
        return {
            "__mode__": edge_feature_mode,
            "__blocks__": list(edge_feature_blocks),
            **block_payload,
        }

    def _current_trio_star():
        if not trio_features_active or W_trio_vars is None:
            return None
        block_payload = {}
        for block in trio_feature_blocks:
            if per_gene_phi_active:
                block_payload[block] = {
                    gene: {
                        share_key: {
                            (g_f, g_m, g_c): _safe_var_value(W_trio_vars[block, gene, share_key, g_f, g_m, g_c])
                            for g_f in gen_states
                            for g_m in gen_states
                            for g_c in gen_states
                        }
                        for share_key in trio_share_keys
                    }
                    for gene in gene_list
                }
            else:
                block_payload[block] = {
                    share_key: {
                        (g_f, g_m, g_c): _safe_var_value(W_trio_vars[block, share_key, g_f, g_m, g_c])
                        for g_f in gen_states
                        for g_m in gen_states
                        for g_c in gen_states
                    }
                    for share_key in trio_share_keys
                }
        share_key_payload = [
            {
                "trio": [parent1, parent2, child],
                "share_key": trio_share_key_by_trio[(parent1, parent2, child)],
            }
            for parent1, parent2, child in pedigree_trios
        ]
        return {
            "__mode__": trio_feature_mode,
            "__blocks__": list(trio_feature_blocks),
            "__sharing__": trio_coef_sharing,
            "__share_keys__": share_key_payload,
            "__trios__": [list(trio) for trio in pedigree_trios],
            **block_payload,
        }

    def _feature_coef_summary(payload):
        if not isinstance(payload, Mapping):
            return 0.0, 0.0, 0

        l1 = 0.0
        l2 = 0.0
        nonzero = 0

        def _walk(value):
            nonlocal l1, l2, nonzero
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(key, str) and key.startswith("__"):
                        continue
                    _walk(item)
                return
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            l1 += abs(numeric)
            l2 += (numeric * numeric)
            if abs(numeric) > 1e-12:
                nonzero += 1

        _walk(payload)
        return l1, math.sqrt(l2), nonzero

    def _edge_coef_summary(payload):
        return _feature_coef_summary(payload)

    def _posterior_marginals(posterior_entry):
        if isinstance(posterior_entry, InferenceResult):
            return posterior_entry.marginals
        return posterior_entry

    trio_feature_cache = {} if trio_features_active else None
    trio_feature_cache_hits_total = 0
    trio_feature_cache_misses_total = 0
    trio_feature_materialization_sec_total = 0.0

    def _record_trio_bundle_telemetry(bundle):
        nonlocal trio_feature_cache_hits_total, trio_feature_cache_misses_total, trio_feature_materialization_sec_total
        if not isinstance(bundle, Mapping):
            return
        cache_payload = bundle.get("cache", {})
        telemetry_payload = bundle.get("telemetry", {})
        if isinstance(cache_payload, Mapping):
            trio_feature_cache_hits_total += int(cache_payload.get("hits", 0) or 0)
            trio_feature_cache_misses_total += int(cache_payload.get("misses", 0) or 0)
        if isinstance(telemetry_payload, Mapping):
            trio_feature_materialization_sec_total += float(telemetry_payload.get("materialization_sec", 0.0) or 0.0)

    def _aaub_expr_for_state(*, tested_obs, posterior_entry):
        if not aaub_apply:
            return gp.LinExpr(0.0)
        p_s = _posterior_marginals(posterior_entry)
        expr = gp.LinExpr()
        for person in I:
            if person in tested_obs:
                continue
            if aaub_u_vars is not None:
                expr.add(aaub_u_vars[person])
            if aaub_v_vars is not None:
                person_probs = p_s.get(person, {})
                p12 = person_probs.get(1, 0.0) + person_probs.get(2, 0.0)
                if abs(p12) > 0.0:
                    expr.add(p12 * aaub_v_vars[person])
        return expr

    def _edge_expr_for_state(*, tested_obs, posterior_entry, state):
        if not edge_features_active or W_edge_vars is None:
            return gp.LinExpr(0.0)
        from ..models.belief import get_pairwise_marginals
        p_s = _posterior_marginals(posterior_entry)
        gene_probs = _ensure_gene_posteriors(state, posterior_entry) or {}
        evidence_dict = dict(tested_obs)
        expr = gp.LinExpr()

        if per_gene_phi_active:
            pair_marginals = get_pairwise_marginals(
                infer, I, gen_states, evidence_dict, pedigree_edges,
                genes=gene_list,
            )
            for gene in gene_list:
                gene_pairs = pair_marginals.get(gene, {})
                gene_parent_probs = gene_probs.get(gene, {})
                for parent, child in pedigree_edges:
                    pm = gene_pairs.get((parent, child), {})
                    parent_probs = gene_parent_probs.get(parent, {})
                    child_probs = gene_parent_probs.get(child, {})
                    for (gp_val, gc_val), prob in pm.items():
                        if abs(prob) <= 0.0:
                            continue
                        if "raw" in edge_feature_blocks:
                            expr.add(prob * W_edge_vars["raw", gene, (parent, child), gp_val, gc_val])
                        if "residual" in edge_feature_blocks:
                            residual = prob - (parent_probs.get(gp_val, 0.0) * child_probs.get(gc_val, 0.0))
                            if abs(residual) > 0.0:
                                expr.add(
                                    residual * W_edge_vars["residual", gene, (parent, child), gp_val, gc_val]
                                )
        else:
            pair_marginals = get_pairwise_marginals(
                infer, I, gen_states, evidence_dict, pedigree_edges,
            )
            for parent, child in pedigree_edges:
                pm = pair_marginals.get((parent, child), {})
                parent_probs = p_s.get(parent, {})
                child_probs = p_s.get(child, {})
                for (gp_val, gc_val), prob in pm.items():
                    if abs(prob) <= 0.0:
                        continue
                    if "raw" in edge_feature_blocks:
                        expr.add(prob * W_edge_vars["raw", (parent, child), gp_val, gc_val])
                    if "residual" in edge_feature_blocks:
                        residual = prob - (parent_probs.get(gp_val, 0.0) * child_probs.get(gc_val, 0.0))
                        if abs(residual) > 0.0:
                                expr.add(
                                    residual * W_edge_vars["residual", (parent, child), gp_val, gc_val]
                                )
        return expr

    def _trio_expr_for_state(*, tested_obs, posterior_entry, state):
        if not trio_features_active or W_trio_vars is None:
            return gp.LinExpr(0.0)
        expr = gp.LinExpr()
        bundle = build_state_feature_bundle(
            infer=infer,
            individuals=I,
            gen_states=gen_states,
            evidence_state=dict(tested_obs),
            pedigree_trios=pedigree_trios,
            posterior_entry=posterior_entry,
            genes=gene_list if per_gene_phi_active else None,
            feature_cache=trio_feature_cache,
            trio_sharing=trio_coef_sharing,
            share_key_map=trio_share_key_by_trio,
        )
        _record_trio_bundle_telemetry(bundle)

        for gene, _trio, share_key, trio_dist, pure3_residuals in iter_trio_feature_rows(bundle):
            for g_f in gen_states:
                for g_m in gen_states:
                    for g_c in gen_states:
                        triple = (g_f, g_m, g_c)
                        joint_prob = float(trio_dist.get(triple, 0.0))
                        pure3 = float(pure3_residuals.get(triple, 0.0))
                        if per_gene_phi_active:
                            if gene is None:
                                continue
                            if "raw" in trio_feature_blocks and abs(joint_prob) > 0.0:
                                expr.add(joint_prob * W_trio_vars["raw", gene, share_key, g_f, g_m, g_c])
                            if "pure3" in trio_feature_blocks and abs(pure3) > 0.0:
                                expr.add(pure3 * W_trio_vars["pure3", gene, share_key, g_f, g_m, g_c])
                        else:
                            if "raw" in trio_feature_blocks and abs(joint_prob) > 0.0:
                                expr.add(joint_prob * W_trio_vars["raw", share_key, g_f, g_m, g_c])
                            if "pure3" in trio_feature_blocks and abs(pure3) > 0.0:
                                expr.add(pure3 * W_trio_vars["pure3", share_key, g_f, g_m, g_c])
        return expr

    # 2) Belief map: state → (p_s, z_s)
    initial_evidence = _state_from_z0(z0)
    initial_state = initial_evidence
    if isinstance(p0, InferenceResult):
        root_posterior = p0
    else:
        if multi_gene:
            base_marginals = {person: dict(dist) for person, dist in p0.items()}
            per_gene_init = (
                {
                    gene: {person: dict(dist) for person, dist in dist_map.items()}
                    for gene, dist_map in p0_gene.items()
                }
                if p0_gene
                else None
            )
            root_posterior = InferenceResult(
                base_marginals,
                per_gene=per_gene_init,
                gene_order=gene_list,
                gen_states=gen_states,
            )
        else:
            base_marginals = {person: dict(dist) for person, dist in p0.items()}
            root_posterior = InferenceResult(
                base_marginals,
                gene_order=("gene",),
                gen_states=gen_states,
            )

    belief = {}
    belief_gene = {}
    tuple_posteriors = {}
    def _store_belief(state, posterior):
        evidence = _evidence(state)
        entry = (posterior, _make_z(evidence))
        belief[state] = entry
        return entry

    def _store_tuple_posteriors(state, tuple_pmfs):
        if not tuple_mode:
            return
        tuple_posteriors[state] = tuple_pmfs

    _store_belief(initial_state, root_posterior)

    def _ensure_gene_posteriors(state, posteriors):
        if not multi_gene:
            return None
        if isinstance(posteriors, InferenceResult):
            per_gene = posteriors.get_per_gene_probs()
            if per_gene:
                belief_gene[state] = {
                    gene: {
                        individual: dict(dist)
                        for individual, dist in per_person.items()
                    }
                    for gene, per_person in per_gene.items()
                }
                return belief_gene[state]
            posteriors_map = posteriors.marginals
        else:
            posteriors_map = posteriors
        if state not in belief_gene:
            belief_gene[state] = lift_single_gene_posteriors_to_genes(posteriors_map, gene_list)
        return belief_gene[state]

    def _per_gene_p12_map(gene_posteriors, individual):
        if not gene_posteriors:
            return None
        carrier = {}
        for gene, probs in gene_posteriors.items():
            dist = probs.get(individual)
            if not dist:
                continue
            carrier_prob = dist.get(1, 0.0) + dist.get(2, 0.0)
            carrier[gene] = carrier_prob
        return carrier or None

    if exhaustive_bellman and precomputed_beliefs is not None:
        for state, entry in precomputed_beliefs.items():
            evidence = _evidence(state)
            if evidence in belief:
                continue
            if isinstance(entry, InferenceResult):
                posterior = entry
            else:
                posterior = InferenceResult(
                    entry,
                    gene_order=gene_list if multi_gene else ("gene",),
                    gen_states=gen_states,
                )
            _store_belief(evidence, posterior)
            _ensure_gene_posteriors(evidence, posterior)
            if tuple_mode and isinstance(posterior, InferenceResult) and posterior.has_tuple_pmfs():
                _store_tuple_posteriors(evidence, posterior.get_tuple_pmfs())

    if multi_gene:
        root_gene_probs = root_posterior.get_per_gene_probs()
        if root_gene_probs:
            belief_gene[initial_state] = {
                gene: {
                    individual: dict(dist)
                    for individual, dist in per_person.items()
                }
                for gene, per_person in root_gene_probs.items()
            }
        elif p0_gene:
            belief_gene[initial_state] = {
                gene: {ind: dict(dist) for ind, dist in dist_map.items()}
                for gene, dist_map in p0_gene.items()
            }
        else:
            _ensure_gene_posteriors(initial_state, root_posterior)

    # --- Optionally auto-discover role_groups (probability mode by default) ---
    import os as _os
    if tuple_mode:
        # Do not merge states across individuals when tuple outcomes are active;
        # canon collisions can erase tested-vs-untested distinctions.
        role_groups = None
    elif role_groups is None and _os.getenv("AUTO_DISCOVER_ROLES", "1") == "1":
        role_groups = discover_role_groups(
            I,
            gen_states,
            p0=p0,
            child_cpds=child_cpds,
            pedigree=pedigree,
            mode="probability",
            a=a,
            b=b,
            c=c,
            delta=delta,
            min_group=2,
        )
        if verbose:
            print("[roles][auto] discovered groups:", {k: sorted(v) for k, v in role_groups.items()})

    # --- Validate role_groups (prints when verbose or VALIDATE_ROLE_GROUPS=1) ---
    if role_groups and (verbose or _os.getenv("VALIDATE_ROLE_GROUPS", "0") == "1"):
        rep = validate_role_groups(I, role_groups, gen_states, p0=p0, child_cpds=child_cpds,
                                   pedigree=pedigree, a=a, b=b, c=c, delta=delta)
        if rep.get('errors'):   print("[roles][errors]",   rep['errors'])
        if rep.get('warnings'): print("[roles][warnings]", rep['warnings'])
        if rep.get('ok'):       print("[roles][ok]",       rep['ok'])

    # Initialize Bellman row generator (probability-only roles)
    bellman_gen = BellmanRowGenerator(I, gen_states, infer, pedigree, tol, verbose,
                                      role_groups=role_groups,
                                      genes=gene_list if multi_gene else None,
                                      tuple_mode=tuple_mode,
                                      outcomes=available_outcomes)

    # Initialize caches once
    if multi_gene:
        def propagate_wrapper(infer_obj, individuals, gen_st, evidence_dict):
            return propagate_multigene_marginals(
                infer_obj,
                individuals,
                gen_st,
                evidence_dict,
                gene_list,
                aggregate_only=not tuple_mode,
            )

        propagate_fn = propagate_wrapper
    else:
        propagate_fn = propagate_all_marginals

    inf_cache = InferenceCache(
        infer,
        individuals=I,
        gen_states=gen_states,
        propagate_fn=propagate_fn,
        tuple_mode=tuple_mode,
    )
    if tuple_mode and inf_cache.tuple_posteriors:
        tuple_posteriors.update(inf_cache.tuple_posteriors)
    
    # Enable cache diagnostics if requested
    if cache_diagnostics:
        inf_cache.enable_diagnostics(True)
    
    # Pre-populate cache with exact DP beliefs if provided
    if precomputed_beliefs is not None:
        print(f"Pre-populating ADP cache with {len(precomputed_beliefs)} exact DP beliefs...")
        for state, belief_dict in precomputed_beliefs.items():
            # Convert state frozenset to evidence dict for cache
            evidence = dict(_evidence(state))  # {'Father': 1, 'Child': 0}

            # Directly populate the cache with precomputed beliefs (no inference needed)
            inf_cache.set_precomputed(evidence, belief_dict)
        
        print(f"Successfully pre-populated {len(precomputed_beliefs)} beliefs into ADP cache")

    cut_cache = CutManager()

    # 2) Φ variables (default identity; merge only if opted-in)
    Phi = {}
    Phi_canon = {}
    Phi_gene = {gene: {} for gene in gene_list} if per_gene_phi_active else {}
    state_projection_cache = {}
    successor_projection_cache = {}

    def _state_label(state):
        """
        Turn a frozenset of (i,g) tuples into a name like
          "Father_0_Mother_2"
        or return "root" if empty.
        """
        evidence = _evidence(state)
        if not evidence:
            base = "root"
        else:
            # every element must be a (person,genotype) pair
            labels = []
            for elt in evidence:
                if not (isinstance(elt, tuple) and len(elt) == 2):
                    raise AssertionError(f"Invalid state element {elt!r}, must be (person,genotype)")
                person, g = elt
                if person not in I:
                    raise AssertionError(f"Invalid state element {elt!r}, unknown person")
                normalized = _normalize_outcome(g)
                g_label = _outcome_label(normalized)
                labels.append(f"{person}_{g_label}")
            base = "_".join(sorted(labels))
        return base

    bellman_row_registry = []

    def _json_outcome_for_row_export(outcome):
        normalized = _normalize_outcome(outcome)
        if tuple_mode:
            return [int(value) if isinstance(value, bool) or isinstance(value, int) else value for value in normalized]
        return int(normalized) if isinstance(normalized, bool) or isinstance(normalized, int) else normalized

    def _json_state_key_for_row_export(state):
        entries = []
        for person, outcome in sorted(_evidence(state), key=lambda item: (str(item[0]), _outcome_label(item[1]))):
            entries.append([str(person), _json_outcome_for_row_export(outcome)])
        return entries

    def _canonical_bellman_row_key(state, row_type, person_tested=None):
        action_label = "STOP" if row_type == "stop" else str(person_tested)
        return f"{_state_label(state)}|{row_type}|{action_label}"

    def _register_bellman_row_constraint(
        constraint,
        *,
        row_type,
        state,
        action,
        person_tested=None,
        rhs_const=None,
        immediate_reward=None,
        successors=None,
        source,
        truncated=False,
    ):
        bellman_row_registry.append(
            {
                "constraint": constraint,
                "constraint_name": None,
                "row_type": str(row_type),
                "state": state,
                "action": str(action),
                "person_tested": None if person_tested is None else str(person_tested),
                "rhs_const": None if rhs_const is None else float(rhs_const),
                "immediate_reward": None if immediate_reward is None else float(immediate_reward),
                "successors": list(successors or ()),
                "source": str(source),
                "truncated": bool(truncated),
                "canonical_row_key": _canonical_bellman_row_key(state, row_type, person_tested),
            }
        )

    def _export_bellman_row_duals():
        rows = []
        if master.status != GRB.Status.OPTIMAL:
            return {
                "schema": "bellman_row_dual_export_v1",
                "available": False,
                "reason": f"master_status_{master.status}",
                "rows": [],
                "aggregated_rows": [],
                "validation": {
                    "dual_component_available": False,
                    "nonzero_dual_rows_not_truncated": False,
                    "duplicate_canonical_rows_aggregated": True,
                    "complementarity_check_pass": False,
                },
            }

        for idx, meta in enumerate(bellman_row_registry):
            constr = meta.get("constraint")
            state = meta["state"]
            row_type = meta["row_type"]
            try:
                constraint_name = str(constr.ConstrName)
            except Exception:
                constraint_name = meta.get("constraint_name")
            try:
                lhs_value = float(get_Phi(state).X)
            except Exception:
                lhs_value = None
            if row_type == "stop":
                rhs_value = meta.get("rhs_const")
            else:
                rhs_value = meta.get("immediate_reward")
                if rhs_value is not None:
                    rhs_value = float(rhs_value) + sum(
                        float(prob) * float(get_Phi(succ).X)
                        for succ, prob in meta.get("successors", ())
                    )
            canonical_slack = None
            if lhs_value is not None and rhs_value is not None:
                canonical_slack = float(lhs_value - float(rhs_value))
            try:
                gurobi_slack = float(constr.Slack)
            except Exception:
                gurobi_slack = None
            try:
                gurobi_pi = float(constr.Pi)
            except Exception:
                gurobi_pi = None
            dual_abs = abs(gurobi_pi) if gurobi_pi is not None and math.isfinite(gurobi_pi) else 0.0
            complementarity_abs = (
                abs(dual_abs * float(canonical_slack))
                if canonical_slack is not None and math.isfinite(float(canonical_slack))
                else None
            )
            stage = len(_evidence(state))
            rows.append(
                {
                    "row_id": f"bellman_row_{idx:06d}",
                    "constraint_name": constraint_name,
                    "row_type": row_type,
                    "state_key": _json_state_key_for_row_export(state),
                    "stage": int(stage),
                    "action": meta.get("action"),
                    "person_tested": meta.get("person_tested"),
                    "lhs_value": lhs_value,
                    "rhs_value": None if rhs_value is None else float(rhs_value),
                    "canonical_slack": canonical_slack,
                    "gurobi_slack": gurobi_slack,
                    "gurobi_pi": gurobi_pi,
                    "dual_abs": float(dual_abs),
                    "tightness_weight": (
                        float(1.0 / (1e-7 + abs(float(canonical_slack))))
                        if canonical_slack is not None and math.isfinite(float(canonical_slack))
                        else 0.0
                    ),
                    "margin_weight": 0.0,
                    "reach_weight": float(1.0 / (1.0 + stage)),
                    "canonical_row_key": meta.get("canonical_row_key"),
                    "source": meta.get("source"),
                    "truncated": bool(meta.get("truncated", False)),
                    "complementarity_abs": complementarity_abs,
                }
            )

        best_rhs_by_state = {}
        for row in rows:
            rhs = row.get("rhs_value")
            if rhs is None:
                continue
            state_key = json.dumps(row.get("state_key"), sort_keys=True, separators=(",", ":"))
            best_rhs_by_state[state_key] = max(float(rhs), best_rhs_by_state.get(state_key, -math.inf))
        for row in rows:
            rhs = row.get("rhs_value")
            state_key = json.dumps(row.get("state_key"), sort_keys=True, separators=(",", ":"))
            best_rhs = best_rhs_by_state.get(state_key)
            if rhs is None or best_rhs is None or not math.isfinite(best_rhs):
                row["action_margin"] = None
                row["margin_weight"] = 0.0
                continue
            margin = max(0.0, float(best_rhs) - float(rhs))
            row["action_margin"] = float(margin)
            row["margin_weight"] = float(1.0 / (1e-7 + margin))

        aggregates = {}
        for row in rows:
            key = row["canonical_row_key"]
            agg = aggregates.setdefault(
                key,
                {
                    "canonical_row_key": key,
                    "row_type": row.get("row_type"),
                    "state_key": row.get("state_key"),
                    "stage": row.get("stage"),
                    "action": row.get("action"),
                    "person_tested": row.get("person_tested"),
                    "dual_abs": 0.0,
                    "raw_constraint_count": 0,
                    "nonzero_dual_row_count": 0,
                    "truncated_nonzero_dual_row_count": 0,
                    "constraint_names": [],
                    "sources": [],
                    "min_abs_canonical_slack": None,
                    "max_complementarity_abs": 0.0,
                },
            )
            dual_abs = float(row.get("dual_abs") or 0.0)
            agg["dual_abs"] += dual_abs
            agg["raw_constraint_count"] += 1
            if row.get("constraint_name") is not None:
                agg["constraint_names"].append(row.get("constraint_name"))
            if row.get("source") not in agg["sources"]:
                agg["sources"].append(row.get("source"))
            if dual_abs > 1e-12:
                agg["nonzero_dual_row_count"] += 1
                if row.get("truncated"):
                    agg["truncated_nonzero_dual_row_count"] += 1
            slack = row.get("canonical_slack")
            if slack is not None:
                abs_slack = abs(float(slack))
                current = agg.get("min_abs_canonical_slack")
                agg["min_abs_canonical_slack"] = abs_slack if current is None else min(float(current), abs_slack)
            comp = row.get("complementarity_abs")
            if comp is not None:
                agg["max_complementarity_abs"] = max(float(agg["max_complementarity_abs"]), abs(float(comp)))

        aggregated_rows = sorted(
            aggregates.values(),
            key=lambda item: (-float(item.get("dual_abs") or 0.0), str(item.get("canonical_row_key"))),
        )
        total_dual_abs = sum(float(row.get("dual_abs") or 0.0) for row in rows)
        nonzero_dual_row_count = sum(1 for row in rows if float(row.get("dual_abs") or 0.0) > 1e-12)
        truncated_nonzero_dual_row_count = sum(
            1 for row in rows if row.get("truncated") and float(row.get("dual_abs") or 0.0) > 1e-12
        )
        duplicate_canonical_row_count = sum(
            1 for row in aggregated_rows if int(row.get("raw_constraint_count") or 0) > 1
        )
        max_complementarity_abs = max(
            (float(row.get("complementarity_abs") or 0.0) for row in rows),
            default=0.0,
        )
        complementarity_tol = 1e-6
        validation = {
            "dual_component_available": bool(nonzero_dual_row_count > 0 and total_dual_abs > 0.0),
            "nonzero_dual_rows_not_truncated": bool(truncated_nonzero_dual_row_count == 0),
            "duplicate_canonical_rows_aggregated": True,
            "complementarity_check_pass": bool(max_complementarity_abs <= complementarity_tol),
            "complementarity_tol": complementarity_tol,
        }
        return {
            "schema": "bellman_row_dual_export_v1",
            "available": True,
            "canonical_form": "Phi(s)-R(s,a)-E[Phi(s')|s,a] >= 0",
            "duplicate_aggregation": "sum_abs_pi_by_canonical_row_key",
            "row_count": int(len(rows)),
            "aggregated_row_count": int(len(aggregated_rows)),
            "duplicate_canonical_row_count": int(duplicate_canonical_row_count),
            "nonzero_dual_row_count": int(nonzero_dual_row_count),
            "truncated_nonzero_dual_row_count": int(truncated_nonzero_dual_row_count),
            "total_dual_abs": float(total_dual_abs),
            "max_complementarity_abs": float(max_complementarity_abs),
            "validation": validation,
            "rows": rows,
            "aggregated_rows": aggregated_rows,
        }

    def _register_projection(state, projection=None):
        if not (multi_gene and tuple_mode):
            return None
        if projection is None:
            projection = project_state_by_gene(_evidence(state), gene_list)
        state_projection_cache[state] = projection
        return projection

    def _projection_label(gene, projection):
        if not projection:
            return f"{gene}_root"
        parts = [f"{person}_{genotype}" for person, genotype in sorted(projection)]
        return f"{gene}_" + "_".join(parts)

    # Seed root projection
    _register_projection(initial_state)

    def _merge_state(state, person, outcome):
        evidence = _evidence(state)
        state_dict = dict(evidence)
        normalized_outcome = _normalize_outcome(outcome)
        state_dict[person] = normalized_outcome
        merged = frozenset(state_dict.items())
        if multi_gene and tuple_mode:
            projection = project_successor_by_gene(evidence, person, normalized_outcome, gene_list)
            state_projection_cache[merged] = projection
            successor_projection_cache[(state, person, normalized_outcome)] = projection
        return merged

    myopic_eval = None
    myopic_policy_map = {}
    myopic_value_map = {}
    myopic_residuals = {}
    myopic_action_margins = {}
    myopic_feature_names = []
    myopic_selected_features = []
    myopic_adp_vars = {}
    myopic_adp_variable_order = []
    myopic_adp_zero_column_features = set()
    myopic_adp_mode = None
    myopic_adp_diagnostics = {"enabled": bool(myopic_adp_active)}
    oracle_adp_active = _env_bool("ORACLE_ADP_ENABLED", False)
    oracle_adp_mode = os.getenv("ORACLE_ADP_MODE", "exact_value_fixed").strip().lower()
    oracle_plumbing_mode = os.getenv("ORACLE_PLUMBING_MODE", "legacy").strip().lower() or "legacy"
    if oracle_plumbing_mode not in {
        "legacy",
        "oracle_only_fixed_phi",
        "oracle_plus_legacy_residual",
        "gauged_oracle_residual",
    }:
        raise ValueError(
            "Unknown ORACLE_PLUMBING_MODE="
            f"{oracle_plumbing_mode!r} (expected legacy, oracle_only_fixed_phi, "
            "oracle_plus_legacy_residual, or gauged_oracle_residual)."
        )
    if oracle_plumbing_mode != "legacy" and (not oracle_adp_active or oracle_adp_mode != "exact_value_fixed"):
        raise ValueError(
            "ORACLE_PLUMBING_MODE other than 'legacy' requires "
            "ORACLE_ADP_ENABLED=1 and ORACLE_ADP_MODE=exact_value_fixed."
        )
    oracle_only_fixed_phi_active = oracle_plumbing_mode == "oracle_only_fixed_phi"
    oracle_gauge_active = oracle_plumbing_mode == "gauged_oracle_residual"
    oracle_plumbing_diagnostic_active = oracle_plumbing_mode != "legacy"
    tuple_strengthening_disabled_active = (
        disable_truncated_tuple_strengthening or oracle_plumbing_diagnostic_active
    )
    oracle_adp_top_k_raw = os.getenv("ORACLE_ADP_TOP_K", "32").strip()
    oracle_adp_top_k = max(1, int(oracle_adp_top_k_raw or "32"))
    oracle_adp_vars = {}
    oracle_adp_star = None
    oracle_residual_expr_by_state = {}
    oracle_gauge_constraint_names = []
    oracle_adp_diagnostics = {
        "enabled": bool(oracle_adp_active),
        "mode": oracle_adp_mode if oracle_adp_active else None,
        "plumbing_mode": oracle_plumbing_mode if oracle_adp_active else None,
        "oracle_plumbing_mode": oracle_plumbing_mode if oracle_adp_active else None,
        "top_k": oracle_adp_top_k if oracle_adp_active else None,
        "payload_available": isinstance(oracle_adp_payload, Mapping),
        "active_in_lp_construction": bool(oracle_adp_active),
        "active_in_phi_reconstruction": bool(oracle_adp_active),
        "legacy_residual_enabled": bool(oracle_adp_active and not oracle_only_fixed_phi_active),
        "gauge_constraints_added": [],
    }
    regime_residual_feature_names = []
    regime_residual_selected_features = []
    regime_residual_vars = {}
    regime_residual_star = None
    regime_residual_diagnostics = {
        "enabled": bool(regime_residual_active),
        "mode": "gauged_regime_residual_v1" if regime_residual_active else None,
        "top_k": regime_residual_top_k if regime_residual_active else None,
        "min_signature_ratio": (
            regime_residual_min_signature_ratio if regime_residual_active else None
        ),
        "incremental_tol": (
            regime_residual_incremental_tol if regime_residual_active else None
        ),
        "uses_oracle_inputs": False,
        "feature_bank": regime_feature_bank if regime_residual_active else None,
        "feature_semantics": regime_feature_semantics if regime_residual_active else None,
        "feature_definition": {
            "raw": "h_j(s)",
            "centered_scaled": "htilde_j(s)=(h_j(s)-h_j(s0))/scale_j",
            "scale": "max_{u in S_pool}|h_j(u)-h_j(s0)|",
        },
    }

    def _myopic_successor_dist(state, person):
        posterior_entry, _ = belief[state]
        if tuple_mode and tuple_posteriors.get(state, {}).get(person):
            return tuple_posteriors[state][person].items()
        posterior = _posterior_marginals(posterior_entry)
        return posterior.get(person, {}).items()

    def _compute_myopic_occupancy(policy):
        occupancy = {initial_state: 1.0}
        ordered = sorted(policy, key=lambda state: (len(_evidence(state)), _state_label(state)))
        for state in ordered:
            prob_state = float(occupancy.get(state, 0.0))
            if prob_state <= 0.0:
                continue
            action = policy.get(state)
            if not action or action[0] != "test":
                continue
            person = action[1]
            for outcome, prob in _myopic_successor_dist(state, person):
                prob_f = float(prob)
                if prob_f <= 0.0:
                    continue
                succ = _merge_state(state, person, outcome)
                if len(_evidence(succ)) >= len(I):
                    continue
                occupancy[succ] = occupancy.get(succ, 0.0) + prob_state * prob_f
        return occupancy

    def _compute_myopic_bellman_residuals(values):
        residuals = {}
        action_margins = {}
        for state in list(belief.keys()):
            if len(_evidence(state)) >= len(I):
                residuals[state] = 0.0
                action_margins[state] = 0.0
                continue
            posterior_entry, _ = belief[state]
            posterior = _posterior_marginals(posterior_entry)
            gene_probs = _ensure_gene_posteriors(state, posterior_entry)
            gene_probs = gene_probs or {}
            tested = {person for person, _ in _evidence(state)}
            stop_rhs = sum(
                r_reward(
                    person,
                    posterior,
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
                for person in I
                if person not in tested
            )
            rhs_values = [float(stop_rhs)]
            for person in I:
                if person in tested:
                    continue
                person_probs = posterior.get(person, {})
                p12 = float(person_probs.get(1, 0.0) + person_probs.get(2, 0.0))
                per_gene_p12 = _per_gene_p12_map(gene_probs, person)
                immediate_reward = r_reward_testp(
                    person,
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
                succ_value = 0.0
                for outcome, prob in _myopic_successor_dist(state, person):
                    prob_f = float(prob)
                    if prob_f <= 0.0:
                        continue
                    succ = _merge_state(state, person, outcome)
                    if len(_evidence(succ)) >= len(I):
                        continue
                    succ_value += prob_f * float(values.get(succ, 0.0))
                rhs_values.append(float(immediate_reward) + succ_value)
            rhs_values.sort(reverse=True)
            best_rhs = rhs_values[0] if rhs_values else 0.0
            second_rhs = rhs_values[1] if len(rhs_values) >= 2 else best_rhs
            residuals[state] = max(0.0, best_rhs - float(values.get(state, 0.0)))
            action_margins[state] = max(0.0, best_rhs - second_rhs)
        return residuals, action_margins

    if myopic_adp_active:
        state_pool = tuple(belief.keys()) if exhaustive_bellman else None
        myopic_eval = evaluate_myopic_policy(
            belief=belief,
            individuals=I,
            gen_states=gen_states,
            infer=infer,
            a=a,
            b=b,
            c=c,
            delta=delta,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            belief_gene=belief_gene,
            genes=gene_list if multi_gene else None,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
            state_pool=state_pool,
        )
        myopic_policy_map = dict(myopic_eval.policy)
        myopic_value_map = dict(myopic_eval.values)
        myopic_residuals, myopic_action_margins = _compute_myopic_bellman_residuals(myopic_value_map)
        myopic_adp_diagnostics.update(
            {
                "myopic_state_count": len(myopic_policy_map),
                "myopic_value_count": len(myopic_value_map),
                "root_myopic_value": myopic_eval.root_value,
                "myopic_residual_positive_count": int(
                    sum(1 for value in myopic_residuals.values() if float(value) > 1e-9)
                ),
            }
        )

        if abcd16_direct_active:
            myopic_adp_mode = "abcd16_direct"
            myopic_feature_names = list(abcd16_direct_myopic_features)
            myopic_selected_features = list(myopic_feature_names)
            myopic_adp_variable_order = list(abcd16_direct_myopic_embedding_order)
            myopic_adp_diagnostics.update(
                {
                    "direct_feature_bank": "ABCD16_DIRECT",
                    "selection": abcd16_direct_selection,
                    "myopic_policy_count": int(len(myopic_policy_map)),
                    "myopic_residual_count": int(len(myopic_residuals)),
                    "myopic_stop_indicator_count": int(len(myopic_policy_map)),
                    "missing_residual_for_phi_state_count": int(
                        sum(1 for state in belief.keys() if state not in myopic_residuals)
                    ),
                    "top_k_selection_used": False,
                    "seed_adp_presolve_used": False,
                }
            )
            feature_abs_max = {name: 0.0 for name in myopic_feature_names}
            for state in belief.keys():
                features = build_state_features(
                    state,
                    belief=belief,
                    individuals=I,
                    pedigree=pedigree,
                    genes=gene_list if multi_gene else None,
                    myopic_policy=myopic_policy_map,
                    myopic_values=myopic_value_map,
                    myopic_residuals=myopic_residuals,
                )
                for name in myopic_feature_names:
                    feature_abs_max[name] = max(
                        feature_abs_max[name],
                        abs(float(features.get(name, 0.0) or 0.0)),
                    )
            myopic_adp_zero_column_features = {
                name for name, max_abs in feature_abs_max.items() if max_abs < 1e-12
            }
            myopic_adp_diagnostics["zero_column_features"] = sorted(myopic_adp_zero_column_features)
            myopic_adp_diagnostics["zero_column_coefficients_fixed_to_zero"] = True
        elif myopic_control_variate_active:
            myopic_adp_mode = "control_variate"
        elif myopic_residual_basis_active:
            myopic_adp_mode = "residual_basis"
            base_candidate_features = (
                "frontier_carrier_mass",
                "frontier_carrier_max",
                "frontier_carrier_variance",
                "bridge_depth_mass",
                "descendant_bridge_mass",
                "sibling_breadth",
                "collateral_block_count",
                "myopic_tests",
                "myopic_stops",
                "myopic_bellman_residual",
            )
            if myopic_residual_only_extra and myopic_residual_extra_features:
                candidate_features = tuple(dict.fromkeys(myopic_residual_extra_features))
            else:
                candidate_features = tuple(
                    dict.fromkeys(tuple(base_candidate_features) + myopic_residual_extra_features)
                )
            occupancy = _compute_myopic_occupancy(myopic_policy_map)
            mixed_states = sorted(set(belief.keys()) | set(myopic_policy_map.keys()), key=_state_label)
            uniform_weight = 1.0 / max(1.0, float(len(mixed_states)))
            violated = {state for state in mixed_states if float(myopic_residuals.get(state, 0.0)) > 1e-9}
            violation_weight = 1.0 / max(1.0, float(len(violated)))
            scores = {}
            for state in mixed_states:
                features = build_state_features(
                    state,
                    belief=belief,
                    individuals=I,
                    pedigree=pedigree,
                    genes=gene_list if multi_gene else None,
                    myopic_policy=myopic_policy_map,
                    myopic_values=myopic_value_map,
                    myopic_residuals=myopic_residuals,
                )
                residual = float(myopic_residuals.get(state, 0.0))
                frontier_mass = max(1e-9, float(features.get("frontier_carrier_mass", 0.0)))
                margin = max(0.0, float(myopic_action_margins.get(state, 0.0)))
                mixture_occupancy = (
                    0.50 * float(occupancy.get(state, 0.0))
                    + 0.25 * uniform_weight
                    + 0.25 * (violation_weight if state in violated else 0.0)
                )
                weight = mixture_occupancy * frontier_mass / (1e-6 + margin)
                for name in candidate_features:
                    scores[name] = scores.get(name, 0.0) + weight * residual * float(features.get(name, 0.0))
            ranked = sorted(candidate_features, key=lambda name: (-abs(scores.get(name, 0.0)), name))
            selected = list(ranked[:myopic_residual_top_k])
            for feature_name in myopic_residual_force_features:
                if feature_name in candidate_features and feature_name not in selected:
                    selected.append(feature_name)
            myopic_selected_features = selected
            myopic_feature_names = list(myopic_selected_features)
            myopic_adp_variable_order = list(myopic_feature_names)
            myopic_adp_diagnostics["mixed_state_pool_count"] = int(len(mixed_states))
            myopic_adp_diagnostics["violated_bellman_state_count"] = int(len(violated))
            myopic_adp_diagnostics["extra_features"] = list(myopic_residual_extra_features)
            myopic_adp_diagnostics["forced_features"] = list(myopic_residual_force_features)
            myopic_adp_diagnostics["only_extra_features"] = bool(myopic_residual_only_extra)
            myopic_adp_diagnostics["residual_feature_scores"] = {
                name: float(scores.get(name, 0.0)) for name in ranked
            }
        elif myopic_piecewise_regions_active:
            myopic_adp_mode = "piecewise_regions"
            myopic_feature_names = [
                "myopic_tests",
                "myopic_stops",
                "myopic_adp_disagreement_region",
                "allele_asymmetry_high_region",
                "allele_asymmetry_low_region",
                "bridge_dominated_region",
                "breadth_dominated_region",
                "collateral_active_region",
                "frontier_carrier_mass",
                "descendant_bridge_mass",
            ]
            myopic_selected_features = list(myopic_feature_names)
            myopic_adp_variable_order = list(myopic_feature_names)

        if not myopic_adp_variable_order:
            myopic_adp_variable_order = list(myopic_feature_names)
        for feature_name in myopic_adp_variable_order:
            var_lb = 0.0 if feature_name in myopic_adp_zero_column_features else -GRB.INFINITY
            var_ub = 0.0 if feature_name in myopic_adp_zero_column_features else GRB.INFINITY
            myopic_adp_vars[feature_name] = master.addVar(
                lb=var_lb,
                ub=var_ub,
                name=f"myopic_adp_{myopic_adp_mode}_{feature_name}",
            )

    if abcd16_direct_active:
        regime_gates = regime_parameter_gates(
            genes=gene_list if multi_gene else None,
            a_gene=a_gene,
            b_gene=b_gene,
            delta_gene=delta_gene,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
        )

        def _direct16_regime_raw_features(state):
            return build_state_features(
                state,
                belief=belief,
                individuals=I,
                pedigree=pedigree,
                genes=gene_list if multi_gene else None,
                regime_gates=regime_gates,
                feature_semantics=regime_feature_semantics,
            )

        candidate_feature_names = tuple(abcd16_direct_regime_features)
        variable_feature_names = tuple(ABCD16_DIRECT_REGIME_EMBEDDING_ORDER)
        state_pool = tuple(sorted(belief.keys(), key=_deterministic_state_sort_key))
        root_raw_features = _direct16_regime_raw_features(initial_state)
        feature_root_values = {
            name: float(root_raw_features.get(name, 0.0) or 0.0)
            for name in candidate_feature_names
        }
        feature_scales = {}
        skipped_low_scale = []
        for name in candidate_feature_names:
            root_value = feature_root_values[name]
            max_abs = 0.0
            for state in state_pool:
                value = float(_direct16_regime_raw_features(state).get(name, 0.0) or 0.0)
                max_abs = max(max_abs, abs(value - root_value))
            if max_abs < 1e-12:
                skipped_low_scale.append(name)
                feature_scales[name] = 1.0
            else:
                feature_scales[name] = float(max_abs)
        regime_residual_selected_features = list(candidate_feature_names)
        regime_residual_feature_names = list(regime_residual_selected_features)
        regime_residual_star = {
            "enabled": True,
            "mode": "abcd16_direct",
            "selector": "fixed_all_16",
            "anchor": "none_no_seed",
            "feature_bank": regime_feature_bank,
            "feature_semantics": regime_feature_semantics,
            "feature_names": list(candidate_feature_names),
            "selected_features": list(regime_residual_selected_features),
            "selected_v1_base_features": [
                "collateral_active_parent_pair_block_count_honest",
                "all_untested_carrier_variance_honest",
            ],
            "selected_v2_features": [
                "allele_asymmetry_high_gene_GeneA_carrier_depth_mass_honest",
            ],
            "feature_root_values": dict(feature_root_values),
            "feature_scales": dict(feature_scales),
            "regime_gates": dict(regime_gates),
            "coefficients": {},
            "diagnostics": {
                "enabled": True,
                "mode": "abcd16_direct",
                "selector": "fixed_all_16",
                "anchor": "none_no_seed",
                "feature_bank": regime_feature_bank,
                "feature_semantics": regime_feature_semantics,
                "candidate_feature_count": int(len(candidate_feature_names)),
                "selected_feature_count": int(len(regime_residual_selected_features)),
                "variable_feature_order": list(variable_feature_names),
                "state_pool_count": int(len(state_pool)),
                "skipped_low_scale_features": list(skipped_low_scale),
                "top_k_selection_used": False,
                "seed_adp_presolve_used": False,
                "active_row_export_used": False,
                "feature_selection_from_seed_used": False,
                "seed_duals_used": False,
                "seed_phi_values_used": False,
                "zero_extra_coefficients_allowed": True,
                "extra_coefficients_forced_nonzero": False,
                "coefficient_bounds_match_selected": True,
                "forbidden_inputs_used": False,
            },
        }
        regime_residual_diagnostics.update(regime_residual_star["diagnostics"])
        regime_zero_column_features = set(skipped_low_scale)
        for feature_name in variable_feature_names:
            var_lb = 0.0 if feature_name in regime_zero_column_features else -GRB.INFINITY
            var_ub = 0.0 if feature_name in regime_zero_column_features else GRB.INFINITY
            regime_residual_vars[feature_name] = master.addVar(
                lb=var_lb,
                ub=var_ub,
                name=f"abcd16_direct_{feature_name}",
            )
    elif regime_residual_active and regime_residual_v2_active and regime_residual_v2_payload_path:
        payload_path = Path(regime_residual_v2_payload_path).expanduser()
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid V2 regime residual payload at {payload_path}: expected JSON object.")
        selected_payload = payload.get("selected_features", ())
        if not isinstance(selected_payload, list) or not selected_payload:
            raise ValueError(
                f"Invalid V2 regime residual payload at {payload_path}: selected_features must be nonempty."
            )
        payload_semantics_raw = payload.get("feature_semantics")
        if not payload_semantics_raw:
            raise ValueError(
                f"Invalid V2 regime residual payload at {payload_path}: feature_semantics is required."
            )
        payload_semantics = resolve_feature_semantics(payload_semantics_raw)
        if payload_semantics != regime_feature_semantics:
            raise ValueError(
                "V2 regime residual payload feature_semantics="
                f"{payload_semantics!r} does not match GAUGED_REGIME_FEATURE_SEMANTICS="
                f"{regime_feature_semantics!r}."
            )
        payload_bank_raw = payload.get("feature_bank")
        if not payload_bank_raw:
            raise ValueError(
                f"Invalid V2 regime residual payload at {payload_path}: feature_bank is required."
            )
        payload_bank = resolve_feature_bank(payload_bank_raw, require=True)
        if payload_bank != regime_feature_bank:
            raise ValueError(
                "V2 regime residual payload feature_bank="
                f"{payload_bank!r} does not match GAUGED_REGIME_FEATURE_BANK="
                f"{regime_feature_bank!r}."
            )
        feature_names_payload = payload.get("feature_names")
        if not isinstance(feature_names_payload, list):
            feature_names_payload = list(
                regime_residual_v2_candidate_features(
                    gene_list if multi_gene else None,
                    feature_bank=regime_feature_bank,
                )
            )
        regime_residual_selected_features = [str(name) for name in selected_payload]
        regime_residual_feature_names = list(regime_residual_selected_features)
        diagnostics_payload = payload.get("diagnostics", {})
        if not isinstance(diagnostics_payload, Mapping):
            diagnostics_payload = {}
        diagnostics = {
            "enabled": True,
            "mode": "gauged_regime_residual_v2_state_pool",
            "selector": regime_residual_selector,
            "anchor": regime_residual_anchor,
            "payload_path": str(payload_path),
            "payload_loaded": True,
            "feature_bank": regime_feature_bank,
            "feature_semantics": regime_feature_semantics,
            "top_k": regime_residual_v2_top_k,
            "min_signature_ratio": regime_residual_v2_min_signature_ratio,
            "incremental_tol": regime_residual_v2_incremental_tol,
            **dict(diagnostics_payload),
        }
        regime_residual_star = {
            "enabled": True,
            "mode": "gauged_regime_residual_v2_state_pool",
            "selector": regime_residual_selector,
            "anchor": regime_residual_anchor,
            "feature_bank": regime_feature_bank,
            "feature_semantics": regime_feature_semantics,
            "feature_names": [str(name) for name in feature_names_payload],
            "selected_features": list(regime_residual_selected_features),
            "selected_v1_base_features": list(payload.get("selected_v1_base_features", ())),
            "selected_v2_features": list(payload.get("selected_v2_features", ())),
            "feature_root_values": dict(payload.get("feature_root_values", {})),
            "feature_scales": dict(payload.get("feature_scales", {})),
            "regime_gates": dict(payload.get("regime_gates", {})),
            "coefficients": {},
            "diagnostics": diagnostics,
        }
        regime_residual_diagnostics.update(diagnostics)
        for feature_name in regime_residual_feature_names:
            regime_residual_vars[feature_name] = master.addVar(
                lb=-GRB.INFINITY,
                name=f"gauged_regime_residual_v2_{feature_name}",
            )
    elif regime_residual_active:
        regime_gates = regime_parameter_gates(
            genes=gene_list if multi_gene else None,
            a_gene=a_gene,
            b_gene=b_gene,
            delta_gene=delta_gene,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
        )

        def _regime_raw_features(state):
            return build_state_features(
                state,
                belief=belief,
                individuals=I,
                pedigree=pedigree,
                genes=gene_list if multi_gene else None,
                regime_gates=regime_gates,
                feature_semantics=regime_feature_semantics,
            )

        candidate_feature_names = regime_residual_candidate_features(regime_feature_bank)
        state_pool = tuple(sorted(belief.keys(), key=_deterministic_state_sort_key))
        root_raw_features = _regime_raw_features(initial_state)
        feature_root_values = {
            name: float(root_raw_features.get(name, 0.0) or 0.0)
            for name in candidate_feature_names
        }
        feature_scales = {}
        skipped_low_scale = []
        for name in candidate_feature_names:
            root_value = feature_root_values[name]
            max_abs = 0.0
            for state in state_pool:
                value = float(_regime_raw_features(state).get(name, 0.0) or 0.0)
                max_abs = max(max_abs, abs(value - root_value))
            if max_abs < 1e-12:
                skipped_low_scale.append(name)
                continue
            feature_scales[name] = float(max_abs)

        def _htilde_value(state, name):
            scale = feature_scales.get(name)
            if scale is None or abs(scale) < 1e-12:
                return 0.0
            value = float(_regime_raw_features(state).get(name, 0.0) or 0.0)
            return (value - feature_root_values[name]) / float(scale)

        def _root_action_successors_for_regime(person):
            successors = []
            for outcome, prob in _myopic_successor_dist(initial_state, person):
                prob_f = float(prob)
                if prob_f <= 0.0:
                    continue
                succ = _merge_state(initial_state, person, outcome)
                if succ not in belief:
                    continue
                successors.append((succ, prob_f))
            return successors

        root_action_order = []
        signature_vectors = {name: [] for name in feature_scales}
        for person in I:
            successors = _root_action_successors_for_regime(person)
            if not successors:
                continue
            root_action_order.append(str(person))
            for name in feature_scales:
                expected = sum(prob * _htilde_value(succ, name) for succ, prob in successors)
                signature_vectors[name].append(float(_htilde_value(initial_state, name) - expected))

        legacy_vectors = []
        if root_action_order:
            legacy_vectors.append([1.0 for _ in root_action_order])
        w_basis_count = int(len(I) * len(gen_states) * (len(gene_list) if per_gene_phi_active else 1))
        legacy_zero_vectors = [[0.0 for _ in root_action_order] for _ in range(w_basis_count)]
        selected, _selected_vectors, selection_diagnostics = select_signature_features(
            signature_vectors,
            legacy_vectors=tuple(legacy_vectors) + tuple(legacy_zero_vectors),
            top_k=regime_residual_top_k,
            min_ratio=regime_residual_min_signature_ratio,
            incremental_tol=regime_residual_incremental_tol,
        )
        regime_residual_selected_features = list(selected)
        regime_residual_feature_names = list(selected)
        regime_residual_star = {
            "enabled": True,
            "mode": "gauged_regime_residual_v1",
            "feature_bank": regime_feature_bank,
            "feature_semantics": regime_feature_semantics,
            "feature_names": list(candidate_feature_names),
            "selected_features": list(regime_residual_selected_features),
            "feature_root_values": dict(feature_root_values),
            "feature_scales": dict(feature_scales),
            "regime_gates": dict(regime_gates),
            "coefficients": {},
            "diagnostics": {
                "candidate_feature_count": int(len(candidate_feature_names)),
                "state_pool_count": int(len(state_pool)),
                "selector": "root_test",
                "anchor": "self",
                "feature_bank": regime_feature_bank,
                "feature_semantics": regime_feature_semantics,
                "root_action_order": list(root_action_order),
                "signature_by_root_action": {
                    name: {
                        str(action): float(value)
                        for action, value in zip(root_action_order, signature_vectors.get(name, ()))
                    }
                    for name in signature_vectors
                },
                "skipped_low_scale_features": list(skipped_low_scale),
                "forbidden_inputs_used": False,
                "legacy_signature_basis_count": int(1 + w_basis_count if root_action_order else w_basis_count),
                "legacy_w_signature_basis_count": int(w_basis_count),
                **selection_diagnostics,
            },
        }
        regime_residual_diagnostics.update(regime_residual_star["diagnostics"])
        for feature_name in regime_residual_feature_names:
            regime_residual_vars[feature_name] = master.addVar(
                lb=-GRB.INFINITY,
                name=f"gauged_regime_residual_v1_{feature_name}",
            )

    if oracle_adp_active:
        if not isinstance(oracle_adp_payload, Mapping):
            oracle_adp_payload = {
                "exact_values": {state: 0.0 for state in belief.keys()},
                "policy_exact": {},
                "diagnostic_fallback": True,
            }
            oracle_adp_diagnostics.update(
                {
                    "payload_available": False,
                    "diagnostic_fallback": True,
                    "fallback_reason": "missing_exact_dp_payload",
                }
            )
        oracle_adp_star = build_oracle_feature_payload(
            exact_values=oracle_adp_payload.get("exact_values", {}),
            policy_exact=oracle_adp_payload.get("policy_exact", {}),
            state_pool=tuple(belief.keys()),
            mode=oracle_adp_mode,
            top_k=oracle_adp_top_k,
        )
        oracle_adp_diagnostics.update(
            {
                "payload_available": not bool(oracle_adp_payload.get("diagnostic_fallback")),
                "feature_names": list(oracle_adp_star.get("feature_names", ())),
                "selected_state_count": len(oracle_adp_star.get("selected_states", ()) or ()),
                "exact_value_count": len(oracle_adp_star.get("exact_values", {}) or {}),
                "policy_state_count": len(oracle_adp_star.get("policy_exact", {}) or {}),
                "root_exact_value": oracle_adp_star.get("root_exact_value"),
                "diagnostic_fallback": bool(oracle_adp_payload.get("diagnostic_fallback")),
            }
        )
        for feature_name in oracle_adp_star.get("feature_names", ()):
            oracle_adp_vars[feature_name] = master.addVar(
                lb=-GRB.INFINITY,
                name=f"oracle_adp_{oracle_adp_mode}_{feature_name}",
            )

    def _fix_var_to_zero(var):
        if var is None:
            return
        var.LB = 0.0
        var.UB = 0.0

    def _fix_var_collection_to_zero(container):
        if container is None:
            return
        if hasattr(container, "values"):
            for var in container.values():
                _fix_var_to_zero(var)
            return
        for var in container:
            _fix_var_to_zero(var)

    if oracle_only_fixed_phi_active:
        _fix_var_collection_to_zero(W_var)
        _fix_var_collection_to_zero(W_gene_var)
        _fix_var_collection_to_zero(theta_stage_base_vars)
        _fix_var_collection_to_zero(theta_vars)
        _fix_var_to_zero(theta_var)
        _fix_var_collection_to_zero(aaub_u_vars)
        _fix_var_collection_to_zero(aaub_v_vars)
        _fix_var_collection_to_zero(W_edge_vars)
        _fix_var_collection_to_zero(W_trio_vars)
        _fix_var_collection_to_zero(myopic_adp_vars)
        _fix_var_collection_to_zero(oracle_adp_vars)
        oracle_adp_diagnostics["legacy_residual_fixed_to_zero"] = True

    def _myopic_adp_expr_for_state(state):
        if not myopic_adp_active:
            return gp.LinExpr(0.0)
        expr = gp.LinExpr()
        if myopic_control_variate_active:
            expr.add(float(myopic_value_map.get(state, 0.0)))
        if myopic_adp_vars:
            features = build_state_features(
                state,
                belief=belief,
                individuals=I,
                pedigree=pedigree,
                genes=gene_list if multi_gene else None,
                myopic_policy=myopic_policy_map,
                myopic_values=myopic_value_map,
                myopic_residuals=myopic_residuals,
            )
            for feature_name, var in myopic_adp_vars.items():
                value = float(features.get(feature_name, 0.0))
                if abs(value) > 0.0:
                    expr.add(value * var)
        return expr

    def _current_myopic_adp_star():
        if not myopic_adp_active:
            return None
        return {
            "enabled": True,
            "mode": myopic_adp_mode,
            "feature_names": list(myopic_feature_names),
            "selected_features": list(myopic_selected_features),
            "coefficients": {
                name: _safe_var_value(var)
                for name, var in myopic_adp_vars.items()
            },
            "myopic_policy": myopic_policy_map,
            "myopic_values": myopic_value_map,
            "myopic_residuals": myopic_residuals,
            "myopic_action_margins": myopic_action_margins,
            "root_myopic_value": myopic_eval.root_value if myopic_eval else None,
            "diagnostics": dict(myopic_adp_diagnostics),
        }

    def _regime_residual_expr_for_state(state):
        if not regime_residual_active or not isinstance(regime_residual_star, Mapping):
            return gp.LinExpr(0.0)
        expr = gp.LinExpr()
        values = regime_residual_feature_values(
            state,
            regime_residual_star,
            belief=belief,
            individuals=I,
            pedigree=pedigree,
            genes=gene_list if multi_gene else None,
        )
        for feature_name, var in regime_residual_vars.items():
            value = float(values.get(feature_name, 0.0) or 0.0)
            if abs(value) > 0.0:
                expr.add(value * var)
        return expr

    def _current_regime_residual_star():
        if not regime_residual_active or not isinstance(regime_residual_star, Mapping):
            return None
        current = dict(regime_residual_star)
        current["coefficients"] = {
            name: _safe_var_value(var)
            for name, var in regime_residual_vars.items()
        }
        diagnostics = dict(regime_residual_diagnostics)
        diagnostics.update(current.get("diagnostics", {}) or {})
        current["diagnostics"] = diagnostics
        return current

    def _oracle_adp_expr_for_state(state):
        if not oracle_adp_active or not isinstance(oracle_adp_star, Mapping):
            return gp.LinExpr(0.0)
        expr = gp.LinExpr()
        if oracle_adp_star.get("mode") == "exact_value_fixed":
            expr.add(float(oracle_adp_term_value(state, oracle_adp_star)))
        if oracle_adp_vars:
            features = oracle_feature_values(state, oracle_adp_star)
            for feature_name, var in oracle_adp_vars.items():
                value = float(features.get(feature_name, 0.0))
                if abs(value) > 0.0:
                    expr.add(value * var)
        return expr

    def _current_oracle_adp_star():
        if not oracle_adp_active or not isinstance(oracle_adp_star, Mapping):
            return None
        current = dict(oracle_adp_star)
        current["coefficients"] = {
            name: _safe_var_value(var)
            for name, var in oracle_adp_vars.items()
        }
        diagnostics = dict(oracle_adp_diagnostics)
        diagnostics.update(current.get("diagnostics", {}) or {})
        diagnostics["gauge_constraints_added"] = list(oracle_gauge_constraint_names)
        current["diagnostics"] = diagnostics
        current["plumbing_mode"] = oracle_plumbing_mode
        current["oracle_plumbing_mode"] = oracle_plumbing_mode
        return current

    def get_Phi(state: frozenset):
        # ---------------- sanity checks -----------------
        assert state in belief, f"Belief missing for state {state}"
        evidence = _evidence(state)
        if evidence and not all(isinstance(elt, tuple) and len(elt) == 2
                                for elt in evidence):
            raise AssertionError("State must be a frozenset of (person,g) pairs")

        # ---------------- create Φ(s) once with optional canonicalization --------------
        if state not in Phi:
            if value_canon_mode == 'identity' or not role_groups:
                ckey = ("RAW", tuple(sorted(evidence)))
            elif value_canon_mode == 'role':
                ckey = canonicalize_state(evidence, role_groups, gen_states, param_sig_fn=None)
            elif value_canon_mode == 'cohort':
                sig_fn = reward_signature_fn(a,b,c,delta)
                ckey = canonicalize_state(evidence, role_groups, gen_states, param_sig_fn=sig_fn)
            else:
                raise ValueError(f"Unknown value_canon_mode: {value_canon_mode}")

            if ckey in Phi_canon:
                Phi[state] = Phi_canon[ckey]
            else:
                label     = _state_label(state)
                Phi_state = master.addVar(lb=-GRB.INFINITY, name=f"Phi_{label}")
                Phi[state] = Phi_state
                Phi_canon[ckey] = Phi_state
            _register_projection(state)

            # ---- 1. build the linear expression Σ W_i(g) -----------------
            tested_obs = {person: g for person, g in evidence}      # e.g. {'Father':2}
            posterior_entry, z_s = belief[state]
            p_s = _posterior_marginals(posterior_entry)
            gene_probs = _ensure_gene_posteriors(state, posterior_entry)
            gene_probs = gene_probs or {}
            tuple_probs_state = tuple_posteriors.get(state, {}) if tuple_mode else {}
            aaub_state_expr = _aaub_expr_for_state(tested_obs=tested_obs, posterior_entry=posterior_entry)
            edge_state_expr = _edge_expr_for_state(tested_obs=tested_obs, posterior_entry=posterior_entry, state=state)
            trio_state_expr = _trio_expr_for_state(tested_obs=tested_obs, posterior_entry=posterior_entry, state=state)
            myopic_adp_state_expr = _myopic_adp_expr_for_state(state)
            regime_residual_state_expr = _regime_residual_expr_for_state(state)
            oracle_adp_state_expr = _oracle_adp_expr_for_state(state)
            if tuple_mode:
                def _tuple_pmfs_valid(pmfs):
                    if not pmfs:
                        return False
                    if not gene_list:
                        return True
                    for _, dist in pmfs.items():
                        if not dist:
                            return False
                        sample_key = next(iter(dist))
                        if not isinstance(sample_key, tuple) or len(sample_key) != len(gene_list):
                            return False
                    return True

                if isinstance(posterior_entry, InferenceResult) and posterior_entry.has_tuple_pmfs():
                    tuple_probs_state = posterior_entry.get_tuple_pmfs()
                    _store_tuple_posteriors(state, tuple_probs_state)
                if not _tuple_pmfs_valid(tuple_probs_state):
                    evidence_dict = dict(evidence)
                    inferred = inf_cache.get(evidence_dict)
                    if isinstance(inferred, InferenceResult) and inferred.has_tuple_pmfs():
                        tuple_probs_state = inferred.get_tuple_pmfs()
                        _store_tuple_posteriors(state, tuple_probs_state)
                    else:
                        raise RuntimeError(
                            "CRITICAL FAILURE: Missing tuple PMFs in tuple mode. "
                            f"State={state!r} evidence={evidence_dict!r}. "
                            "THIS RUN IS INVALID — STOP AND FIX BEFORE CONTINUING."
                        )

            if oracle_only_fixed_phi_active:
                oracle_residual_expr_by_state[state] = gp.LinExpr(0.0)
                master.addConstr(
                    Phi_state == oracle_adp_state_expr,
                    name=f"phi_oracle_only_{label}",
                )
            elif per_gene_phi_active:
                projection = _register_projection(state)
                phi_gene_sum = gp.LinExpr()
                for gene in gene_list:
                    proj_state = projection.get(gene, frozenset()) if projection else frozenset()
                    phi_gene_map = Phi_gene.setdefault(gene, {})
                    if proj_state not in phi_gene_map:
                        proj_label = _projection_label(gene, proj_state)
                        phi_gene_var = master.addVar(lb=-GRB.INFINITY, name=f"Phi_{proj_label}")
                        phi_gene_map[proj_state] = phi_gene_var
                        F_gene = gp.LinExpr()
                        tested_proj = dict(proj_state)
                        for person in I:
                            if person in tested_proj:
                                F_gene.add(_get_w_gene(gene, person, tested_proj[person]))
                                continue
                            dist = gene_probs.get(gene, {}).get(person)
                            if dist:
                                for g_val, prob in dist.items():
                                    if prob <= 0.0:
                                        continue
                                    F_gene.add(prob * _get_w_gene(gene, person, g_val))
                        master.addConstr(
                            phi_gene_var == _theta_for_state_gene(state, gene) + F_gene,
                            name=f"phi_def_{proj_label}",
                        )
                    phi_gene_sum.add(phi_gene_map[proj_state])
                legacy_residual_expr = (
                    phi_gene_sum
                    + _theta_model_correction_expr(state)
                    + aaub_state_expr
                    + edge_state_expr
                    + trio_state_expr
                    + myopic_adp_state_expr
                    + regime_residual_state_expr
                )
                oracle_residual_expr_by_state[state] = legacy_residual_expr
                master.addConstr(
                    Phi_state
                    == legacy_residual_expr + oracle_adp_state_expr,
                    name=f"phi_sum_{label}",
                )
            else:
                F_s = gp.LinExpr()
                for i in I:
                    if i in tested_obs:              # we observed genotype g*=tested_obs[i]
                        g_star = tested_obs[i]
                        F_s.add(_get_w(i, g_star))    # z_{i,g*}=1 so coeff is 1
                    else:                            # still untested → use prior probs
                        if tuple_mode and tuple_probs_state.get(i):
                            for outcome, prob in tuple_probs_state[i].items():
                                if prob <= 0.0:
                                    continue
                                F_s.add(prob * _get_w(i, outcome))
                        else:
                            for g in gen_states:
                                weight = p_s[i][g]
                                if weight <= 0.0:
                                    continue
                                if tuple_mode:
                                    normalized = _normalize_outcome(g)
                                    F_s.add(weight * _get_w(i, normalized))
                                else:
                                    F_s.add(weight * _get_w(i, g))
                legacy_residual_expr = (
                    _theta_for_state(state)
                    + _theta_model_correction_expr(state)
                    + F_s
                    + aaub_state_expr
                    + edge_state_expr
                    + trio_state_expr
                    + myopic_adp_state_expr
                    + regime_residual_state_expr
                )
                oracle_residual_expr_by_state[state] = legacy_residual_expr
                master.addConstr(
                    Phi_state
                    == legacy_residual_expr + oracle_adp_state_expr,
                    name=f"phi_def_{label}",
                )

            # ---- 2. stopping‑reward lower bound ---------------------------
            Rstop = sum(
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
                for k in I if k not in tested_obs        # only those still untested
            )
            stop_constraint = master.addConstr(Phi_state >= Rstop, name=f"stop_def_{label}")
            _register_bellman_row_constraint(
                stop_constraint,
                row_type="stop",
                state=state,
                action="stop",
                rhs_const=Rstop,
                source="phi_stop_bound",
            )
    
        return Phi[state]


    # 3) Initial state
    s0 = initial_state
    get_Phi(s0)

    # 4) Seed cuts
    # 4a) stop immediately
    # master.addConstr(get_Phi(s0) >= 0, name="init_stop")
    
    # ------------------------------------------------------------------
    # 4b) one‑step “test‑then‑stop” seeds  (only from the root state)
    # ------------------------------------------------------------------
    p_s0, z_s0 = belief[s0]
    gene_probs_root = _ensure_gene_posteriors(s0, p_s0)
    root_tuple_pmfs = {}
    if tuple_mode:
        root_result = inf_cache.get({})
        if hasattr(root_result, 'get_tuple_pmfs'):
            root_tuple_pmfs = root_result.get_tuple_pmfs()
        _store_tuple_posteriors(s0, root_tuple_pmfs)
    # ----------  NEW: root‑stop cut  ---------------------------------
    Rstop_root = sum(
        r_reward(
            k,
            p_s0,
            a,
            b,
            c,
            delta,
            per_gene_probs=gene_probs_root,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
        )
        for k in I                   # every individual still untested
    )
    if verbose:
        print(f"[SEED] Adding root-stop constraint: Φ(root) >= {Rstop_root:.6f}")
    root_stop_constraint = master.addConstr(get_Phi(s0) >= Rstop_root,
                                            name="root_stop")          # <-- add this line
    _register_bellman_row_constraint(
        root_stop_constraint,
        row_type="stop",
        state=s0,
        action="stop",
        rhs_const=Rstop_root,
        source="seed_root_stop",
    )

    seed_root_init_constraint_names = []
    seed_stage1_constraint_names = []
    stage1_seed_states = set()
    stage1_seed_state_count = 0
    stage2_clinical_seed_states = set()
    stage2_clinical_seed_state_count = 0

    def _seed_test_then_stop_constraints(
        *,
        source_state,
        source_posteriors,
        source_gene_probs,
        source_tuple_pmfs,
        name_prefix,
        add_tuple_strengthening,
        restrict_people=None,
    ):
        nonlocal stage1_seed_state_count
        tested_in_source = {person for person, _ in _evidence(source_state)}
        candidate_people = list(restrict_people) if restrict_people is not None else list(I)
        for person in candidate_people:
            if person in tested_in_source:
                continue
            p12 = source_posteriors[person].get(1, 0.0) + source_posteriors[person].get(2, 0.0)
            per_gene_p12 = _per_gene_p12_map(source_gene_probs, person)
            immediate_reward = r_reward_testp(
                person,
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

            if tuple_mode and source_tuple_pmfs.get(person):
                all_succ_candidates = [
                    (outcome, prob)
                    for outcome, prob in source_tuple_pmfs[person].items()
                    if _keep_successor_prob(prob)
                ]
                topk_candidates = (
                    select_tuple_successors(source_tuple_pmfs[person], K=topk_successors, pmin=successor_pmin)
                    if add_tuple_strengthening and topk_successors > 0
                    else []
                )
                top1_candidate = (
                    select_tuple_successors(source_tuple_pmfs[person], K=1, pmin=successor_pmin)
                    if add_tuple_strengthening
                    else []
                )
            else:
                if exhaustive_bellman:
                    all_succ_candidates = [
                        (outcome, prob) for outcome, prob in source_posteriors[person].items() if prob > 0.0
                    ]
                else:
                    all_succ_candidates = [(outcome, source_posteriors[person][outcome]) for outcome in gen_states]
                topk_candidates = []
                top1_candidate = []

            successor_terms = []
            successor_entries = []
            for outcome, prob in all_succ_candidates:
                if prob <= successor_prob_cut:
                    continue
                evidence = dict(_evidence(source_state))
                evidence[person] = outcome
                result = inf_cache.get(evidence)
                if not isinstance(result, InferenceResult):
                    result = InferenceResult(
                        result,
                        gene_order=gene_list if multi_gene else ("gene",),
                        gen_states=gen_states,
                    )
                succ_state = _merge_state(source_state, person, outcome)
                _store_belief(succ_state, result)
                _ensure_gene_posteriors(succ_state, result)
                if tuple_mode and result.has_tuple_pmfs():
                    _store_tuple_posteriors(succ_state, result.get_tuple_pmfs())
                if len(_evidence(source_state)) == 0 and len(_evidence(succ_state)) == 1:
                    stage1_seed_states.add(succ_state)
                successor_terms.append(prob * get_Phi(succ_state))
                successor_entries.append((succ_state, float(prob)))

            if name_prefix == "init_test":
                constraint_name = f"{name_prefix}_{person}"
                seed_root_init_constraint_names.append(constraint_name)
            else:
                source_label = _state_label(source_state)
                stage1_seed_state_count = max(stage1_seed_state_count, len(stage1_seed_states))
                source_digest = hashlib.sha1(source_label.encode("utf-8")).hexdigest()[:10]
                constraint_name = f"{name_prefix}_{source_digest}_{person}"
                seed_stage1_constraint_names.append(constraint_name)

            constraint = master.addConstr(
                get_Phi(source_state) >= immediate_reward + gp.quicksum(successor_terms),
                name=constraint_name,
            )
            _register_bellman_row_constraint(
                constraint,
                row_type="test",
                state=source_state,
                action=f"test:{person}",
                person_tested=person,
                immediate_reward=immediate_reward,
                successors=successor_entries,
                source="seed_root_test" if name_prefix == "init_test" else name_prefix,
            )

            if add_tuple_strengthening and tuple_mode and topk_candidates:
                succ_terms_topk = []
                succ_entries_topk = []
                for outcome, prob in topk_candidates:
                    if prob <= successor_prob_cut:
                        continue
                    succ_state = _merge_state(source_state, person, outcome)
                    succ_terms_topk.append(prob * get_Phi(succ_state))
                    succ_entries_topk.append((succ_state, float(prob)))
                topk_name = f"{name_prefix}_topk_{person}" if name_prefix == "init_test" else f"{constraint_name}_topk"
                topk_constraint = master.addConstr(
                    get_Phi(source_state) >= immediate_reward + gp.quicksum(succ_terms_topk),
                    name=topk_name,
                )
                _register_bellman_row_constraint(
                    topk_constraint,
                    row_type="test",
                    state=source_state,
                    action=f"test:{person}",
                    person_tested=person,
                    immediate_reward=immediate_reward,
                    successors=succ_entries_topk,
                    source=f"{name_prefix}_topk",
                    truncated=True,
                )
            if add_tuple_strengthening and tuple_mode and top1_candidate:
                succ_terms_top1 = []
                succ_entries_top1 = []
                for outcome, prob in top1_candidate:
                    if prob <= successor_prob_cut:
                        continue
                    succ_state = _merge_state(source_state, person, outcome)
                    succ_terms_top1.append(prob * get_Phi(succ_state))
                    succ_entries_top1.append((succ_state, float(prob)))
                top1_name = f"{name_prefix}_top1_{person}" if name_prefix == "init_test" else f"{constraint_name}_top1"
                top1_constraint = master.addConstr(
                    get_Phi(source_state) >= immediate_reward + gp.quicksum(succ_terms_top1),
                    name=top1_name,
                )
                _register_bellman_row_constraint(
                    top1_constraint,
                    row_type="test",
                    state=source_state,
                    action=f"test:{person}",
                    person_tested=person,
                    immediate_reward=immediate_reward,
                    successors=succ_entries_top1,
                    source=f"{name_prefix}_top1",
                    truncated=True,
                )

            if verbose:
                master.update()
                print(
                    f"[SEED] Added {constraint.ConstrName}: "
                    f"Φ({_state_label(source_state)}) >= {immediate_reward:.6f} + Σ p(g)·Φ(succ)"
                )

    _seed_test_then_stop_constraints(
        source_state=s0,
        source_posteriors=p_s0,
        source_gene_probs=gene_probs_root,
        source_tuple_pmfs=root_tuple_pmfs,
        name_prefix="init_test",
        add_tuple_strengthening=not tuple_strengthening_disabled_active,
    )

    if seed_stage1_enabled:
        stage1_states_sorted = sorted(stage1_seed_states, key=_state_label)
        stage1_seed_state_count = len(stage1_states_sorted)
        for stage1_state in stage1_states_sorted:
            stage1_posterior, _ = belief[stage1_state]
            stage1_posteriors = _posterior_marginals(stage1_posterior)
            stage1_gene_probs = _ensure_gene_posteriors(stage1_state, stage1_posterior)
            stage1_tuple_pmfs = tuple_posteriors.get(stage1_state, {}) if tuple_mode else {}
            _seed_test_then_stop_constraints(
                source_state=stage1_state,
                source_posteriors=stage1_posteriors,
                source_gene_probs=stage1_gene_probs or {},
                source_tuple_pmfs=stage1_tuple_pmfs,
                name_prefix="seed_stage1_test",
                add_tuple_strengthening=False,
            )
        if seed_clinical_stage2_enabled:
            for stage1_state in stage1_states_sorted:
                stage1_posterior, _ = belief[stage1_state]
                stage1_posteriors = _posterior_marginals(stage1_posterior)
                stage1_gene_probs = _ensure_gene_posteriors(stage1_state, stage1_posterior)
                stage1_tuple_pmfs = tuple_posteriors.get(stage1_state, {}) if tuple_mode else {}
                frontier_person = _clinical_frontier_person(stage1_state)
                if frontier_person is None:
                    continue
                _seed_test_then_stop_constraints(
                    source_state=stage1_state,
                    source_posteriors=stage1_posteriors,
                    source_gene_probs=stage1_gene_probs or {},
                    source_tuple_pmfs=stage1_tuple_pmfs,
                    name_prefix="seed_clinical_stage1_test",
                    add_tuple_strengthening=False,
                    restrict_people=[frontier_person],
                )
                if tuple_mode and stage1_tuple_pmfs.get(frontier_person):
                    top1_candidates = select_tuple_successors(stage1_tuple_pmfs[frontier_person], K=1, pmin=0.0)
                else:
                    frontier_dist = stage1_posteriors.get(frontier_person, {})
                    top1_candidates = sorted(
                        (
                            (outcome, prob)
                            for outcome, prob in frontier_dist.items()
                            if prob > 0.0
                        ),
                        key=lambda item: (-float(item[1]), _outcome_label(item[0])),
                    )[:1]
                for outcome, prob in top1_candidates:
                    if prob <= 0.0:
                        continue
                    succ_state = _ensure_successor_belief(stage1_state, frontier_person, outcome)
                    stage2_clinical_seed_states.add(succ_state)

            stage2_clinical_seed_state_count = len(stage2_clinical_seed_states)
            for stage2_state in sorted(stage2_clinical_seed_states, key=_state_label):
                stage2_posterior, _ = belief[stage2_state]
                stage2_posteriors = _posterior_marginals(stage2_posterior)
                stage2_gene_probs = _ensure_gene_posteriors(stage2_state, stage2_posterior)
                stage2_tuple_pmfs = tuple_posteriors.get(stage2_state, {}) if tuple_mode else {}
                frontier_person = _clinical_frontier_person(stage2_state)
                if frontier_person is None:
                    continue
                _seed_test_then_stop_constraints(
                    source_state=stage2_state,
                    source_posteriors=stage2_posteriors,
                    source_gene_probs=stage2_gene_probs or {},
                    source_tuple_pmfs=stage2_tuple_pmfs,
                    name_prefix="seed_clinical_stage2_test",
                    add_tuple_strengthening=False,
                    restrict_people=[frontier_person],
                )


    if exhaustive_bellman:
        if verbose:
            print(f"[config] Exhaustive Bellman enabled: materializing Φ for {len(belief)} states.")
        for state in list(belief.keys()):
            get_Phi(state)

    def _root_successor_weight(state):
        evidence = tuple(_evidence(state))
        if len(evidence) != 1 or not I:
            return 0.0
        person, outcome = evidence[0]
        if person not in I:
            return 0.0
        if tuple_mode and root_tuple_pmfs.get(person):
            return float(root_tuple_pmfs.get(person, {}).get(outcome, 0.0)) / float(len(I))
        return float(p_s0.get(person, {}).get(outcome, 0.0)) / float(len(I))

    def _add_oracle_gauge_constraints():
        if not oracle_gauge_active:
            return
        root_expr = oracle_residual_expr_by_state.get(s0)
        if root_expr is not None:
            name = "oracle_residual_gauge_root_anchor"
            master.addConstr(root_expr == 0.0, name=name)
            oracle_gauge_constraint_names.append(name)

        stage1_weighted = gp.LinExpr()
        stage1_weight_total = 0.0
        for state in sorted(belief, key=_deterministic_state_sort_key):
            if len(_evidence(state)) != 1:
                continue
            expr = oracle_residual_expr_by_state.get(state)
            if expr is None:
                continue
            weight = _root_successor_weight(state)
            if weight <= 0.0:
                continue
            stage1_weighted.add(weight * expr)
            stage1_weight_total += weight
        if stage1_weight_total > 0.0:
            name = "oracle_residual_gauge_root_stage1_weighted_mean"
            master.addConstr(stage1_weighted == 0.0, name=name)
            oracle_gauge_constraint_names.append(name)

        if per_gene_phi_active and W_gene_var is not None:
            root_gene_probs_for_gauge = gene_probs_root or {}
            for gene in gene_list:
                gene_root = root_gene_probs_for_gauge.get(gene, {})
                for person in I:
                    dist = gene_root.get(person, {})
                    if not dist:
                        continue
                    expr = gp.LinExpr()
                    for g_val, prob in dist.items():
                        prob_f = float(prob)
                        if prob_f <= 0.0:
                            continue
                        expr.add(prob_f * _get_w_gene(gene, person, g_val))
                    name = f"oracle_residual_gauge_w_center_{gene}_{person}"
                    master.addConstr(expr == 0.0, name=name)
                    oracle_gauge_constraint_names.append(name)
        elif W_var is not None:
            for person in I:
                if tuple_mode and root_tuple_pmfs.get(person):
                    dist = root_tuple_pmfs.get(person, {})
                else:
                    dist = p_s0.get(person, {})
                if not dist:
                    continue
                expr = gp.LinExpr()
                for outcome, prob in dist.items():
                    prob_f = float(prob)
                    if prob_f <= 0.0:
                        continue
                    expr.add(prob_f * _get_w(person, outcome))
                name = f"oracle_residual_gauge_w_center_{person}"
                master.addConstr(expr == 0.0, name=name)
                oracle_gauge_constraint_names.append(name)

        oracle_adp_diagnostics["gauge_constraints_added"] = list(oracle_gauge_constraint_names)

    _add_oracle_gauge_constraints()

    def _oracle_exact_value_for_state(state):
        if not isinstance(oracle_adp_star, Mapping):
            return None
        exact_values = oracle_adp_star.get("exact_values", {})
        if not isinstance(exact_values, Mapping):
            return None
        key_tuple = tuple(sorted(_evidence(state)))
        label = _state_label(state)
        value = exact_values.get(state, exact_values.get(key_tuple, exact_values.get(label)))
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _root_action_successors_for_signature(person):
        if tuple_mode and root_tuple_pmfs.get(person):
            dist = root_tuple_pmfs.get(person, {})
        else:
            dist = p_s0.get(person, {})
        successors = []
        for outcome, prob in dist.items():
            prob_f = float(prob)
            if prob_f <= 0.0:
                continue
            succ = _merge_state(s0, person, outcome)
            if succ not in belief:
                continue
            successors.append((succ, prob_f))
        return successors

    def _constant_residual_summary(values):
        clean = [float(value) for value in values if math.isfinite(float(value))]
        if not clean:
            return {
                "row_count": 0,
                "candidate_norm": 0.0,
                "constant_projection": 0.0,
                "residual_norm": 0.0,
                "residual_ratio_after_stage_constant": None,
            }
        mean_value = sum(clean) / float(len(clean))
        residual_norm = math.sqrt(sum((value - mean_value) * (value - mean_value) for value in clean))
        candidate_norm = math.sqrt(sum(value * value for value in clean))
        residual_ratio = residual_norm / candidate_norm if candidate_norm > 1e-12 else 0.0
        return {
            "row_count": int(len(clean)),
            "candidate_norm": float(candidate_norm),
            "constant_projection": float(mean_value),
            "residual_norm": float(residual_norm),
            "residual_ratio_after_stage_constant": float(residual_ratio),
        }

    def _compute_bellman_signature_diagnostic():
        if not isinstance(oracle_adp_star, Mapping):
            return None
        feature_names = [
            "oracle_exact_value",
            "frontier_carrier_mass",
            "frontier_carrier_variance",
            "bridge_depth_mass",
            "descendant_bridge_mass",
            "sibling_breadth",
            "collateral_block_count",
        ]
        signatures = {name: [] for name in feature_names}

        def _feature_value(name, state):
            if name == "oracle_exact_value":
                value = _oracle_exact_value_for_state(state)
                return 0.0 if value is None else float(value)
            try:
                features = build_state_features(
                    state,
                    belief=belief,
                    individuals=I,
                    pedigree=pedigree,
                    genes=gene_list if multi_gene else None,
                    myopic_policy=myopic_policy_map,
                    myopic_values=myopic_value_map,
                    myopic_residuals=myopic_residuals,
                )
            except Exception:
                return 0.0
            return float(features.get(name, 0.0) or 0.0)

        root_rows = 0
        for person in I:
            successors = _root_action_successors_for_signature(person)
            if not successors:
                continue
            root_rows += 1
            for name in feature_names:
                root_value = _feature_value(name, s0)
                expected_successor = sum(prob * _feature_value(name, succ) for succ, prob in successors)
                signatures[name].append(float(root_value - expected_successor))

        feature_summaries = {
            name: _constant_residual_summary(values)
            for name, values in signatures.items()
        }
        return {
            "schema": "root_test_bellman_signature_v1",
            "row_scope": "root_test_actions",
            "root_test_row_count": int(root_rows),
            "legacy_span_probe": "stage_constant_plus_W_martingale_zero",
            "feature_summaries": feature_summaries,
        }

    oracle_bellman_signature_diagnostic = _compute_bellman_signature_diagnostic()

    def _oracle_payload_coverage_counts(star):
        if not isinstance(star, Mapping):
            return 0, len(belief)
        exact_values = star.get("exact_values", {})
        if not isinstance(exact_values, Mapping):
            return 0, len(belief)
        covered = 0
        for state in belief:
            key_tuple = tuple(sorted(_evidence(state)))
            label = _state_label(state)
            if state in exact_values or key_tuple in exact_values or label in exact_values:
                covered += 1
        return int(covered), int(max(0, len(belief) - covered))

    # 5) Objective
    # theta3 3. Modify the Master Objective to Include theta
    master.setObjective(
        gp.quicksum(mu0.get(_evidence(s), 0.0) * get_Phi(s) for s in Phi),
        GRB.MINIMIZE
    )
    master.optimize()
    master.update()  # Ensure constraint attributes are available

    # Safely get slack value with error handling
    try:
        root_constraint = master.getConstrByName("root_stop")
        if root_constraint is not None and master.status == GRB.Status.OPTIMAL:
            root_slack = root_constraint.Slack
        else:
            root_slack = 0.0
    except Exception:
        root_slack = 0.0

    if verbose:
        print(f"[debug] Φ(root) = {Phi[s0].X: .6f}   "
              f"R_stop(root) = {Rstop_root: .6f}   "
              f"slack = {root_slack: .6e}")
    # Optional debug dump
    if debug_lp_path:
        # 1) dump the LP so you can eyeball it
        prev_presolve = int(master.Params.Presolve)
        master.Params.Presolve = 0
        master.write(debug_lp_path)       
        master.Params.Presolve = prev_presolve
        # 2) compute an irreducible infeasible subsystem
        if master.status == GRB.Status.INFEASIBLE:


            master.computeIIS()               
            master.write("master_theta_iis.ilp")    

            raise RuntimeError(
                f"🛑 infeasible right after seeding –\n"
                f"LP dumped to {debug_lp_path!r} and IIS to 'master_iis.ilp'"
            )

    def _collect_root_constraint_diagnostics(limit: int = 8):
        root_entries = []
        for constr in master.getConstrs():
            name = constr.ConstrName or ""
            if not (
                name == "root_stop"
                or name.startswith("init_test_")
                or name.startswith("tail_reg_root_")
                or name.endswith("_root_lb")
                or name.endswith("_root_ub")
            ):
                continue
            entry = {
                "name": name,
                "sense": constr.Sense,
                "rhs": float(constr.RHS),
            }
            if master.status == GRB.Status.OPTIMAL:
                try:
                    entry["slack"] = float(constr.Slack)
                except Exception:
                    entry["slack"] = None
            root_entries.append(entry)

        if master.status == GRB.Status.OPTIMAL:
            root_entries.sort(
                key=lambda item: (
                    abs(item["slack"]) if isinstance(item.get("slack"), (int, float)) else float("inf"),
                    item["name"],
                )
            )
        else:
            root_entries.sort(key=lambda item: item["name"])

        return root_entries[:limit], len(root_entries)

    def _collect_seed_probe():
        root_phi_lp = None
        objective_value = None
        root_stop_rhs = None
        root_stop_slack = None
        if master.status == GRB.Status.OPTIMAL:
            try:
                root_phi_lp = float(get_Phi(s0).X)
            except Exception:
                root_phi_lp = None
            try:
                objective_value = float(master.ObjVal)
            except Exception:
                objective_value = None
        root_constraint = master.getConstrByName("root_stop")
        if root_constraint is not None:
            try:
                root_stop_rhs = float(root_constraint.RHS)
            except Exception:
                root_stop_rhs = None
            if master.status == GRB.Status.OPTIMAL:
                try:
                    root_stop_slack = float(root_constraint.Slack)
                except Exception:
                    root_stop_slack = None

        init_test_constraints = []
        rhs_values = []
        if master.status == GRB.Status.OPTIMAL and root_phi_lp is not None:
            for name in seed_root_init_constraint_names:
                constr = master.getConstrByName(name)
                if constr is None:
                    continue
                try:
                    slack_val = float(constr.Slack)
                except Exception:
                    slack_val = None
                rhs_val = None if slack_val is None else float(root_phi_lp - slack_val)
                if rhs_val is not None:
                    rhs_values.append(rhs_val)
                init_test_constraints.append(
                    {
                        "name": name,
                        "slack": slack_val,
                        "rhs": rhs_val,
                    }
                )
        edge_star = _current_edge_star()
        trio_star = _current_trio_star()
        myopic_adp_star = _current_myopic_adp_star()
        oracle_adp_star_current = _current_oracle_adp_star()
        regime_residual_star_current = _current_regime_residual_star()
        edge_coef_l1, edge_coef_l2, edge_coef_nonzero = _feature_coef_summary(edge_star)
        trio_coef_l1, trio_coef_l2, trio_coef_nonzero = _feature_coef_summary(trio_star)
        oracle_coeffs = (
            oracle_adp_star_current.get("coefficients", {})
            if isinstance(oracle_adp_star_current, Mapping)
            else {}
        )
        oracle_coef_l1, oracle_coef_l2, oracle_coef_nonzero = _feature_coef_summary(oracle_coeffs)
        return {
            "objective_value": objective_value,
            "root_phi_lp": root_phi_lp,
            "root_stop_rhs": root_stop_rhs,
            "root_stop_slack": root_stop_slack,
            "seed_init_test_constraints": init_test_constraints,
            "seed_init_test_rhs_nonnegative_count": int(sum(1 for value in rhs_values if value >= 0.0)),
            "seed_init_test_rhs_negative_count": int(sum(1 for value in rhs_values if value < 0.0)),
            "seed_init_test_rhs_min": min(rhs_values) if rhs_values else None,
            "seed_init_test_rhs_max": max(rhs_values) if rhs_values else None,
            "edge_feature_mode": edge_feature_mode,
            "edge_feature_blocks": list(edge_feature_blocks),
            "edge_coef_l1": float(edge_coef_l1),
            "edge_coef_l2": float(edge_coef_l2),
            "edge_coef_nonzero": int(edge_coef_nonzero),
            "edge_star": edge_star,
            "trio_feature_mode": trio_feature_mode,
            "trio_feature_blocks": list(trio_feature_blocks),
            "trio_coef_sharing": trio_coef_sharing,
            "trio_coef_l1": float(trio_coef_l1),
            "trio_coef_l2": float(trio_coef_l2),
            "trio_coef_nonzero": int(trio_coef_nonzero),
            "trio_star": trio_star,
            "myopic_adp": serializable_summary(myopic_adp_star),
            "myopic_adp_star": myopic_adp_star,
            "oracle_adp": oracle_serializable_summary(oracle_adp_star_current),
            "oracle_adp_star": oracle_adp_star_current,
            "oracle_coef_l1": float(oracle_coef_l1),
            "oracle_coef_l2": float(oracle_coef_l2),
            "oracle_coef_nonzero": int(oracle_coef_nonzero),
        }

    seed_probe = _collect_seed_probe()

    # --- Row‑generation Loop ---
    iterator = range(1, max_iters+1)
    if HAS_TQDM and not verbose:
        iterator = tqdm(iterator, desc="Row Gen", unit="iter", leave=False)

    stop_rhs_cache = {}
    rowgen_state_cursor = 0
    rowgen_exit_reason = None
    rowgen_converged = False
    rowgen_pass_signature = ""
    rowgen_states_total = 0
    rowgen_states_scanned = 0
    rowgen_states_truncated_count = 0
    rowgen_candidate_cuts_total = 0
    rowgen_stop_cut_candidates_total = 0
    rowgen_test_cut_candidates_total = 0
    rowgen_cuts_added = 0
    rowgen_cuts_truncated_count = 0
    rowgen_oracle_only_truncated_tuple_cuts_suppressed = 0
    rowgen_last_worst_gap = float("inf")

    runtime_start_time = time.time()
    runtime_last_progress_time = runtime_start_time
    runtime_last_progress_iter = 0
    runtime_heartbeat_count = 0
    runtime_last_heartbeat_time = runtime_start_time
    runtime_last_heartbeat_iter = 0
    progress_coverage_anchor = 0
    progress_gap_anchor = float("inf")
    cumulative_states_seen = set()
    it = 0

    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _write_runtime_sidecar(status: str, runtime_reason, rowgen_reason) -> None:
        if runtime_sidecar_path is None:
            return
        now = time.time()
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "run_id": runtime_sidecar_run_id,
            "tier": runtime_sidecar_tier,
            "contract_profile": runtime_sidecar_contract_profile,
            "case": runtime_sidecar_case,
            "theta_mode": theta_mode,
            "child_pid": int(os.getpid()),
            "status": status,
            "elapsed_sec": float(max(0.0, now - runtime_start_time)),
            "runtime_exit_reason": runtime_reason,
            "rowgen_exit_reason": rowgen_reason,
            "rowgen_states_total": int(rowgen_states_total),
            "rowgen_states_scanned": int(rowgen_states_scanned),
            "rowgen_candidate_cuts_total": int(rowgen_candidate_cuts_total),
            "rowgen_stop_cut_candidates_total": int(rowgen_stop_cut_candidates_total),
            "rowgen_test_cut_candidates_total": int(rowgen_test_cut_candidates_total),
            "rowgen_cuts_added": int(rowgen_cuts_added),
            "runtime_last_progress_sec_ago": float(max(0.0, now - runtime_last_progress_time)),
            "runtime_last_progress_iter": int(runtime_last_progress_iter),
            "runtime_heartbeat_count": int(runtime_heartbeat_count),
            "trio_feature_cache_hits": int(trio_feature_cache_hits_total),
            "trio_feature_cache_misses": int(trio_feature_cache_misses_total),
            "trio_feature_materialization_sec_total": float(trio_feature_materialization_sec_total),
        }
        try:
            _atomic_write_json(runtime_sidecar_path, payload)
        except Exception:
            pass

    def _finalize_analysis_return(
        *,
        analysis_exit_reason: str,
        rowgen_telemetry: Mapping[str, object],
        runtime_telemetry: Mapping[str, object],
        rowgen_first_pass_probe=None,
    ):
        tight_root_constraints, root_constraint_count = _collect_root_constraint_diagnostics(limit=8)
        edge_star = _current_edge_star()
        trio_star = _current_trio_star()
        myopic_adp_star = _current_myopic_adp_star()
        oracle_adp_star_current = _current_oracle_adp_star()
        edge_coef_l1, edge_coef_l2, edge_coef_nonzero = _feature_coef_summary(edge_star)
        trio_coef_l1, trio_coef_l2, trio_coef_nonzero = _feature_coef_summary(trio_star)
        oracle_coeffs = (
            oracle_adp_star_current.get("coefficients", {})
            if isinstance(oracle_adp_star_current, Mapping)
            else {}
        )
        oracle_coef_l1, oracle_coef_l2, oracle_coef_nonzero = _feature_coef_summary(oracle_coeffs)
        oracle_payload_coverage_count, oracle_payload_missing_count = _oracle_payload_coverage_counts(
            oracle_adp_star_current
        )
        analysis_root_diagnostics = {
            "analysis_mode": analysis_mode,
            "model_status": int(master.status),
            "objective_value": seed_probe.get("objective_value"),
            "objective_sense": "minimize",
            "root_state_size": len(_evidence(s0)),
            "root_phi_lp": seed_probe.get("root_phi_lp"),
            "root_stop_rhs": seed_probe.get("root_stop_rhs"),
            "root_stop_seed_rhs": float(Rstop_root),
            "root_stop_slack": seed_probe.get("root_stop_slack"),
            "root_constraint_count": root_constraint_count,
            "root_constraint_tightest": tight_root_constraints,
            "seed_scope": effective_seed_scope,
            "edge_seed_scope": edge_seed_scope,
            "trio_seed_scope": trio_seed_scope,
            "seed_stage1_state_count": int(stage1_seed_state_count),
            "seed_stage1_constraint_count": int(len(seed_stage1_constraint_names)),
            "trio_stage2_clinical_seed_state_count": int(stage2_clinical_seed_state_count),
            "seed_init_test_constraint_count": int(len(seed_root_init_constraint_names)),
            "seed_init_test_rhs_min": seed_probe.get("seed_init_test_rhs_min"),
            "seed_init_test_rhs_max": seed_probe.get("seed_init_test_rhs_max"),
            "seed_init_test_rhs_nonnegative_count": int(seed_probe.get("seed_init_test_rhs_nonnegative_count") or 0),
            "secondary_phi_objective": secondary_phi_objective,
            "secondary_phi_objective_applied": False,
            "secondary_phi_objective_root_anchor": None,
            "secondary_phi_objective_stage_states": 0,
            "secondary_phi_objective_reason": analysis_exit_reason,
            "slack_refactor_enabled": False,
            "tail_regularization_applied": False,
            "telemetry_schema_version": telemetry_schema_version,
            "rowgen_states_total": int(rowgen_telemetry.get("rowgen_states_total") or 0),
            "rowgen_states_scanned": int(rowgen_telemetry.get("rowgen_states_scanned") or 0),
            "rowgen_states_truncated_count": int(rowgen_telemetry.get("rowgen_states_truncated_count") or 0),
            "rowgen_states_truncated": bool(rowgen_telemetry.get("rowgen_states_truncated", False)),
            "rowgen_candidate_cuts_total": int(rowgen_telemetry.get("rowgen_candidate_cuts_total") or 0),
            "rowgen_stop_cut_candidates_total": int(rowgen_telemetry.get("rowgen_stop_cut_candidates_total") or 0),
            "rowgen_test_cut_candidates_total": int(rowgen_telemetry.get("rowgen_test_cut_candidates_total") or 0),
            "rowgen_cuts_added": int(rowgen_telemetry.get("rowgen_cuts_added") or 0),
            "rowgen_cuts_truncated_count": int(rowgen_telemetry.get("rowgen_cuts_truncated_count") or 0),
            "rowgen_cuts_truncated": bool(rowgen_telemetry.get("rowgen_cuts_truncated", False)),
            "rowgen_pass_signature": rowgen_telemetry.get("rowgen_pass_signature", ""),
            "rowgen_exit_reason": rowgen_telemetry.get("rowgen_exit_reason", analysis_exit_reason),
            "rowgen_converged": bool(rowgen_telemetry.get("rowgen_converged", False)),
            "rowgen_last_worst_gap": float(rowgen_telemetry.get("rowgen_last_worst_gap") or 0.0),
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed": int(
                rowgen_telemetry.get("rowgen_oracle_only_truncated_tuple_cuts_suppressed") or 0
            ),
            "exhaustive_mode_active": bool(rowgen_telemetry.get("exhaustive_mode_active", exhaustive_bellman)),
            "exhaustive_strict_active": bool(rowgen_telemetry.get("exhaustive_strict_active", exhaustive_strict)),
            "runtime_walltime_sec": float(runtime_telemetry.get("runtime_walltime_sec") or 0.0),
            "runtime_exit_reason": runtime_telemetry.get("runtime_exit_reason", analysis_exit_reason),
            "runtime_last_progress_sec_ago": float(runtime_telemetry.get("runtime_last_progress_sec_ago") or 0.0),
            "runtime_last_progress_iter": int(runtime_telemetry.get("runtime_last_progress_iter") or 0),
            "runtime_heartbeat_count": int(runtime_telemetry.get("runtime_heartbeat_count") or 0),
            "trio_feature_cache_hits": int(trio_feature_cache_hits_total),
            "trio_feature_cache_misses": int(trio_feature_cache_misses_total),
            "trio_feature_materialization_sec_total": float(trio_feature_materialization_sec_total),
            "edge_feature_mode": edge_feature_mode,
            "edge_feature_blocks": list(edge_feature_blocks),
            "edge_coef_l1": float(edge_coef_l1),
            "edge_coef_l2": float(edge_coef_l2),
            "edge_coef_nonzero": int(edge_coef_nonzero),
            "trio_feature_mode": trio_feature_mode,
            "trio_feature_blocks": list(trio_feature_blocks),
            "trio_coef_sharing": trio_coef_sharing,
            "trio_coef_l1": float(trio_coef_l1),
            "trio_coef_l2": float(trio_coef_l2),
            "trio_coef_nonzero": int(trio_coef_nonzero),
            "myopic_adp": serializable_summary(myopic_adp_star),
            "oracle_adp": oracle_serializable_summary(oracle_adp_star_current),
            "oracle_plumbing_mode": oracle_plumbing_mode if oracle_adp_active else None,
            "oracle_payload_coverage_count": int(oracle_payload_coverage_count),
            "oracle_payload_missing_count": int(oracle_payload_missing_count),
            "oracle_active_in_lp": bool(oracle_adp_active),
            "oracle_active_in_reconstruction": bool(oracle_adp_active),
            "gauge_constraints_added": list(oracle_gauge_constraint_names),
            "gauge_constraint_count": int(len(oracle_gauge_constraint_names)),
            "bellman_signature_diagnostic": oracle_bellman_signature_diagnostic,
            "oracle_coef_l1": float(oracle_coef_l1),
            "oracle_coef_l2": float(oracle_coef_l2),
            "oracle_coef_nonzero": int(oracle_coef_nonzero),
        }
        oracle_summary = analysis_root_diagnostics.get("oracle_adp")
        if isinstance(oracle_summary, dict):
            oracle_summary.update(
                {
                    "oracle_plumbing_mode": oracle_plumbing_mode,
                    "oracle_payload_coverage_count": int(oracle_payload_coverage_count),
                    "oracle_payload_missing_count": int(oracle_payload_missing_count),
                    "active_in_lp": bool(oracle_adp_active),
                    "active_in_reconstruction": bool(oracle_adp_active),
                    "legacy_residual_enabled": bool(oracle_adp_active and not oracle_only_fixed_phi_active),
                    "gauge_constraints_added": list(oracle_gauge_constraint_names),
                    "gauge_constraint_count": int(len(oracle_gauge_constraint_names)),
                }
            )

        Phi_sol = _current_phi_solution()
        theta_star = _current_theta_star()
        W_sol = _current_w_solution()
        W_gene_sol = _current_w_gene_solution()
        aaub_star = _current_aaub_star()
        adp_inference_time = inf_cache.total_inference_time
        phi_eval = {}
        if return_phi_eval:
            from .postprocess import phi_hat
            try:
                phi_eval[s0] = phi_hat(
                    s0,
                    theta_star=theta_star,
                    W_star=W_sol,
                    belief=belief,
                    gen_states=gen_states,
                    individuals=I,
                    theta_mode=theta_mode,
                    pedigree=pedigree,
                    tuple_pmfs=tuple_posteriors if tuple_mode else None,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=edge_star,
                    W_trio_star=trio_star,
                    pedigree_edges=pedigree_edges if edge_features_active else None,
                    pedigree_trios=pedigree_trios if trio_features_active else None,
                    infer=infer,
                    genes=gene_list if multi_gene else None,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star_current,
                    regime_residual_star=regime_residual_star_current,
                    theta_model=theta_model_info.get("theta_model"),
                    theta_model_spec=theta_model_info.get("theta_model_spec"),
                )
            except Exception:
                phi_eval = {}

        if return_stats:
            stats = {
                "num_vars": master.NumVars,
                "num_constrs": master.NumConstrs,
                "phi_vars": len(Phi),
                "elapsed_iters": 0,
                "role_groups": {k: len(v) for k, v in (role_groups or {}).items()},
                "adp_inference_time": adp_inference_time,
                "adp_inference_calls": inf_cache.inference_calls,
                "cache_hits": bellman_gen.cache_hits,
                "cache_misses": bellman_gen.cache_misses,
                "bn_time": bellman_gen.bn_time,
                "tail_regularization": {"enabled": False, "applied": False},
                "slack_refactor": {"enabled": False},
                "aaub": aaub_star,
                "edge_star": edge_star,
                "trio_star": trio_star,
                "myopic_adp_star": myopic_adp_star,
                "myopic_adp": serializable_summary(myopic_adp_star),
                "oracle_adp_star": oracle_adp_star_current,
                "oracle_adp": oracle_serializable_summary(oracle_adp_star_current),
                "root_diagnostics": analysis_root_diagnostics,
                "rowgen_telemetry": dict(rowgen_telemetry),
                "runtime_telemetry": dict(runtime_telemetry),
                "candidate_pre_polish": {},
                "candidate_post_polish": {},
                "W_gene_star": W_gene_sol,
                "seed_probe": seed_probe,
                "rowgen_first_pass_probe": rowgen_first_pass_probe,
            }
            payload = (Phi_sol, W_sol, belief, theta_star, master, adp_inference_time, inf_cache)
            if return_phi_eval:
                payload = payload + (phi_eval,)
            return payload, stats

        inf_cache.tuple_posteriors = tuple_posteriors if tuple_mode else {}
        inf_cache.tail_regularization = {"enabled": False, "applied": False}
        inf_cache.slack_refactor = {"enabled": False}
        inf_cache.aaub_star = aaub_star
        inf_cache.edge_star = edge_star
        inf_cache.trio_star = trio_star
        inf_cache.myopic_adp_star = myopic_adp_star
        inf_cache.oracle_adp_star = oracle_adp_star_current
        inf_cache.pedigree_edges = pedigree_edges if edge_features_active else None
        inf_cache.pedigree_trios = pedigree_trios if trio_features_active else None
        inf_cache.root_diagnostics = analysis_root_diagnostics
        inf_cache.rowgen_telemetry = dict(rowgen_telemetry)
        inf_cache.runtime_telemetry = dict(runtime_telemetry)
        inf_cache.candidate_pre_polish = {}
        inf_cache.candidate_post_polish = {}
        inf_cache.candidate_eval_payloads = {}
        inf_cache.w_gene_star = W_gene_sol
        payload = (Phi_sol, W_sol, belief, theta_star, master, adp_inference_time, inf_cache)
        if return_phi_eval:
            payload = payload + (phi_eval,)
        return payload

    def _scan_rowgen_candidates():
        nonlocal rowgen_state_cursor
        candidate_cuts = []
        worst_gap = 0.0
        phi_values = {s: phi_var.X for s, phi_var in Phi.items()}
        all_states = list(belief.keys()) if exhaustive_bellman else list(Phi.keys())
        all_states.sort(key=_deterministic_state_sort_key)
        states_total_iter = len(all_states)
        if states_total_iter <= max_states_per_iter:
            state_cursor = 0
            states_to_check = all_states
        else:
            if exhaustive_bellman:
                state_cursor = 0
            else:
                state_cursor = rowgen_state_cursor % states_total_iter
            states_to_check = [
                all_states[(state_cursor + idx) % states_total_iter]
                for idx in range(min(max_states_per_iter, states_total_iter))
            ]
            if not exhaustive_bellman:
                rowgen_state_cursor = (state_cursor + len(states_to_check)) % states_total_iter

        states_scanned_iter = len(states_to_check)
        state_truncated_iter = max(0, states_total_iter - states_scanned_iter)
        pass_signature = _rowgen_signature_payload(
            state_labels=[_state_label(state_key) for state_key in states_to_check],
            cursor=state_cursor,
            mode_label="exhaustive" if exhaustive_bellman else "capped",
        )
        stop_candidates_iter = 0
        test_candidates_iter = 0
        candidate_counts_by_stage = {}
        candidate_counts_by_person = {}
        positive_violation_states = set()
        top_violations = []

        for S in states_to_check:
            p_s, z_s = belief[S]
            del z_s
            gene_probs_S = _ensure_gene_posteriors(S, p_s)
            tested_inds = {i for (i, g) in _evidence(S)}
            evidence_key = _evidence(S)
            state_stage = len(evidence_key)
            state_label = _state_label(S)

            rhs_const = stop_rhs_cache.get(evidence_key)
            if rhs_const is None:
                if multi_gene and tuple_mode:
                    rhs_const = sum(
                        r_reward(
                            k,
                            p_s,
                            a,
                            b,
                            c,
                            delta,
                            per_gene_probs=gene_probs_S,
                            a_gene=a_gene,
                            b_gene=b_gene,
                            c_gene=c_gene,
                            delta_gene=delta_gene,
                        )
                        for k in I
                        if k not in tested_inds
                    )
                else:
                    rhs_const = sum(r_reward(k, p_s, a, b, c, delta) for k in I if k not in tested_inds)
                stop_rhs_cache[evidence_key] = rhs_const
            if rhs_const is not None:
                stop_violation = rhs_const - phi_values[S]
                if stop_violation > tol:
                    candidate_cuts.append(("stop", stop_violation, S, None, rhs_const, None))
                    stop_candidates_iter += 1
                    worst_gap = max(worst_gap, stop_violation)
                    top_violations.append(
                        {
                            "kind": "stop",
                            "state": state_label,
                            "state_stage": state_stage,
                            "person": None,
                            "violation": float(stop_violation),
                        }
                    )

            evidence = dict(_evidence(S))
            result = inf_cache.get(evidence)
            if not isinstance(result, InferenceResult):
                result = InferenceResult(
                    result,
                    gene_order=gene_list if multi_gene else ("gene",),
                    gen_states=gen_states,
                )
            p_post = result
            tuple_pmfs_state = result.get_tuple_pmfs() if tuple_mode else {}

            if tuple_mode:
                _store_tuple_posteriors(S, tuple_pmfs_state)
            _register_projection(S)

            for i in I:
                if i in tested_inds:
                    continue

                p_post_i = p_post[i]
                tuple_post_i = tuple_pmfs_state.get(i, {}) if tuple_mode else {}
                scalar_max = max(p_post_i.values()) if p_post_i else 0.0
                tuple_max = max(tuple_post_i.values()) if tuple_post_i else 0.0
                if max(scalar_max, tuple_max) >= 1.0 - tol:
                    continue
                if tuple_mode and tuple_pmfs_state.get(i):
                    succs_all = [(g, p) for g, p in tuple_pmfs_state[i].items() if _keep_successor_prob(p)]
                else:
                    if exhaustive_bellman:
                        succs_all = [(g, p) for g, p in p_post_i.items() if p > 0.0]
                    else:
                        succs_all = select_successors(p_post_i, K=None, pmin=successor_pmin)
                if not succs_all:
                    continue
                p12 = p_post_i.get(1, 0.0) + p_post_i.get(2, 0.0)
                per_gene_p12 = _per_gene_p12_map(gene_probs_S, i)
                immediate_reward = r_reward_testp(
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

                phi_S = phi_values[S]
                phi_succ = {g: phi_values.get(_merge_state(S, i, g), 0.0) for g, p in succs_all}
                probs_full = {g: p for g, p in succs_all}
                violation, _rhs_numeric = bellman_violation(
                    phi_S=phi_S,
                    phi_succ=phi_succ,
                    probs=probs_full,
                    r_immediate=immediate_reward,
                )
                if violation > GAP_TOL:
                    cut_key = cut_cache.dedup_key(S, i, tuple(succs_all), immediate_reward)
                    if cut_cache.is_new(cut_key):
                        candidate_cuts.append(("bellman_test", violation, S, i, immediate_reward, succs_all))
                        test_candidates_iter += 1
                        worst_gap = max(worst_gap, violation)
                        positive_violation_states.add(S)
                        stage_key = str(state_stage)
                        candidate_counts_by_stage[stage_key] = int(candidate_counts_by_stage.get(stage_key, 0)) + 1
                        candidate_counts_by_person[i] = int(candidate_counts_by_person.get(i, 0)) + 1
                        top_violations.append(
                            {
                                "kind": "bellman_test",
                                "state": state_label,
                                "state_stage": state_stage,
                                "person": i,
                                "violation": float(violation),
                            }
                        )

        top_violations.sort(key=lambda item: (-float(item["violation"]), item["kind"], item["state"]))
        return {
            "candidate_cuts": candidate_cuts,
            "worst_gap": worst_gap,
            "states_total_iter": states_total_iter,
            "states_scanned_iter": states_scanned_iter,
            "state_truncated_iter": state_truncated_iter,
            "rowgen_pass_signature": pass_signature,
            "stop_candidates_iter": stop_candidates_iter,
            "test_candidates_iter": test_candidates_iter,
            "candidate_counts_by_stage": candidate_counts_by_stage,
            "candidate_counts_by_person": candidate_counts_by_person,
            "states_with_positive_bellman_violation": int(len(positive_violation_states)),
            "top_violations": top_violations[:10],
        }

    _write_runtime_sidecar("running", None, None)

    if analysis_mode == "seeded":
        runtime_walltime_sec = time.time() - runtime_start_time
        runtime_telemetry = {
            "runtime_walltime_sec": float(runtime_walltime_sec),
            "runtime_exit_reason": "analysis_seeded",
            "runtime_last_progress_sec_ago": 0.0,
            "runtime_last_progress_iter": 0,
            "runtime_heartbeat_count": int(runtime_heartbeat_count),
        }
        rowgen_telemetry = {
            "telemetry_schema_version": telemetry_schema_version,
            "rowgen_states_total": 0,
            "rowgen_states_scanned": 0,
            "rowgen_states_truncated_count": 0,
            "rowgen_states_truncated": False,
            "rowgen_candidate_cuts_total": 0,
            "rowgen_stop_cut_candidates_total": 0,
            "rowgen_test_cut_candidates_total": 0,
            "rowgen_cuts_added": 0,
            "rowgen_cuts_truncated_count": 0,
            "rowgen_cuts_truncated": False,
            "rowgen_pass_signature": "",
            "rowgen_exit_reason": "analysis_seeded",
            "rowgen_converged": False,
            "rowgen_last_worst_gap": 0.0,
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed": 0,
            "exhaustive_mode_active": bool(exhaustive_bellman),
            "exhaustive_strict_active": bool(exhaustive_strict),
        }
        _write_runtime_sidecar("completed", "analysis_seeded", "analysis_seeded")
        return _finalize_analysis_return(
            analysis_exit_reason="analysis_seeded",
            rowgen_telemetry=rowgen_telemetry,
            runtime_telemetry=runtime_telemetry,
            rowgen_first_pass_probe=None,
        )

    if analysis_mode == "first_pass":
        scan = _scan_rowgen_candidates()
        runtime_walltime_sec = time.time() - runtime_start_time
        runtime_telemetry = {
            "runtime_walltime_sec": float(runtime_walltime_sec),
            "runtime_exit_reason": "analysis_first_pass",
            "runtime_last_progress_sec_ago": 0.0,
            "runtime_last_progress_iter": 0,
            "runtime_heartbeat_count": int(runtime_heartbeat_count),
        }
        rowgen_telemetry = {
            "telemetry_schema_version": telemetry_schema_version,
            "rowgen_states_total": int(scan["states_total_iter"]),
            "rowgen_states_scanned": int(scan["states_scanned_iter"]),
            "rowgen_states_truncated_count": int(scan["state_truncated_iter"]),
            "rowgen_states_truncated": bool(scan["state_truncated_iter"] > 0),
            "rowgen_candidate_cuts_total": int(len(scan["candidate_cuts"])),
            "rowgen_stop_cut_candidates_total": int(scan["stop_candidates_iter"]),
            "rowgen_test_cut_candidates_total": int(scan["test_candidates_iter"]),
            "rowgen_cuts_added": 0,
            "rowgen_cuts_truncated_count": 0,
            "rowgen_cuts_truncated": False,
            "rowgen_pass_signature": str(scan["rowgen_pass_signature"]),
            "rowgen_exit_reason": "analysis_first_pass",
            "rowgen_converged": False,
            "rowgen_last_worst_gap": float(scan["worst_gap"]),
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed": 0,
            "exhaustive_mode_active": bool(exhaustive_bellman),
            "exhaustive_strict_active": bool(exhaustive_strict),
        }
        rowgen_first_pass_probe = {
            "candidate_cut_count": int(len(scan["candidate_cuts"])),
            "stop_candidate_cut_count": int(scan["stop_candidates_iter"]),
            "test_candidate_cut_count": int(scan["test_candidates_iter"]),
            "candidate_counts_by_stage": dict(scan["candidate_counts_by_stage"]),
            "candidate_counts_by_person": dict(scan["candidate_counts_by_person"]),
            "states_with_positive_bellman_violation": int(scan["states_with_positive_bellman_violation"]),
            "top_violations": list(scan["top_violations"]),
            "rowgen_pass_signature": str(scan["rowgen_pass_signature"]),
            "worst_gap": float(scan["worst_gap"]),
        }
        _write_runtime_sidecar("completed", "analysis_first_pass", "analysis_first_pass")
        return _finalize_analysis_return(
            analysis_exit_reason="analysis_first_pass",
            rowgen_telemetry=rowgen_telemetry,
            runtime_telemetry=runtime_telemetry,
            rowgen_first_pass_probe=rowgen_first_pass_probe,
        )

    for it in iterator:
        now = time.time()
        elapsed = now - runtime_start_time
        if elapsed > exhaustive_walltime_limit_sec:
            rowgen_exit_reason = "walltime_limit"
            break
        if (now - runtime_last_progress_time) > exhaustive_no_progress_limit_sec:
            rowgen_exit_reason = "no_progress_timeout"
            break
        if exhaustive_max_rss_mb is not None:
            rss_mb = _current_rss_mb()
            if rss_mb is not None and rss_mb > exhaustive_max_rss_mb:
                rowgen_exit_reason = "memory_limit"
                break
        iter_heartbeat_due = (it - runtime_last_heartbeat_iter) >= exhaustive_heartbeat_every_iters
        time_heartbeat_due = (now - runtime_last_heartbeat_time) >= exhaustive_heartbeat_every_sec
        if iter_heartbeat_due or time_heartbeat_due:
            runtime_heartbeat_count += 1
            runtime_last_heartbeat_time = now
            runtime_last_heartbeat_iter = it
            _write_runtime_sidecar("running", None, rowgen_exit_reason)
            if verbose:
                print(
                    f"[watchdog] heartbeat iter={it} elapsed={elapsed:.1f}s "
                    f"last_progress={now - runtime_last_progress_time:.1f}s ago"
                )
        master.optimize()
        # Safely get slack value with error handling
        try:
            root_constraint = master.getConstrByName("root_stop")
            if root_constraint is not None and master.status == GRB.Status.OPTIMAL:
                root_slack = root_constraint.Slack
            else:
                root_slack = 0.0
        except Exception:
            root_slack = 0.0
            
        if verbose:
            print(f"[debug] Φ(root) = {Phi[s0].X: .6f}   "
                  f"R_stop(root) = {Rstop_root: .6f}   "
                  f"slack = {root_slack: .6e}")
        if master.status == GRB.Status.INFEASIBLE:   # ← catch later crashes
            master.computeIIS()
            master.write(f"iter{it:03d}_theta_iis.ilp")
            raise RuntimeError(f"Master infeasible in iteration {it} – "
                           f"IIS written to iter{it:03d}_theta_iis.ilp")
        if master.status in {GRB.Status.INF_OR_UNBD, GRB.Status.UNBOUNDED}:
            raise RuntimeError(
                f"Master status {master.status} in iteration {it} "
                "(INF_OR_UNBD / UNBOUNDED) before row generation."
            )
        if master.status != GRB.Status.OPTIMAL:
            raise RuntimeError(
                f"Master status {master.status} in iteration {it} before row generation."
            )
        # --- optional diagnostic: ensure no blank‑LHS rows were added ---
        for r in master.getConstrs():
            # r.Sense is one of '=', '<', '>' (for ≥ rows it’s '>')
            if r.Sense == '>':
                # r.RHS is the right‑hand‑side constant
                if abs(r.RHS) > tol:              # only care about non‑zero RHS
                    # master.getRow(r) returns a LinExpr of LHS
                    if master.getRow(r).size() == 0:
                        raise RuntimeError(f"❌ Constraint {r.ConstrName!r} has empty LHS but RHS={r.RHS}")


        if verbose:
            print(f"[master] iter {it} obj={master.ObjVal:.6f}")
            print(f"[ITER {it}] Current variable values:")
            if theta_model is not None:
                print(f"         theta_model={theta_model}")
            if stage_theta_active:
                theta0 = theta_vars[0].X
                thetan = theta_vars[len(I)].X
                print(f"         theta[0]={theta0:.6f} theta[{len(I)}]={thetan:.6f}")
            elif stage_gene_theta_active:
                for k in range(len(I) + 1):
                    stage_base = theta_stage_base_vars[k].X
                    print(f"         theta_stage_base[{k}]={stage_base:.6f}")
                    for gene in gene_list:
                        theta_dev = theta_vars[k, gene].X
                        theta_effective = stage_base * stage_gene_shared_scale + theta_dev
                        print(
                            f"         theta_stage_gene[{k},{gene}]={theta_dev:.6f} "
                            f"theta_stage_gene_effective[{k},{gene}]={theta_effective:.6f}"
                        )
            elif person_theta_active:
                for i in I:
                    print(f"         theta_person[{i}]={theta_vars[i].X:.6f}")
            elif person_stage_theta_active:
                for i in I:
                    for k in range(1, len(I) + 1):
                        print(f"         theta_person_stage[{i},{k}]={theta_vars[i, k].X:.6f}")
            else:
                print(f"         theta = {theta_var.X:.6f}")
            if W_var is not None:
                for i in I:
                    for g in available_outcomes:
                        print(f"         W[{i},{g}] = {_get_w(i, g).X:.6f}")
            if per_gene_phi_active:
                for gene in gene_list:
                    for i in I:
                        for g in gen_states:
                            print(f"         W_gene[{gene},{i},{g}] = {_get_w_gene(gene, i, g).X:.6f}")

        # --- Bellman-consistent row generation (Refactored) ---
        worst_gap = 0.0
        candidate_cuts = []
        
        # Get current Phi values
        phi_values = {s: phi_var.X for s, phi_var in Phi.items()}
        all_states = list(belief.keys()) if exhaustive_bellman else list(Phi.keys())
        all_states.sort(key=_deterministic_state_sort_key)
        states_total_iter = len(all_states)
        rowgen_states_total = max(rowgen_states_total, states_total_iter)
        if states_total_iter <= max_states_per_iter:
            state_cursor = 0
            states_to_check = all_states
        else:
            if exhaustive_bellman:
                state_cursor = 0
            else:
                state_cursor = rowgen_state_cursor % states_total_iter
            states_to_check = [
                all_states[(state_cursor + idx) % states_total_iter]
                for idx in range(min(max_states_per_iter, states_total_iter))
            ]
            if not exhaustive_bellman:
                rowgen_state_cursor = (state_cursor + len(states_to_check)) % states_total_iter

        states_scanned_iter = len(states_to_check)
        rowgen_states_scanned += states_scanned_iter
        state_truncated_iter = max(0, states_total_iter - states_scanned_iter)
        rowgen_states_truncated_count += state_truncated_iter
        for state_key in states_to_check:
            cumulative_states_seen.add(state_key)
        rowgen_pass_signature = _rowgen_signature_payload(
            state_labels=[_state_label(state_key) for state_key in states_to_check],
            cursor=state_cursor,
            mode_label="exhaustive" if exhaustive_bellman else "capped",
        )
        if exhaustive_strict and state_truncated_iter > 0:
            rowgen_exit_reason = "state_truncation"
            rowgen_last_worst_gap = worst_gap
            break

        for S in states_to_check:

            # (a) stopping subproblem - cache RHS by evidence to avoid re-solving per order
            p_s, z_s = belief[S]
            gene_probs_S = _ensure_gene_posteriors(S, p_s)
            tested_inds = {i for (i, g) in _evidence(S)}
            evidence_key = _evidence(S)

            rhs_const = stop_rhs_cache.get(evidence_key)
            if rhs_const is None:
                if multi_gene and tuple_mode:
                    rhs_const = sum(
                        r_reward(
                            k,
                            p_s,
                            a,
                            b,
                            c,
                            delta,
                            per_gene_probs=gene_probs_S,
                            a_gene=a_gene,
                            b_gene=b_gene,
                            c_gene=c_gene,
                            delta_gene=delta_gene,
                        )
                        for k in I
                        if k not in tested_inds
                    )
                else:
                    rhs_const = sum(
                        r_reward(k, p_s, a, b, c, delta)
                        for k in I
                        if k not in tested_inds
                    )
                stop_rhs_cache[evidence_key] = rhs_const
            if rhs_const is not None:
                stop_violation = rhs_const - phi_values[S]
                if stop_violation > tol:
                    candidate_cuts.append(('stop', stop_violation, S, None, rhs_const, None))
                    rowgen_stop_cut_candidates_total += 1
                    if stop_violation > worst_gap:
                        worst_gap = stop_violation
            
            # (b) Bellman testing violations
            evidence = dict(_evidence(S))
            result = inf_cache.get(evidence)
            if not isinstance(result, InferenceResult):
                result = InferenceResult(
                    result,
                    gene_order=gene_list if multi_gene else ("gene",),
                    gen_states=gen_states,
                )
            p_post = result
            tuple_pmfs_state = result.get_tuple_pmfs() if tuple_mode else {}

            if tuple_mode:
                _store_tuple_posteriors(S, tuple_pmfs_state)
            _register_projection(S)

            for i in I:
                if i in tested_inds:
                    continue

                p_post_i = p_post[i]
                tuple_post_i = tuple_pmfs_state.get(i, {}) if tuple_mode else {}
                scalar_max = max(p_post_i.values()) if p_post_i else 0.0
                tuple_max = max(tuple_post_i.values()) if tuple_post_i else 0.0
                if max(scalar_max, tuple_max) >= 1.0 - tol:
                    continue
                if tuple_mode and tuple_pmfs_state.get(i):
                    succs_all = [
                        (g, p)
                        for g, p in tuple_pmfs_state[i].items()
                        if _keep_successor_prob(p)
                    ]
                    suppress_truncated_tuple_cuts = tuple_strengthening_disabled_active and exhaustive_bellman
                    if suppress_truncated_tuple_cuts:
                        # Oracle plumbing diagnostics are certificate tests for the true
                        # Bellman equations. Top-k/top-1 tuple cuts drop negative successor
                        # terms and can make exact-root/gauged probes infeasible.
                        rowgen_oracle_only_truncated_tuple_cuts_suppressed += 1 + int(topk_successors > 0)
                        succs_topk = []
                        succs_top1 = []
                    else:
                        succs_topk = (
                            select_tuple_successors(tuple_pmfs_state[i], K=topk_successors, pmin=successor_pmin)
                            if topk_successors > 0
                            else []
                        )
                        succs_top1 = select_tuple_successors(tuple_pmfs_state[i], K=1, pmin=successor_pmin)
                else:
                    if exhaustive_bellman:
                        succs_all = [(g, p) for g, p in p_post_i.items() if p > 0.0]
                    else:
                        succs_all = select_successors(p_post_i, K=None, pmin=successor_pmin)
                    succs_topk = []
                    succs_top1 = []
                if not succs_all:
                    continue
                p12 = p_post_i.get(1, 0.0) + p_post_i.get(2, 0.0)
                per_gene_p12 = _per_gene_p12_map(gene_probs_S, i)
                immediate_reward = r_reward_testp(
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

                phi_S = phi_values[S]
                phi_succ = {}

                for g, p in succs_all:
                    phi_succ[g] = phi_values.get(_merge_state(S, i, g), 0.0)

                # Main cut with all successors
                probs_full = {g: p for g, p in succs_all}
                violation, _rhs_numeric = bellman_violation(
                    phi_S=phi_S,
                    phi_succ=phi_succ,
                    probs=probs_full,
                    r_immediate=immediate_reward
                )
                if violation > GAP_TOL:
                    state_key = S
                    cut_key = cut_cache.dedup_key(state_key, i, tuple(succs_all), immediate_reward)
                    if cut_cache.is_new(cut_key):
                        candidate_cuts.append((
                            'bellman_test',
                            violation,
                            S,
                            i,
                            immediate_reward,
                            succs_all,
                        ))
                        rowgen_test_cut_candidates_total += 1
                        worst_gap = max(worst_gap, violation)

                # Optional strengthening cut using top-K successors (tuple mode)
                if tuple_mode and succs_topk:
                    probs_topk = {g: p for g, p in succs_topk}
                    violation_topk, _rhs_numeric_topk = bellman_violation(
                        phi_S=phi_S,
                        phi_succ=phi_succ,
                        probs=probs_topk,
                        r_immediate=immediate_reward
                    )
                    if violation_topk > GAP_TOL:
                        state_key = S
                        cut_key = cut_cache.dedup_key(state_key, i, tuple(succs_topk), immediate_reward)
                        if cut_cache.is_new(cut_key):
                            candidate_cuts.append((
                                'bellman_test',
                                violation_topk,
                                S,
                                i,
                                immediate_reward,
                                succs_topk,
                            ))
                            rowgen_test_cut_candidates_total += 1
                            worst_gap = max(worst_gap, violation_topk)

                if tuple_mode and succs_top1:
                    probs_top1 = {g: p for g, p in succs_top1}
                    violation_top1, _rhs_numeric_top1 = bellman_violation(
                        phi_S=phi_S,
                        phi_succ=phi_succ,
                        probs=probs_top1,
                        r_immediate=immediate_reward
                    )
                    if violation_top1 > GAP_TOL:
                        state_key = S
                        cut_key = cut_cache.dedup_key(state_key, i, tuple(succs_top1), immediate_reward)
                        if cut_cache.is_new(cut_key):
                            candidate_cuts.append((
                                'bellman_test',
                                violation_top1,
                                S,
                                i,
                                immediate_reward,
                                succs_top1,
                            ))
                            rowgen_test_cut_candidates_total += 1
                            worst_gap = max(worst_gap, violation_top1)

        rowgen_last_worst_gap = worst_gap
        # Multi-cut: add multiple cuts per iteration
        candidate_cuts.sort(key=lambda x: x[1], reverse=True)
        candidate_cuts_total_iter = len(candidate_cuts)
        rowgen_candidate_cuts_total += candidate_cuts_total_iter
        if candidate_cuts_total_iter > exhaustive_max_cut_queue:
            candidate_cuts = candidate_cuts[:exhaustive_max_cut_queue]
        queue_drop_count = max(0, candidate_cuts_total_iter - len(candidate_cuts))
        cuts_cap = min(max_cuts_per_iter, exhaustive_max_cut_queue)
        cuts_to_add = candidate_cuts[:cuts_cap]
        cut_truncated_iter = queue_drop_count + max(0, len(candidate_cuts) - len(cuts_to_add))
        rowgen_cuts_truncated_count += cut_truncated_iter
        
        # no more violations?
        if worst_gap <= tol:
            rowgen_converged = True
            rowgen_exit_reason = "converged"
            runtime_last_progress_time = time.time()
            runtime_last_progress_iter = it
            if verbose:
                print(f"Converged at iter {it} (gap {worst_gap:.2e})")
            if HAS_TQDM and not verbose and hasattr(iterator, 'close'):
                iterator.close()
            break

        if exhaustive_strict and cut_truncated_iter > 0:
            rowgen_exit_reason = "cut_truncation"
            break

        # --- Add Bellman cuts (Refactored) ---
        cuts_added = 0
        
        for cut_info in cuts_to_add:
            kind = cut_info[0]
            if kind == 'stop':
                _, _, s, _, rhs_const, _ = cut_info
                if verbose:
                    print(f"[CUT] Adding stop cut at iter {it}: Φ({_state_label(s)}) >= {rhs_const:.6f}")
                cut_name = f"cut_stop_{it}_{cuts_added}_{_state_label(s)}"
                stop_cut = master.addConstr(get_Phi(s) >= rhs_const, name=cut_name)
                _register_bellman_row_constraint(
                    stop_cut,
                    row_type="stop",
                    state=s,
                    action="stop",
                    rhs_const=rhs_const,
                    source="rowgen_stop",
                )
                cuts_added += 1
                
            elif kind == 'bellman_test':
                _, violation, S, i, r_immediate, succs = cut_info

                # Create beliefs for successors if they don't exist
                for g, p in succs:
                    successor_state = _merge_state(S, i, g)
                    if successor_state not in belief:
                        evidence = dict(_evidence(S))
                        evidence[i] = g
                        result_g = inf_cache.get(evidence)  # Changed to use cache
                        if not isinstance(result_g, InferenceResult):
                            result_g = InferenceResult(
                                result_g,
                                gene_order=gene_list if multi_gene else ("gene",),
                                gen_states=gen_states,
                            )
                        p_post_g = result_g
                        _store_belief(successor_state, p_post_g)
                        _ensure_gene_posteriors(successor_state, p_post_g)
                        if tuple_mode and result_g.has_tuple_pmfs():
                            _store_tuple_posteriors(successor_state, result_g.get_tuple_pmfs())
                        _register_projection(
                            successor_state,
                            successor_projection_cache.get((S, i, _normalize_outcome(g)))
                            if (multi_gene and tuple_mode)
                            else None,
                        )

                # Add canonical Bellman cut with tiny slack
                lhs = get_Phi(S)
                rhs = r_immediate + gp.quicksum(p * get_Phi(_merge_state(S, i, g)) for (g, p) in succs)
                bellman_name = f"bellman_{it}_{cuts_added}[{_state_label(S)},{i}]"
                bellman_cut = master.addConstr(lhs >= rhs - GAP_TOL, name=bellman_name)
                _register_bellman_row_constraint(
                    bellman_cut,
                    row_type="test",
                    state=S,
                    action=f"test:{i}",
                    person_tested=i,
                    immediate_reward=r_immediate,
                    successors=[(_merge_state(S, i, g), float(p)) for (g, p) in succs],
                    source="rowgen_test",
                )
                cuts_added += 1
                if verbose:
                    log_bellman_cut(S, i, violation, {'rhs': r_immediate, 'prob_sum': sum(p for g,p in succs)}, it)
        rowgen_cuts_added += cuts_added

        progress_coverage = len(cumulative_states_seen)
        progress_flag = False
        if progress_coverage > progress_coverage_anchor:
            progress_flag = True
            progress_coverage_anchor = progress_coverage
        if cuts_added > 0:
            progress_flag = True
        if (progress_gap_anchor - worst_gap) >= exhaustive_progress_eps:
            progress_flag = True
            progress_gap_anchor = worst_gap
        if progress_flag:
            runtime_last_progress_time = time.time()
            runtime_last_progress_iter = it
        
        if verbose:
            print(f"[BELLMAN] Added {cuts_added} cuts in iteration {it}")



        # optional debug dump each iteration
        if debug_lp_path:
            master.write(debug_lp_path)

    if rowgen_exit_reason is None:
        if rowgen_converged:
            rowgen_exit_reason = "converged"
        else:
            rowgen_exit_reason = "max_iters_nonconverged"
    runtime_walltime_sec = time.time() - runtime_start_time
    runtime_last_progress_sec_ago = max(0.0, time.time() - runtime_last_progress_time)
    runtime_exit_reason = rowgen_exit_reason

    rowgen_telemetry = {
        "telemetry_schema_version": telemetry_schema_version,
        "rowgen_states_total": int(rowgen_states_total),
        "rowgen_states_scanned": int(rowgen_states_scanned),
        "rowgen_states_truncated_count": int(rowgen_states_truncated_count),
        "rowgen_states_truncated": bool(rowgen_states_truncated_count > 0),
        "rowgen_candidate_cuts_total": int(rowgen_candidate_cuts_total),
        "rowgen_stop_cut_candidates_total": int(rowgen_stop_cut_candidates_total),
        "rowgen_test_cut_candidates_total": int(rowgen_test_cut_candidates_total),
        "rowgen_cuts_added": int(rowgen_cuts_added),
        "rowgen_cuts_truncated_count": int(rowgen_cuts_truncated_count),
        "rowgen_cuts_truncated": bool(rowgen_cuts_truncated_count > 0),
        "rowgen_pass_signature": rowgen_pass_signature,
        "rowgen_exit_reason": rowgen_exit_reason,
        "rowgen_converged": bool(rowgen_converged),
        "rowgen_last_worst_gap": float(rowgen_last_worst_gap),
        "rowgen_oracle_only_truncated_tuple_cuts_suppressed": int(
            rowgen_oracle_only_truncated_tuple_cuts_suppressed
        ),
        "exhaustive_mode_active": bool(exhaustive_bellman),
        "exhaustive_strict_active": bool(exhaustive_strict),
    }
    runtime_telemetry = {
        "runtime_walltime_sec": float(runtime_walltime_sec),
        "runtime_exit_reason": runtime_exit_reason,
        "runtime_last_progress_sec_ago": float(runtime_last_progress_sec_ago),
        "runtime_last_progress_iter": int(runtime_last_progress_iter),
        "runtime_heartbeat_count": int(runtime_heartbeat_count),
    }
    terminal_sidecar_status = "completed" if rowgen_exit_reason == "converged" else "failed"
    _write_runtime_sidecar(terminal_sidecar_status, runtime_exit_reason, rowgen_exit_reason)

    if rowgen_exit_reason != "converged":
        diagnostic_payload = {
            "rowgen_exit_reason": rowgen_exit_reason,
            "runtime_exit_reason": runtime_exit_reason,
            "rowgen_states_total": rowgen_states_total,
            "rowgen_states_scanned": rowgen_states_scanned,
            "rowgen_candidate_cuts_total": rowgen_candidate_cuts_total,
            "rowgen_stop_cut_candidates_total": rowgen_stop_cut_candidates_total,
            "rowgen_test_cut_candidates_total": rowgen_test_cut_candidates_total,
            "rowgen_cuts_added": rowgen_cuts_added,
            "rowgen_last_worst_gap": rowgen_last_worst_gap,
        }
        raise RuntimeError(
            "Row generation did not converge and cannot proceed to polish: "
            + json.dumps(diagnostic_payload, sort_keys=True)
        )

    def _current_theta_star():
        if stage_gene_theta_active:
            return {
                (k, gene): theta_stage_base_vars[k].X * stage_gene_shared_scale + theta_vars[k, gene].X
                for k in range(len(I) + 1)
                for gene in gene_list
            }
        if stage_theta_active:
            return [theta_vars[k].X for k in range(len(I) + 1)]
        if person_theta_active:
            return {person: theta_vars[person].X for person in I}
        if person_stage_theta_active:
            return {(person, k): theta_vars[person, k].X for person in I for k in range(1, len(I) + 1)}
        return theta_var.X

    def _current_w_solution():
        if per_gene_phi_active and W_gene_var is not None:
            w_gene_sol = _current_w_gene_solution() or {}
            w_sol = {i: {} for i in I}
            for outcome in available_outcomes:
                for i in I:
                    total = 0.0
                    for idx, gene in enumerate(gene_list):
                        comp = outcome[idx]
                        total += w_gene_sol[gene][i][comp]
                    w_sol[i][outcome] = total
            return w_sol
        return {i: {g: _get_w(i, g).X for g in available_outcomes} for i in I}

    def _current_w_gene_solution():
        if not (per_gene_phi_active and W_gene_var is not None):
            return None
        return {
            gene: {i: {g: _get_w_gene(gene, i, g).X for g in gen_states} for i in I}
            for gene in gene_list
        }

    def _current_phi_solution():
        phi_sol = {}
        for state, phi_var in Phi.items():
            try:
                phi_sol[state] = phi_var.X
            except Exception:
                continue
        return phi_sol

    def _current_aaub_star():
        if not aaub_apply:
            return None
        payload = {
            "fixed_enabled": aaub_u_vars is not None,
            "p12_enabled": aaub_v_vars is not None,
            "u": {person: aaub_u_vars[person].X for person in I} if aaub_u_vars is not None else {},
            "v": {person: aaub_v_vars[person].X for person in I} if aaub_v_vars is not None else {},
        }
        return payload

    def _current_edge_star():
        if not edge_features_active or W_edge_vars is None:
            return None
        block_payload = {}
        for block in edge_feature_blocks:
            if per_gene_phi_active:
                block_payload[block] = {
                    gene: {
                        (p, c): {
                            (gp, gc): W_edge_vars[block, gene, (p, c), gp, gc].X
                            for gp in gen_states
                            for gc in gen_states
                        }
                        for p, c in pedigree_edges
                    }
                    for gene in gene_list
                }
            else:
                block_payload[block] = {
                    (p, c): {
                        (gp, gc): W_edge_vars[block, (p, c), gp, gc].X
                        for gp in gen_states
                        for gc in gen_states
                    }
                    for p, c in pedigree_edges
                }

        if edge_feature_mode == "raw":
            return block_payload["raw"]
        return {
            "__mode__": edge_feature_mode,
            "__blocks__": list(edge_feature_blocks),
            **block_payload,
        }

    def _resolve_posterior_entry(state):
        posterior_entry, _ = belief[state]
        p_s = _posterior_marginals(posterior_entry)
        gene_probs = _ensure_gene_posteriors(state, posterior_entry)
        return posterior_entry, p_s, (gene_probs or {})

    def _decode_dfvr_state(state_items):
        if not isinstance(state_items, list):
            return None
        parsed = {}
        for item in state_items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None
            person = item[0]
            outcome = item[1]
            if isinstance(outcome, list):
                outcome = tuple(outcome)
            parsed[person] = _normalize_outcome(outcome)
        return frozenset(parsed.items())

    def _ensure_successor_belief(state, person, outcome):
        successor_state = _merge_state(state, person, outcome)
        if successor_state in belief:
            return successor_state
        evidence = dict(_evidence(state))
        evidence[person] = outcome
        result = inf_cache.get(evidence)
        if not isinstance(result, InferenceResult):
            result = InferenceResult(
                result,
                gene_order=gene_list if multi_gene else ("gene",),
                gen_states=gen_states,
            )
        _store_belief(successor_state, result)
        _ensure_gene_posteriors(successor_state, result)
        if tuple_mode and result.has_tuple_pmfs():
            _store_tuple_posteriors(successor_state, result.get_tuple_pmfs())
        _register_projection(
            successor_state,
            successor_projection_cache.get((state, person, _normalize_outcome(outcome)))
            if (multi_gene and tuple_mode)
            else None,
        )
        return successor_state

    def _action_rhs_expr(state, action):
        action_kind = action.get("kind") if isinstance(action, Mapping) else None
        if action_kind not in {"stop", "test"}:
            return None
        posterior_entry, p_s, gene_probs = _resolve_posterior_entry(state)
        tested = {person for person, _ in _evidence(state)}
        untested = [person for person in I if person not in tested]
        if action_kind == "stop":
            rhs_const = sum(
                r_reward(
                    person,
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
                for person in untested
            )
            return gp.LinExpr(rhs_const)

        person = action.get("person")
        if person not in untested:
            return None
        person_probs = p_s.get(person, {})
        p12 = person_probs.get(1, 0.0) + person_probs.get(2, 0.0)
        per_gene_p12 = _per_gene_p12_map(gene_probs, person)
        immediate_reward = r_reward_testp(
            person,
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
        if tuple_mode and tuple_posteriors.get(state, {}).get(person):
            dist = tuple_posteriors[state][person]
        else:
            dist = p_s.get(person, {})

        rhs_expr = gp.LinExpr(immediate_reward)
        prob_total = 0.0
        for outcome, prob in dist.items():
            if prob <= 0.0:
                continue
            prob_total += prob
            succ_state = _ensure_successor_belief(state, person, outcome)
            rhs_expr.add(prob * get_Phi(succ_state))
        if prob_total <= 0.0:
            return None
        return rhs_expr

    def _evaluate_dfvr(top_k):
        from ..optimisation.dfvr_bound import compute_dfvr_bound

        config_obj = SimpleNamespace(
            a=a,
            b=b,
            c=c,
            delta=delta,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
        )
        result = compute_dfvr_bound(
            Phi_star=_current_phi_solution(),
            W_star=_current_w_solution(),
            belief=belief,
            theta_star=_current_theta_star(),
            theta_mode=theta_mode,
            individuals=I,
            gen_states=gen_states,
            infer=infer,
            config=config_obj,
            tuple_mode=tuple_mode,
            genes=gene_list if multi_gene else None,
            state_mode="belief",
            max_states=None,
            top_k=top_k,
            max_outcomes_per_action=10,
            aaub_star=_current_aaub_star(),
            W_edge_star=_current_edge_star(),
            W_trio_star=_current_trio_star(),
            pedigree_edges=pedigree_edges if edge_features_active else None,
            pedigree_trios=pedigree_trios if trio_features_active else None,
            regime_residual_star=_current_regime_residual_star(),
            no_mutation=dfvr_eval_no_mutation,
            fixed_states=dfvr_fixed_states,
            enforce_fixed_states=dfvr_enforce_fixed_stateset and dfvr_fixed_states is not None,
        )
        if dfvr_fixed_states is not None and dfvr_enforce_fixed_stateset:
            integrity = result.get("fixed_state_integrity")
            if not isinstance(integrity, Mapping) or not integrity.get("match", False):
                raise RuntimeError(f"DFVR fixed state-set integrity failed in candidate solve: {integrity!r}")
        return result

    def _build_hotspot_entries(top_residuals, limit):
        hotspots = []
        if not isinstance(top_residuals, list):
            return hotspots
        for payload in top_residuals:
            if not isinstance(payload, Mapping):
                continue
            state = _decode_dfvr_state(payload.get("state"))
            action = payload.get("action")
            if state is None or not isinstance(action, Mapping):
                continue
            if action.get("kind") not in {"stop", "test"}:
                continue
            try:
                scaled = float(payload.get("scaled", 0.0) or 0.0)
            except (TypeError, ValueError):
                scaled = 0.0
            try:
                residual = float(payload.get("residual", 0.0) or 0.0)
            except (TypeError, ValueError):
                residual = 0.0
            hotspots.append(
                {
                    "state": state,
                    "action": {"kind": action.get("kind"), "person": action.get("person")},
                    "scaled": scaled,
                    "residual": residual,
                }
            )
            if len(hotspots) >= limit:
                break
        return hotspots

    def _collect_trust_region_states(hotspots):
        trust_states = set()
        for hotspot in hotspots:
            state = hotspot.get("state")
            action = hotspot.get("action")
            if state is None:
                continue
            trust_states.add(state)
            if not isinstance(action, Mapping):
                continue
            if action.get("kind") != "test":
                continue
            person = action.get("person")
            if not isinstance(person, str):
                continue
            posterior_entry, p_s, _ = _resolve_posterior_entry(state)
            del posterior_entry
            if tuple_mode and tuple_posteriors.get(state, {}).get(person):
                dist = tuple_posteriors[state][person]
            else:
                dist = p_s.get(person, {})
            for outcome, prob in dist.items():
                if prob <= 0.0:
                    continue
                succ_state = _ensure_successor_belief(state, person, outcome)
                trust_states.add(succ_state)
        return sorted(trust_states, key=_state_label)

    def _build_candidate_snapshot(
        *,
        candidate_id,
        phi_star,
        w_star,
        w_gene_star,
        theta_star_value,
        aaub_value,
        edge_value=None,
        trio_value=None,
        myopic_adp_value=None,
        oracle_adp_value=None,
        regime_residual_value=None,
    ):
        phi_root = None
        try:
            phi_root = float(phi_star.get(s0)) if isinstance(phi_star, dict) else None
        except Exception:
            phi_root = None
        objective_value = None
        if master.status == GRB.Status.OPTIMAL:
            try:
                objective_value = float(master.ObjVal)
            except Exception:
                objective_value = None
        return {
            "candidate_id": candidate_id,
            "theta_mode": theta_mode,
            "phi_root": phi_root,
            "policy_eval_inputs_ref": f"{candidate_id}_payload",
            "rowgen_pass_signature": rowgen_pass_signature,
            "slack_refactor": {
                "enabled": bool(slack_polish_active or aaub_apply),
                "track_a_active": bool(slack_polish_active),
                "track_b_active": bool(aaub_apply),
            },
            "root_diagnostics": {
                "model_status": int(master.status),
                "objective_value": objective_value,
            },
            "payload": {
                "candidate_id": candidate_id,
                "Phi_star": phi_star,
                "W_star": w_star,
                "W_star_gene": w_gene_star,
                "theta_star": theta_star_value,
                "aaub_star": aaub_value,
                "edge_star": edge_value,
                "trio_star": trio_value,
                "myopic_adp_star": myopic_adp_value,
                "oracle_adp_star": oracle_adp_value,
                "regime_residual_star": regime_residual_value,
                "belief": belief,
                "tuple_pmfs": tuple_posteriors if tuple_mode else None,
            },
        }

    # final polish
    master.optimize()
    master.update()  # Ensure constraint attributes are available
    pre_phi_star = _current_phi_solution()
    pre_theta_star = _current_theta_star()
    pre_w_star = _current_w_solution()
    pre_w_gene_star = _current_w_gene_solution()
    pre_aaub_star = _current_aaub_star()
    pre_edge_star = _current_edge_star()
    pre_trio_star = _current_trio_star()
    pre_myopic_adp_star = _current_myopic_adp_star()
    pre_oracle_adp_star = _current_oracle_adp_star()
    pre_regime_residual_star = _current_regime_residual_star()
    candidate_pre_polish = _build_candidate_snapshot(
        candidate_id="pre_polish",
        phi_star=pre_phi_star,
        w_star=pre_w_star,
        w_gene_star=pre_w_gene_star,
        theta_star_value=pre_theta_star,
        aaub_value=pre_aaub_star,
        edge_value=pre_edge_star,
        trio_value=pre_trio_star,
        myopic_adp_value=pre_myopic_adp_star,
        oracle_adp_value=pre_oracle_adp_star,
        regime_residual_value=pre_regime_residual_star,
    )
    tail_regularization = {
        "enabled": tail_phi_reg_enable,
        "active_for_mode": tail_phi_reg_active,
        "theta_mode": theta_mode,
        "modes": sorted(tail_phi_reg_modes),
        "min_stage": tail_phi_reg_min_stage,
        "weight": tail_phi_reg_weight,
        "max_states": tail_phi_reg_max_states,
        "max_root_delta": tail_phi_reg_max_root_delta,
        "applied": False,
    }
    root_var = get_Phi(s0)
    slack_refactor = {
        "enabled": bool(slack_polish_active or aaub_apply),
        "theta_mode": theta_mode,
        "track_a": {
            "enabled": slack_polish_enable,
            "active_for_mode": slack_polish_active,
            "topk": slack_polish_topk,
            "rounds": slack_polish_rounds,
            "objective": "minmax" if hotspot_minmax_enable else "weighted_residual",
            "max_root_delta": slack_polish_max_root_delta,
            "phi_delta_max": slack_polish_phi_delta_max,
            "dfvr_eval_no_mutation": dfvr_eval_no_mutation,
            "fixed_states_count": len(dfvr_fixed_states) if dfvr_fixed_states is not None else None,
            "round_telemetry": [],
        },
        "track_b": {
            "enabled": aaub_enable,
            "active_for_mode": aaub_apply,
            "fixed_enabled": aaub_u_vars is not None,
            "p12_enabled": aaub_v_vars is not None,
            "sign_safe_mode": aaub_sign_safe_mode,
            "max_root_delta": aaub_max_root_delta,
            "coef_abs_cap": aaub_coef_abs_cap,
            "applied": False,
        },
    }

    def _build_hotspot_objective_round(*, label, hotspots, root_anchor, max_root_delta, phi_delta_cap=None):
        round_constraints = []
        round_vars = []
        weights = []
        for idx, hotspot in enumerate(hotspots):
            state = hotspot["state"]
            action = hotspot["action"]
            rhs_expr = _action_rhs_expr(state, action)
            if rhs_expr is None:
                continue
            residual_var = master.addVar(lb=0.0, name=f"{label}_residual_{idx}")
            round_vars.append(residual_var)
            round_constraints.append(
                master.addConstr(
                    residual_var >= get_Phi(state) - rhs_expr,
                    name=f"{label}_residual_link_{idx}",
                )
            )
            weight = hotspot.get("scaled", 0.0)
            if weight <= 0.0:
                weight = hotspot.get("residual", 0.0)
            if weight <= 0.0:
                weight = 1.0
            weights.append(float(weight))

        if not round_vars:
            return None

        round_constraints.append(
            master.addConstr(
                root_var >= root_anchor - max_root_delta,
                name=f"{label}_root_lb",
            )
        )
        round_constraints.append(
            master.addConstr(
                root_var <= root_anchor + max_root_delta,
                name=f"{label}_root_ub",
            )
        )

        if phi_delta_cap is not None:
            trust_states = _collect_trust_region_states(hotspots)
            for idx, state in enumerate(trust_states):
                phi_var = get_Phi(state)
                try:
                    phi_anchor = float(phi_var.X)
                except Exception:
                    # Skip freshly materialized states that do not yet have
                    # a solved incumbent value in the current master solution.
                    continue
                round_constraints.append(
                    master.addConstr(
                        phi_var >= phi_anchor - phi_delta_cap,
                        name=f"{label}_phi_lb_{idx}",
                    )
                )
                round_constraints.append(
                    master.addConstr(
                        phi_var <= phi_anchor + phi_delta_cap,
                        name=f"{label}_phi_ub_{idx}",
                    )
                )

        if hotspot_minmax_enable:
            t_var = master.addVar(lb=0.0, name=f"{label}_residual_max")
            for idx, residual_var in enumerate(round_vars):
                round_constraints.append(
                    master.addConstr(
                        residual_var <= t_var,
                        name=f"{label}_residual_max_link_{idx}",
                    )
                )
            residual_avg = gp.quicksum(round_vars) / float(len(round_vars))
            master.setObjective(t_var + (1e-6 * residual_avg), GRB.MINIMIZE)
            cleanup_vars = list(round_vars) + [t_var]
        else:
            residual_expr = gp.LinExpr()
            for var, weight in zip(round_vars, weights):
                residual_expr.add(weight * var)
            normalizer = max(1.0, float(len(round_vars)))
            master.setObjective(root_var + (1e-6 * residual_expr / normalizer), GRB.MINIMIZE)
            cleanup_vars = list(round_vars)
        return {
            "vars": cleanup_vars,
            "constraints": round_constraints,
            "count": len(round_vars),
        }

    def _remove_round(round_payload):
        if not round_payload:
            return
        objs = []
        objs.extend(round_payload.get("constraints", []))
        objs.extend(round_payload.get("vars", []))
        if objs:
            master.remove(objs)
            master.update()

    if aaub_apply and master.status == GRB.Status.OPTIMAL:
        track_b = slack_refactor["track_b"]
        aaub_vars = []
        if aaub_u_vars is not None:
            aaub_vars.extend(aaub_u_vars[person] for person in I)
        if aaub_v_vars is not None:
            aaub_vars.extend(aaub_v_vars[person] for person in I)

        root_anchor_no_aaub = float(root_var.X)
        bounds_snapshot = [(var, var.LB, var.UB) for var in aaub_vars]
        for var in aaub_vars:
            var.LB = 0.0
            var.UB = 0.0
        master.optimize()
        master.update()
        if master.status == GRB.Status.OPTIMAL:
            root_anchor_no_aaub = float(root_var.X)
        for var, lb, ub in bounds_snapshot:
            var.LB = lb
            var.UB = ub
        master.optimize()
        master.update()
        if master.status != GRB.Status.OPTIMAL:
            raise RuntimeError(f"AAUB optimize failed before polish: status={master.status}.")

        dfvr_before = _evaluate_dfvr(slack_polish_topk)
        hotspots = _build_hotspot_entries(dfvr_before.get("top_residuals"), slack_polish_topk)
        before_residual = float(dfvr_before.get("residual_norm", 0.0))
        before_signature = (dfvr_before.get("coverage") or {}).get("state_signature")
        track_b["root_phi_anchor_no_aaub"] = root_anchor_no_aaub
        track_b["residual_norm_before"] = before_residual
        track_b["dfvr_bound_before"] = dfvr_before.get("dfvr_bound")
        track_b["hotspots"] = len(hotspots)

        if hotspots:
            round_payload = _build_hotspot_objective_round(
                label="aaub",
                hotspots=hotspots,
                root_anchor=root_anchor_no_aaub,
                max_root_delta=aaub_max_root_delta,
                phi_delta_cap=None,
            )
            if round_payload is not None:
                master.optimize()
                master.update()
                if master.status != GRB.Status.OPTIMAL:
                    _remove_round(round_payload)
                    master.setObjective(root_var, GRB.MINIMIZE)
                    master.optimize()
                    master.update()
                else:
                    dfvr_after = _evaluate_dfvr(slack_polish_topk)
                    after_residual = float(dfvr_after.get("residual_norm", 0.0))
                    after_signature = (dfvr_after.get("coverage") or {}).get("state_signature")
                    integrity_after = dfvr_after.get("fixed_state_integrity")
                    fixed_match = (
                        not isinstance(integrity_after, Mapping)
                        or bool(integrity_after.get("match", True))
                    )
                    signature_match = (
                        not before_signature
                        or not after_signature
                        or before_signature == after_signature
                    )
                    if fixed_match and signature_match and (after_residual + 1e-12 < before_residual):
                        track_b["residual_norm_after"] = after_residual
                        track_b["dfvr_bound_after"] = dfvr_after.get("dfvr_bound")
                        track_b["applied"] = True
                    else:
                        _remove_round(round_payload)
                        master.setObjective(root_var, GRB.MINIMIZE)
                        master.optimize()
                        master.update()
                        reverted = _evaluate_dfvr(slack_polish_topk)
                        track_b["residual_norm_after"] = reverted.get("residual_norm")
                        track_b["dfvr_bound_after"] = reverted.get("dfvr_bound")
                        if not fixed_match:
                            track_b["reason"] = "fixed_state_integrity_mismatch"
                        elif not signature_match:
                            track_b["reason"] = "state_signature_mismatch"
                        else:
                            track_b["reason"] = "non_improving_residual"
        track_b["aaub_coefficients"] = _current_aaub_star()

    if slack_polish_active and master.status == GRB.Status.OPTIMAL:
        track_a = slack_refactor["track_a"]
        current_dfvr = _evaluate_dfvr(slack_polish_topk)
        best_residual = float(current_dfvr.get("residual_norm", 0.0))
        expected_signature = (current_dfvr.get("coverage") or {}).get("state_signature")
        track_a["residual_norm_before"] = best_residual
        track_a["dfvr_bound_before"] = current_dfvr.get("dfvr_bound")
        for round_idx in range(slack_polish_rounds):
            hotspots = _build_hotspot_entries(current_dfvr.get("top_residuals"), slack_polish_topk)
            round_info = {
                "round": round_idx + 1,
                "hotspots": len(hotspots),
                "accepted": False,
            }
            if not hotspots:
                round_info["reason"] = "no_hotspots"
                track_a["round_telemetry"].append(round_info)
                break
            root_anchor = float(root_var.X)
            round_payload = _build_hotspot_objective_round(
                label=f"slack_polish_r{round_idx + 1}",
                hotspots=hotspots,
                root_anchor=root_anchor,
                max_root_delta=slack_polish_max_root_delta,
                phi_delta_cap=slack_polish_phi_delta_max,
            )
            if round_payload is None:
                round_info["reason"] = "no_valid_hotspot_constraints"
                track_a["round_telemetry"].append(round_info)
                break
            master.optimize()
            master.update()
            if master.status != GRB.Status.OPTIMAL:
                round_info["reason"] = f"opt_status_{master.status}"
                _remove_round(round_payload)
                master.setObjective(root_var, GRB.MINIMIZE)
                master.optimize()
                master.update()
                track_a["round_telemetry"].append(round_info)
                break

            candidate_dfvr = _evaluate_dfvr(slack_polish_topk)
            candidate_residual = float(candidate_dfvr.get("residual_norm", 0.0))
            candidate_signature = (candidate_dfvr.get("coverage") or {}).get("state_signature")
            integrity = candidate_dfvr.get("fixed_state_integrity")
            fixed_match = (
                not isinstance(integrity, Mapping)
                or bool(integrity.get("match", True))
            )
            round_info["residual_before"] = best_residual
            round_info["residual_after"] = candidate_residual
            round_info["dfvr_bound_after"] = candidate_dfvr.get("dfvr_bound")
            if isinstance(integrity, Mapping):
                round_info["fixed_state_match"] = bool(integrity.get("match", False))
            if expected_signature and candidate_signature:
                round_info["state_signature_match"] = (candidate_signature == expected_signature)

            if not fixed_match:
                round_info["reason"] = "fixed_state_integrity_mismatch"
                _remove_round(round_payload)
                master.setObjective(root_var, GRB.MINIMIZE)
                master.optimize()
                master.update()
                current_dfvr = _evaluate_dfvr(slack_polish_topk)
            elif expected_signature and candidate_signature and candidate_signature != expected_signature:
                round_info["reason"] = "state_signature_mismatch"
                _remove_round(round_payload)
                master.setObjective(root_var, GRB.MINIMIZE)
                master.optimize()
                master.update()
                current_dfvr = _evaluate_dfvr(slack_polish_topk)
            elif candidate_residual + 1e-12 < best_residual:
                round_info["accepted"] = True
                best_residual = candidate_residual
                current_dfvr = candidate_dfvr
                expected_signature = candidate_signature or expected_signature
            else:
                round_info["reason"] = "non_improving_residual"
                _remove_round(round_payload)
                master.setObjective(root_var, GRB.MINIMIZE)
                master.optimize()
                master.update()
                current_dfvr = _evaluate_dfvr(slack_polish_topk)
            track_a["round_telemetry"].append(round_info)

        track_a["residual_norm_after"] = current_dfvr.get("residual_norm")
        track_a["dfvr_bound_after"] = current_dfvr.get("dfvr_bound")
    
    # Safely get slack value with error handling
    try:
        root_constraint = master.getConstrByName("root_stop")
        if root_constraint is not None and master.status == GRB.Status.OPTIMAL:
            root_slack = root_constraint.Slack
        else:
            root_slack = 0.0
    except Exception:
        root_slack = 0.0
        
    if verbose:
        print(f"[debug] Φ(root) = {Phi[s0].X: .6f}   "
              f"R_stop(root) = {Rstop_root: .6f}   "
              f"slack = {root_slack: .6e}")
    if tail_phi_reg_active and master.status == GRB.Status.OPTIMAL and tail_phi_reg_weight > 0.0:
        # Apply a second-stage objective that keeps root Φ near its incumbent and
        # adds pressure to reduce deep-state Φ values where DFVR hotspots occur.
        root_phi_anchor = float(get_Phi(s0).X)
        deep_states = [
            state
            for state in Phi
            if tail_phi_reg_min_stage <= len(_evidence(state)) < len(I)
        ]
        deep_states.sort(key=lambda state: (-len(_evidence(state)), _state_label(state)))
        if tail_phi_reg_max_states > 0 and len(deep_states) > tail_phi_reg_max_states:
            deep_states = deep_states[:tail_phi_reg_max_states]

        tail_regularization.update(
            {
                "root_phi_before": root_phi_anchor,
                "deep_states_considered": len(deep_states),
            }
        )
        if deep_states:
            deep_mean_before = float(sum(get_Phi(state).X for state in deep_states) / len(deep_states))
            old_lb = master.getConstrByName("tail_reg_root_lb")
            old_ub = master.getConstrByName("tail_reg_root_ub")
            if old_lb is not None:
                master.remove(old_lb)
            if old_ub is not None:
                master.remove(old_ub)
            master.update()

            root_var = get_Phi(s0)
            master.addConstr(
                root_var >= root_phi_anchor - tail_phi_reg_max_root_delta,
                name="tail_reg_root_lb",
            )
            master.addConstr(
                root_var <= root_phi_anchor + tail_phi_reg_max_root_delta,
                name="tail_reg_root_ub",
            )
            deep_expr = gp.quicksum(get_Phi(state) for state in deep_states) / float(len(deep_states))
            master.setObjective(root_var + tail_phi_reg_weight * deep_expr, GRB.MINIMIZE)
            master.optimize()
            master.update()
            if master.status != GRB.Status.OPTIMAL:
                raise RuntimeError(
                    "Tail deep-state regularization reoptimize failed: "
                    f"status={master.status}."
                )
            deep_mean_after = float(sum(get_Phi(state).X for state in deep_states) / len(deep_states))
            root_phi_after = float(get_Phi(s0).X)
            tail_regularization.update(
                {
                    "applied": True,
                    "root_phi_after": root_phi_after,
                    "deep_mean_before": deep_mean_before,
                    "deep_mean_after": deep_mean_after,
                }
            )
            if verbose:
                print(
                    "[tail_phi_reg] applied: "
                    f"root_before={root_phi_anchor:.6f} root_after={root_phi_after:.6f} "
                    f"deep_mean_before={deep_mean_before:.6f} deep_mean_after={deep_mean_after:.6f} "
                    f"states={len(deep_states)}"
                )
        else:
            tail_regularization["reason"] = "no_deep_states_selected"
    elif tail_phi_reg_active and tail_phi_reg_weight <= 0.0:
        tail_regularization["reason"] = "non_positive_weight"

    secondary_objective_applied = False
    secondary_objective_stage_states = 0
    secondary_objective_root_anchor = None
    secondary_objective_reason = None
    if secondary_phi_objective == "stage12_mean" and master.status == GRB.Status.OPTIMAL:
        stage_states = [
            state for state in belief
            if 1 <= len(_evidence(state)) <= 2 and len(_evidence(state)) < len(I)
        ]
        stage_states.sort(key=_state_label)
        secondary_objective_stage_states = len(stage_states)
        if stage_states:
            root_var = get_Phi(s0)
            secondary_objective_root_anchor = float(root_var.X)
            old_lb = master.getConstrByName("secondary_phi_root_lb")
            old_ub = master.getConstrByName("secondary_phi_root_ub")
            if old_lb is not None:
                master.remove(old_lb)
            if old_ub is not None:
                master.remove(old_ub)
            master.update()
            master.addConstr(
                root_var >= secondary_objective_root_anchor - secondary_phi_root_tol,
                name="secondary_phi_root_lb",
            )
            master.addConstr(
                root_var <= secondary_objective_root_anchor + secondary_phi_root_tol,
                name="secondary_phi_root_ub",
            )
            stage_mean_expr = gp.quicksum(get_Phi(state) for state in stage_states) / float(len(stage_states))
            master.setObjective(root_var + (1e-6 * stage_mean_expr), GRB.MINIMIZE)
            master.optimize()
            master.update()
            if master.status == GRB.Status.OPTIMAL:
                secondary_objective_applied = True
            else:
                secondary_objective_reason = f"opt_status_{master.status}"
        else:
            secondary_objective_reason = "no_stage12_states"

    def _collect_root_constraint_diagnostics(limit: int = 8):
        root_entries = []
        for constr in master.getConstrs():
            name = constr.ConstrName or ""
            if not (
                name == "root_stop"
                or name.startswith("init_test_")
                or name.startswith("tail_reg_root_")
                or name.endswith("_root_lb")
                or name.endswith("_root_ub")
            ):
                continue
            entry = {
                "name": name,
                "sense": constr.Sense,
                "rhs": float(constr.RHS),
            }
            if master.status == GRB.Status.OPTIMAL:
                try:
                    entry["slack"] = float(constr.Slack)
                except Exception:
                    entry["slack"] = None
            root_entries.append(entry)

        if master.status == GRB.Status.OPTIMAL:
            root_entries.sort(
                key=lambda item: (
                    abs(item["slack"]) if isinstance(item.get("slack"), (int, float)) else float("inf"),
                    item["name"],
                )
            )
        else:
            root_entries.sort(key=lambda item: item["name"])

        return root_entries[:limit], len(root_entries)

    root_constraint = master.getConstrByName("root_stop")
    tight_root_constraints, root_constraint_count = _collect_root_constraint_diagnostics(limit=8)
    root_phi_lp = None
    root_stop_rhs = None
    root_stop_slack = None
    objective_value = None
    if master.status == GRB.Status.OPTIMAL:
        try:
            root_phi_lp = float(get_Phi(s0).X)
        except Exception:
            root_phi_lp = None
        try:
            objective_value = float(master.ObjVal)
        except Exception:
            objective_value = None
    if root_constraint is not None:
        try:
            root_stop_rhs = float(root_constraint.RHS)
        except Exception:
            root_stop_rhs = None
        if master.status == GRB.Status.OPTIMAL:
            try:
                root_stop_slack = float(root_constraint.Slack)
            except Exception:
                root_stop_slack = None

    seed_init_test_rhs_values = []
    if master.status == GRB.Status.OPTIMAL and root_phi_lp is not None:
        for name in seed_root_init_constraint_names:
            constr = master.getConstrByName(name)
            if constr is None:
                continue
            try:
                slack_val = float(constr.Slack)
            except Exception:
                continue
            seed_init_test_rhs_values.append(root_phi_lp - slack_val)
    seed_init_test_rhs_min = min(seed_init_test_rhs_values) if seed_init_test_rhs_values else None
    seed_init_test_rhs_max = max(seed_init_test_rhs_values) if seed_init_test_rhs_values else None
    seed_init_test_rhs_nonnegative_count = int(
        sum(1 for rhs_val in seed_init_test_rhs_values if rhs_val >= 0.0)
    ) if seed_init_test_rhs_values else 0
    bellman_row_dual_export = _export_bellman_row_duals()
    bellman_row_dual_validation = bellman_row_dual_export.get("validation", {})

    root_diagnostics = {
        "model_status": int(master.status),
        "objective_value": objective_value,
        "objective_sense": "minimize",
        "root_state_size": len(_evidence(s0)),
        "root_phi_lp": root_phi_lp,
        "root_stop_rhs": root_stop_rhs,
        "root_stop_seed_rhs": float(Rstop_root),
        "root_stop_slack": root_stop_slack,
        "root_constraint_count": root_constraint_count,
        "root_constraint_tightest": tight_root_constraints,
        "seed_scope": effective_seed_scope,
        "edge_seed_scope": edge_seed_scope,
        "trio_seed_scope": trio_seed_scope,
        "seed_stage1_state_count": int(stage1_seed_state_count),
        "seed_stage1_constraint_count": int(len(seed_stage1_constraint_names)),
        "trio_stage2_clinical_seed_state_count": int(stage2_clinical_seed_state_count),
        "seed_init_test_constraint_count": int(len(seed_root_init_constraint_names)),
        "seed_init_test_rhs_min": seed_init_test_rhs_min,
        "seed_init_test_rhs_max": seed_init_test_rhs_max,
        "seed_init_test_rhs_nonnegative_count": int(seed_init_test_rhs_nonnegative_count),
        "bellman_row_dual_export": bellman_row_dual_export,
        "dual_component_available": bool(bellman_row_dual_validation.get("dual_component_available", False)),
        "nonzero_dual_row_count": int(bellman_row_dual_export.get("nonzero_dual_row_count") or 0),
        "aggregated_dual_row_count": int(bellman_row_dual_export.get("aggregated_row_count") or 0),
        "truncated_nonzero_dual_row_count": int(
            bellman_row_dual_export.get("truncated_nonzero_dual_row_count") or 0
        ),
        "max_dual_complementarity_abs": float(
            bellman_row_dual_export.get("max_complementarity_abs") or 0.0
        ),
        "bellman_row_dual_validation": dict(bellman_row_dual_validation),
        "secondary_phi_objective": secondary_phi_objective,
        "secondary_phi_objective_applied": bool(secondary_objective_applied),
        "secondary_phi_objective_root_anchor": secondary_objective_root_anchor,
        "secondary_phi_objective_stage_states": int(secondary_objective_stage_states),
        "secondary_phi_objective_reason": secondary_objective_reason,
        "slack_refactor_enabled": bool(slack_refactor.get("enabled")),
        "tail_regularization_applied": bool(tail_regularization.get("applied", False)),
        "telemetry_schema_version": telemetry_schema_version,
        "rowgen_states_total": rowgen_telemetry["rowgen_states_total"],
        "rowgen_states_scanned": rowgen_telemetry["rowgen_states_scanned"],
        "rowgen_states_truncated_count": rowgen_telemetry["rowgen_states_truncated_count"],
        "rowgen_states_truncated": rowgen_telemetry["rowgen_states_truncated"],
        "rowgen_candidate_cuts_total": rowgen_telemetry["rowgen_candidate_cuts_total"],
        "rowgen_stop_cut_candidates_total": rowgen_telemetry["rowgen_stop_cut_candidates_total"],
        "rowgen_test_cut_candidates_total": rowgen_telemetry["rowgen_test_cut_candidates_total"],
        "rowgen_cuts_added": rowgen_telemetry["rowgen_cuts_added"],
        "rowgen_cuts_truncated_count": rowgen_telemetry["rowgen_cuts_truncated_count"],
        "rowgen_cuts_truncated": rowgen_telemetry["rowgen_cuts_truncated"],
        "rowgen_pass_signature": rowgen_telemetry["rowgen_pass_signature"],
        "rowgen_exit_reason": rowgen_telemetry["rowgen_exit_reason"],
        "rowgen_converged": rowgen_telemetry["rowgen_converged"],
        "rowgen_last_worst_gap": rowgen_telemetry["rowgen_last_worst_gap"],
        "rowgen_oracle_only_truncated_tuple_cuts_suppressed": rowgen_telemetry[
            "rowgen_oracle_only_truncated_tuple_cuts_suppressed"
        ],
        "exhaustive_mode_active": rowgen_telemetry["exhaustive_mode_active"],
        "exhaustive_strict_active": rowgen_telemetry["exhaustive_strict_active"],
        "runtime_walltime_sec": runtime_telemetry["runtime_walltime_sec"],
        "runtime_exit_reason": runtime_telemetry["runtime_exit_reason"],
        "runtime_last_progress_sec_ago": runtime_telemetry["runtime_last_progress_sec_ago"],
        "runtime_last_progress_iter": runtime_telemetry["runtime_last_progress_iter"],
        "runtime_heartbeat_count": runtime_telemetry["runtime_heartbeat_count"],
        "trio_feature_cache_hits": int(trio_feature_cache_hits_total),
        "trio_feature_cache_misses": int(trio_feature_cache_misses_total),
        "trio_feature_materialization_sec_total": float(trio_feature_materialization_sec_total),
    }

    # --- optional diagnostic: ensure no blank‑LHS rows were added ---
    for r in master.getConstrs():
        # r.Sense is one of '=', '<', '>' (for ≥ rows it’s '>')
        if r.Sense == '>':
            # r.RHS is the right‑hand‑side constant
            if abs(r.RHS) > tol:              # only care about non‑zero RHS
                # master.getRow(r) returns a LinExpr of LHS
                if master.getRow(r).size() == 0:
                    raise RuntimeError(f"❌ Constraint {r.ConstrName!r} has empty LHS but RHS={r.RHS}")

    # gather solutions
    Phi_sol = _current_phi_solution()
    theta_star = _current_theta_star()
    W_sol = _current_w_solution()
    W_gene_sol = _current_w_gene_solution()
    aaub_star = _current_aaub_star()
    edge_star = _current_edge_star()
    trio_star = _current_trio_star()
    myopic_adp_star = _current_myopic_adp_star()
    oracle_adp_star = _current_oracle_adp_star()
    regime_residual_star = _current_regime_residual_star()
    regime_residual_star_current = _current_regime_residual_star()

    edge_coef_l1, edge_coef_l2, edge_coef_nonzero = _feature_coef_summary(edge_star)
    trio_coef_l1, trio_coef_l2, trio_coef_nonzero = _feature_coef_summary(trio_star)
    oracle_coeffs = (
        oracle_adp_star.get("coefficients", {})
        if isinstance(oracle_adp_star, Mapping)
        else {}
    )
    oracle_coef_l1, oracle_coef_l2, oracle_coef_nonzero = _feature_coef_summary(oracle_coeffs)
    regime_coeffs = (
        regime_residual_star.get("coefficients", {})
        if isinstance(regime_residual_star, Mapping)
        else {}
    )
    regime_coef_l1, regime_coef_l2, regime_coef_nonzero = _feature_coef_summary(regime_coeffs)
    root_diagnostics["edge_feature_mode"] = edge_feature_mode
    root_diagnostics["edge_feature_blocks"] = list(edge_feature_blocks)
    root_diagnostics["edge_coef_l1"] = float(edge_coef_l1)
    root_diagnostics["edge_coef_l2"] = float(edge_coef_l2)
    root_diagnostics["edge_coef_nonzero"] = int(edge_coef_nonzero)
    root_diagnostics["trio_feature_mode"] = trio_feature_mode
    root_diagnostics["trio_feature_blocks"] = list(trio_feature_blocks)
    root_diagnostics["trio_coef_sharing"] = trio_coef_sharing
    root_diagnostics["trio_coef_l1"] = float(trio_coef_l1)
    root_diagnostics["trio_coef_l2"] = float(trio_coef_l2)
    root_diagnostics["trio_coef_nonzero"] = int(trio_coef_nonzero)
    root_diagnostics["trio_feature_cache_hits"] = int(trio_feature_cache_hits_total)
    root_diagnostics["trio_feature_cache_misses"] = int(trio_feature_cache_misses_total)
    root_diagnostics["trio_feature_materialization_sec_total"] = float(trio_feature_materialization_sec_total)
    root_diagnostics["myopic_adp"] = serializable_summary(myopic_adp_star)
    root_diagnostics["oracle_adp"] = oracle_serializable_summary(oracle_adp_star)
    root_diagnostics["oracle_coef_l1"] = float(oracle_coef_l1)
    root_diagnostics["oracle_coef_l2"] = float(oracle_coef_l2)
    root_diagnostics["oracle_coef_nonzero"] = int(oracle_coef_nonzero)
    root_diagnostics["gauged_regime_residual_enabled"] = bool(regime_residual_active)
    root_diagnostics["gauged_regime_residual"] = serializable_summary(regime_residual_star)
    root_diagnostics["regime_feature_bank"] = (
        regime_residual_star.get("feature_bank")
        if isinstance(regime_residual_star, Mapping)
        else None
    )
    root_diagnostics["regime_feature_semantics"] = (
        regime_residual_star.get("feature_semantics")
        if isinstance(regime_residual_star, Mapping)
        else None
    )
    root_diagnostics["selected_regime_features"] = (
        list(regime_residual_star.get("selected_features", ()))
        if isinstance(regime_residual_star, Mapping)
        else []
    )
    root_diagnostics["selected_v1_base_features"] = (
        list(regime_residual_star.get("selected_v1_base_features", ()))
        if isinstance(regime_residual_star, Mapping)
        else []
    )
    root_diagnostics["selected_v2_features"] = (
        list(regime_residual_star.get("selected_v2_features", ()))
        if isinstance(regime_residual_star, Mapping)
        else []
    )
    root_diagnostics["regime_residual_selector"] = (
        regime_residual_star.get("selector")
        if isinstance(regime_residual_star, Mapping)
        else None
    )
    root_diagnostics["regime_residual_anchor"] = (
        regime_residual_star.get("anchor")
        if isinstance(regime_residual_star, Mapping)
        else None
    )
    root_diagnostics["regime_feature_scales"] = (
        dict(regime_residual_star.get("feature_scales", {}))
        if isinstance(regime_residual_star, Mapping)
        else {}
    )
    root_diagnostics["regime_feature_root_values"] = (
        dict(regime_residual_star.get("feature_root_values", {}))
        if isinstance(regime_residual_star, Mapping)
        else {}
    )
    regime_diag = (
        dict(regime_residual_star.get("diagnostics", {}))
        if isinstance(regime_residual_star, Mapping)
        else {}
    )
    root_diagnostics["regime_signature_by_root_action"] = regime_diag.get("signature_by_root_action", {})
    root_diagnostics["regime_signature_residual_norms"] = regime_diag.get("candidate_residual_norms", {})
    root_diagnostics["regime_signature_incremental_norms"] = regime_diag.get("candidate_incremental_norms", {})
    root_diagnostics["regime_weighted_signature_diagnostics"] = regime_diag
    root_diagnostics["legacy_signature_rank"] = regime_diag.get("legacy_signature_rank")
    root_diagnostics["selected_signature_rank"] = regime_diag.get("selected_signature_rank")
    root_diagnostics["regime_coef_l1"] = float(regime_coef_l1)
    root_diagnostics["regime_coef_l2"] = float(regime_coef_l2)
    root_diagnostics["regime_coef_nonzero"] = int(regime_coef_nonzero)
    root_diagnostics["truncated_tuple_cuts_suppressed"] = rowgen_telemetry[
        "rowgen_oracle_only_truncated_tuple_cuts_suppressed"
    ]
    oracle_payload_coverage_count, oracle_payload_missing_count = _oracle_payload_coverage_counts(oracle_adp_star)
    root_diagnostics["oracle_plumbing_mode"] = oracle_plumbing_mode if oracle_adp_active else None
    root_diagnostics["oracle_payload_coverage_count"] = int(oracle_payload_coverage_count)
    root_diagnostics["oracle_payload_missing_count"] = int(oracle_payload_missing_count)
    root_diagnostics["oracle_active_in_lp"] = bool(oracle_adp_active)
    root_diagnostics["oracle_active_in_reconstruction"] = bool(oracle_adp_active)
    root_diagnostics["gauge_constraints_added"] = list(oracle_gauge_constraint_names)
    root_diagnostics["gauge_constraint_count"] = int(len(oracle_gauge_constraint_names))
    root_diagnostics["bellman_signature_diagnostic"] = oracle_bellman_signature_diagnostic
    if isinstance(oracle_adp_star, Mapping):
        try:
            root_diagnostics["oracle_root_term"] = float(oracle_adp_term_value(s0, oracle_adp_star))
        except Exception:
            root_diagnostics["oracle_root_term"] = None
        stage1_terms = []
        for state in belief:
            if len(_evidence(state)) != 1:
                continue
            try:
                stage1_terms.append(float(oracle_adp_term_value(state, oracle_adp_star)))
            except Exception:
                continue
        root_diagnostics["oracle_stage1_term_mean"] = (
            float(sum(stage1_terms) / len(stage1_terms)) if stage1_terms else None
        )
        root_diagnostics["oracle_stage1_state_count"] = int(len(stage1_terms))
    else:
        root_diagnostics["oracle_root_term"] = None
        root_diagnostics["oracle_stage1_term_mean"] = None
        root_diagnostics["oracle_stage1_state_count"] = 0
    root_diagnostics["legacy_residual_root_term"] = (
        float(root_phi_lp - root_diagnostics["oracle_root_term"])
        if root_phi_lp is not None and root_diagnostics.get("oracle_root_term") is not None
        else None
    )
    if isinstance(regime_residual_star, Mapping):
        try:
            root_diagnostics["regime_residual_root_term"] = float(
                regime_residual_term_value(
                    s0,
                    regime_residual_star,
                    belief=belief,
                    individuals=I,
                    pedigree=pedigree,
                    genes=gene_list if multi_gene else None,
                )
            )
        except Exception:
            root_diagnostics["regime_residual_root_term"] = None
    else:
        root_diagnostics["regime_residual_root_term"] = None

    if abcd16_direct_active:
        direct_feature_count = int(len(abcd16_direct_features))
        materialized_phi_state_count = int(len(Phi))
        root_diagnostics["abcd16_direct"] = {
            "enabled": True,
            "feature_bank_name": "ABCD16_DIRECT",
            "selection": abcd16_direct_selection,
            "selected_feature_names": list(abcd16_direct_features),
            "myopic_feature_names": list(abcd16_direct_myopic_features),
            "regime_feature_names": list(abcd16_direct_regime_features),
            "materialized_phi_state_count": materialized_phi_state_count,
            "feature_slot_count": int(direct_feature_count * materialized_phi_state_count),
            "feature_coefficient_variable_count": int(len(myopic_adp_vars) + len(regime_residual_vars)),
            "myopic_precompute_done": bool(myopic_eval is not None),
            "adp_seed_presolve_done": False,
            "active_row_export_used": False,
            "feature_selection_from_seed_used": False,
            "seed_duals_used": False,
            "seed_phi_values_used": False,
            "bellman_active_features_enter_phi": bool(myopic_adp_vars and regime_residual_vars),
            "features_enter_before_bellman_rows": True,
            "postprocess_only_feature_count": 0,
            "phi_reconstruction_includes_direct16_terms": True,
            "zero_extra_coefficients_allowed": True,
            "extra_coefficients_forced_nonzero": False,
            "coefficient_bounds_match_selected": True,
            "zero_column_feature_names": sorted(
                set(myopic_adp_zero_column_features)
                | set(
                    regime_residual_diagnostics.get("skipped_low_scale_features", [])
                    if isinstance(regime_residual_diagnostics, Mapping)
                    else []
                )
            ),
            "zero_column_coefficients_fixed_to_zero": True,
            "gurobi_variable_count": int(master.NumVars),
            "gurobi_constraint_count": int(master.NumConstrs),
        }

    def _root_binding_from_tight_constraints(entries):
        best = None
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name.startswith("init_test_"):
                continue
            if "_top" in name:
                continue
            slack = entry.get("slack")
            if not isinstance(slack, (int, float)):
                continue
            action = name[len("init_test_") :]
            candidate = (abs(float(slack)), action, float(slack))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None, None
        return best[1], best[2]

    binding_action, binding_margin = _root_binding_from_tight_constraints(tight_root_constraints)
    root_diagnostics["candidate_root_binding_action"] = binding_action
    root_diagnostics["candidate_root_action_margin"] = binding_margin

    oracle_summary = root_diagnostics.get("oracle_adp")
    if isinstance(oracle_summary, dict):
        oracle_summary.update(
            {
                "coefficient_l1": float(oracle_coef_l1),
                "coefficient_l2": float(oracle_coef_l2),
                "coefficient_nonzero": int(oracle_coef_nonzero),
                "oracle_root_term": root_diagnostics.get("oracle_root_term"),
                "oracle_stage1_term_mean": root_diagnostics.get("oracle_stage1_term_mean"),
                "oracle_stage1_state_count": root_diagnostics.get("oracle_stage1_state_count"),
                "oracle_plumbing_mode": oracle_plumbing_mode,
                "exact_value_coverage_count": int(oracle_payload_coverage_count),
                "exact_value_missing_state_count": int(oracle_payload_missing_count),
                "oracle_payload_coverage_count": int(oracle_payload_coverage_count),
                "oracle_payload_missing_count": int(oracle_payload_missing_count),
                "active_in_lp_construction": bool(oracle_adp_active),
                "active_in_phi_reconstruction": bool(oracle_adp_active),
                "active_in_lp": bool(oracle_adp_active),
                "active_in_reconstruction": bool(oracle_adp_active),
                "legacy_residual_enabled": bool(oracle_adp_active and not oracle_only_fixed_phi_active),
                "legacy_residual_root_term": root_diagnostics.get("legacy_residual_root_term"),
                "gauge_constraints_added": list(oracle_gauge_constraint_names),
                "gauge_constraint_count": int(len(oracle_gauge_constraint_names)),
                "policy_source": "adp_policy",
            }
        )

    if edge_diagnostics_enable and edge_features_active and edge_star is not None:
        from .postprocess import phi_hat

        try:
            phi_root_with_edge = phi_hat(
                s0,
                theta_star=theta_star,
                W_star=W_sol,
                belief=belief,
                gen_states=gen_states,
                individuals=I,
                theta_mode=theta_mode,
                tuple_pmfs=tuple_posteriors if tuple_mode else None,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=edge_star,
                W_trio_star=trio_star,
                pedigree_edges=pedigree_edges if edge_features_active else None,
                pedigree_trios=pedigree_trios if trio_features_active else None,
                infer=infer,
                genes=gene_list if multi_gene else None,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
            phi_root_without_edge = phi_hat(
                s0,
                theta_star=theta_star,
                W_star=W_sol,
                belief=belief,
                gen_states=gen_states,
                individuals=I,
                theta_mode=theta_mode,
                tuple_pmfs=tuple_posteriors if tuple_mode else None,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=None,
                W_trio_star=trio_star,
                pedigree_edges=pedigree_edges if edge_features_active else None,
                pedigree_trios=pedigree_trios if trio_features_active else None,
                infer=infer,
                genes=gene_list if multi_gene else None,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
            root_diagnostics["edge_root_term"] = float(phi_root_with_edge - phi_root_without_edge)
        except Exception:
            root_diagnostics["edge_root_term"] = None

        stage1_terms = []
        for state in belief:
            if len(_evidence(state)) != 1:
                continue
            try:
                phi_with_edge = phi_hat(
                    state,
                    theta_star=theta_star,
                    W_star=W_sol,
                    belief=belief,
                    gen_states=gen_states,
                    individuals=I,
                    theta_mode=theta_mode,
                    tuple_pmfs=tuple_posteriors if tuple_mode else None,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=edge_star,
                    W_trio_star=trio_star,
                    pedigree_edges=pedigree_edges if edge_features_active else None,
                    pedigree_trios=pedigree_trios if trio_features_active else None,
                    infer=infer,
                    genes=gene_list if multi_gene else None,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
                phi_without_edge = phi_hat(
                    state,
                    theta_star=theta_star,
                    W_star=W_sol,
                    belief=belief,
                    gen_states=gen_states,
                    individuals=I,
                    theta_mode=theta_mode,
                    tuple_pmfs=tuple_posteriors if tuple_mode else None,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=None,
                    W_trio_star=trio_star,
                    pedigree_edges=pedigree_edges if edge_features_active else None,
                    pedigree_trios=pedigree_trios if trio_features_active else None,
                    infer=infer,
                    genes=gene_list if multi_gene else None,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
                stage1_terms.append(phi_with_edge - phi_without_edge)
            except Exception:
                continue
        root_diagnostics["edge_stage1_term_mean"] = (
            float(sum(stage1_terms) / len(stage1_terms)) if stage1_terms else None
        )
        root_diagnostics["edge_stage1_state_count"] = int(len(stage1_terms))

    if trio_diagnostics_enable and trio_features_active and trio_star is not None:
        from .postprocess import phi_hat

        try:
            phi_root_with_trio = phi_hat(
                s0,
                theta_star=theta_star,
                W_star=W_sol,
                belief=belief,
                gen_states=gen_states,
                individuals=I,
                theta_mode=theta_mode,
                tuple_pmfs=tuple_posteriors if tuple_mode else None,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=edge_star,
                W_trio_star=trio_star,
                pedigree_edges=pedigree_edges if edge_features_active else None,
                pedigree_trios=pedigree_trios if trio_features_active else None,
                infer=infer,
                genes=gene_list if multi_gene else None,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
            phi_root_without_trio = phi_hat(
                s0,
                theta_star=theta_star,
                W_star=W_sol,
                belief=belief,
                gen_states=gen_states,
                individuals=I,
                theta_mode=theta_mode,
                tuple_pmfs=tuple_posteriors if tuple_mode else None,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=edge_star,
                W_trio_star=None,
                pedigree_edges=pedigree_edges if edge_features_active else None,
                pedigree_trios=pedigree_trios if trio_features_active else None,
                infer=infer,
                genes=gene_list if multi_gene else None,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
            )
            root_diagnostics["trio_root_term"] = float(phi_root_with_trio - phi_root_without_trio)
        except Exception:
            root_diagnostics["trio_root_term"] = None

        trio_stage1_terms = []
        for state in belief:
            if len(_evidence(state)) != 1:
                continue
            try:
                phi_with_trio = phi_hat(
                    state,
                    theta_star=theta_star,
                    W_star=W_sol,
                    belief=belief,
                    gen_states=gen_states,
                    individuals=I,
                    theta_mode=theta_mode,
                    tuple_pmfs=tuple_posteriors if tuple_mode else None,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=edge_star,
                    W_trio_star=trio_star,
                    pedigree_edges=pedigree_edges if edge_features_active else None,
                    pedigree_trios=pedigree_trios if trio_features_active else None,
                    infer=infer,
                    genes=gene_list if multi_gene else None,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
                phi_without_trio = phi_hat(
                    state,
                    theta_star=theta_star,
                    W_star=W_sol,
                    belief=belief,
                    gen_states=gen_states,
                    individuals=I,
                    theta_mode=theta_mode,
                    tuple_pmfs=tuple_posteriors if tuple_mode else None,
                    tuple_mode=tuple_mode,
                    aaub_star=aaub_star,
                    W_edge_star=edge_star,
                    W_trio_star=None,
                    pedigree_edges=pedigree_edges if edge_features_active else None,
                    pedigree_trios=pedigree_trios if trio_features_active else None,
                    infer=infer,
                    genes=gene_list if multi_gene else None,
                    myopic_adp_star=myopic_adp_star,
                    oracle_adp_star=oracle_adp_star,
                    regime_residual_star=regime_residual_star,
                )
                trio_stage1_terms.append(phi_with_trio - phi_without_trio)
            except Exception:
                continue
        root_diagnostics["trio_stage1_term_mean"] = (
            float(sum(trio_stage1_terms) / len(trio_stage1_terms)) if trio_stage1_terms else None
        )
        root_diagnostics["trio_stage1_state_count"] = int(len(trio_stage1_terms))

    candidate_post_polish = _build_candidate_snapshot(
        candidate_id="post_polish",
        phi_star=Phi_sol,
        w_star=W_sol,
        w_gene_star=W_gene_sol,
        theta_star_value=theta_star,
        aaub_value=aaub_star,
        edge_value=edge_star,
        trio_value=trio_star,
        myopic_adp_value=myopic_adp_star,
        oracle_adp_value=oracle_adp_star,
        regime_residual_value=regime_residual_star_current,
    )
    adp_inference_time = inf_cache.total_inference_time

    # Optional safety: recompute root Φ from θ/W to catch misaligned Phi_sol
    phi_eval = {}
    if return_phi_eval:
        from .postprocess import phi_hat
        try:
            root_state = s0
            phi_eval[root_state] = phi_hat(
                root_state,
                theta_star=theta_star,
                W_star=W_sol,
                belief=belief,
                gen_states=gen_states,
                individuals=I,
                theta_mode=theta_mode,
                pedigree=pedigree,
                tuple_pmfs=tuple_posteriors if tuple_mode else None,
                tuple_mode=tuple_mode,
                aaub_star=aaub_star,
                W_edge_star=edge_star,
                W_trio_star=trio_star,
                pedigree_edges=pedigree_edges if edge_features_active else None,
                pedigree_trios=pedigree_trios if trio_features_active else None,
                infer=infer,
                genes=gene_list if multi_gene else None,
                myopic_adp_star=myopic_adp_star,
                oracle_adp_star=oracle_adp_star,
                regime_residual_star=regime_residual_star,
                theta_model=theta_model_info.get("theta_model"),
                theta_model_spec=theta_model_info.get("theta_model_spec"),
            )
        except Exception:
            phi_eval = {}
    root_phi_reconstructed = phi_eval.get(s0) if isinstance(phi_eval, Mapping) else None
    root_diagnostics["phi_root_reconstructed"] = (
        float(root_phi_reconstructed)
        if root_phi_reconstructed is not None and math.isfinite(float(root_phi_reconstructed))
        else None
    )
    root_diagnostics["phi_root_lp"] = root_phi_lp
    if (
        root_diagnostics.get("phi_root_reconstructed") is not None
        and root_phi_lp is not None
    ):
        root_diagnostics["phi_root_lp_reconstruction_diff"] = float(
            root_phi_lp - root_diagnostics["phi_root_reconstructed"]
        )
    else:
        root_diagnostics["phi_root_lp_reconstruction_diff"] = None

    if return_stats:
        stats = {
            "num_vars": master.NumVars,
            "num_constrs": master.NumConstrs,
            "phi_vars": len(Phi),                 # identity-space count
            "elapsed_iters": it,
            "role_groups": {k: len(v) for k,v in (role_groups or {}).items()},
            "adp_inference_time": adp_inference_time,
            "adp_inference_calls": inf_cache.inference_calls,
        }
        if use_bellman_rowgen:
            stats.update({
                "cache_hits": bellman_gen.cache_hits,
                "cache_misses": bellman_gen.cache_misses,
                "bn_time": bellman_gen.bn_time,
            })
        stats["tail_regularization"] = tail_regularization
        stats["slack_refactor"] = slack_refactor
        stats["aaub"] = aaub_star
        stats["edge_star"] = edge_star
        stats["trio_star"] = trio_star
        stats["myopic_adp_star"] = myopic_adp_star
        stats["myopic_adp"] = serializable_summary(myopic_adp_star)
        stats["oracle_adp_star"] = oracle_adp_star
        stats["oracle_adp"] = oracle_serializable_summary(oracle_adp_star)
        stats["regime_residual_star"] = regime_residual_star
        stats["gauged_regime_residual"] = serializable_summary(regime_residual_star)
        stats["root_diagnostics"] = root_diagnostics
        stats["rowgen_telemetry"] = rowgen_telemetry
        stats["runtime_telemetry"] = runtime_telemetry
        stats["candidate_pre_polish"] = {k: v for k, v in candidate_pre_polish.items() if k != "payload"}
        stats["candidate_post_polish"] = {k: v for k, v in candidate_post_polish.items() if k != "payload"}
        stats["W_gene_star"] = W_gene_sol
        stats["seed_probe"] = seed_probe
        stats["rowgen_first_pass_probe"] = None
        stats["theta_model"] = theta_model_info.get("theta_model")
        stats["theta_model_spec_path"] = theta_model_info.get("theta_model_spec_path")
        stats["theta_model_signature"] = theta_model_info.get("theta_model_signature")
        stats["theta_model_spec"] = theta_model_info.get("theta_model_spec")
        payload = (Phi_sol, W_sol, belief, theta_star, master, adp_inference_time, inf_cache)
        if return_phi_eval:
            payload = payload + (phi_eval,)
        return payload, stats
    inf_cache.tuple_posteriors = tuple_posteriors if tuple_mode else {}
    inf_cache.tail_regularization = tail_regularization
    inf_cache.slack_refactor = slack_refactor
    inf_cache.aaub_star = aaub_star
    inf_cache.edge_star = edge_star
    inf_cache.trio_star = trio_star
    inf_cache.myopic_adp_star = myopic_adp_star
    inf_cache.oracle_adp_star = oracle_adp_star
    inf_cache.regime_residual_star = regime_residual_star
    inf_cache.pedigree_edges = pedigree_edges if edge_features_active else None
    inf_cache.pedigree_trios = pedigree_trios if trio_features_active else None
    inf_cache.root_diagnostics = root_diagnostics
    inf_cache.rowgen_telemetry = rowgen_telemetry
    inf_cache.runtime_telemetry = runtime_telemetry
    inf_cache.theta_model_metadata = theta_model_info
    inf_cache.candidate_pre_polish = {k: v for k, v in candidate_pre_polish.items() if k != "payload"}
    inf_cache.candidate_post_polish = {k: v for k, v in candidate_post_polish.items() if k != "payload"}
    inf_cache.candidate_eval_payloads = {
        "pre_polish_payload": candidate_pre_polish.get("payload"),
        "post_polish_payload": candidate_post_polish.get("payload"),
    }
    inf_cache.w_gene_star = W_gene_sol

    payload = (Phi_sol, W_sol, belief, theta_star, master, adp_inference_time, inf_cache)
    if return_phi_eval:
        payload = payload + (phi_eval,)
    return payload


class DualDPResult:
    """Result class for dual DP solver with convenient access to root value."""
    def __init__(self, Phi_sol, W_sol, belief, theta, master, phi_eval=None):
        self.Phi_sol = Phi_sol
        self.W_sol = W_sol
        self.belief = belief
        self.theta = theta
        self.master = master
        # Extract root value for convenience
        def _state_size(state):
            return len(state)

        root_state = min(Phi_sol.keys(), key=_state_size) if Phi_sol else frozenset()
        self.root_state = root_state
        self.root_value_phi = None
        if phi_eval and root_state in phi_eval:
            self.root_value_phi = phi_eval[root_state]
        else:
            self.root_value_phi = Phi_sol.get(root_state, None)

def solve_dual_dp(pedigree, config, verbose=False, debug_lp_path=None, role_groups=None):
    """
    Wrapper function for solve_dual_dp_with_domain that matches the interface expected by debug scripts.
    """
    from ..models.genetics_cpd import make_founder_genotype_cpd, make_inheritance_genotype_cpd_with_table
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    import numpy as np

    # Extract individuals and setup genotype states
    individuals = pedigree.to_list()
    gen_states = [0, 1, 2]
    
    # Create x dictionary (testability indicator)
    x = {i: 1 for i in individuals}
    
    # Setup initial belief state
    mu0 = {frozenset(): 1.0}
    
    # Create Bayesian network using the same approach as core.py
    bn_edges = []
    all_cpds = []
    initial_p = {}
    child_cpds = {}
    
    # Founder prior helper
    def founder_prior(p): 
        return [(1-p)**2, 2*p*(1-p), p**2]
    
    # Use topological sort to ensure parents are processed before children
    for individual in individuals:
        parents = pedigree.get_parents(individual)
        if not parents:
            # Founder
            cpd = make_founder_genotype_cpd(individual, allele_freq=config.allele_freq)
            all_cpds.append(cpd)
            initial_p[individual] = {g: founder_prior(config.allele_freq)[g] for g in gen_states}
        else:
            # Child
            if len(parents) != 2:
                raise ValueError(f"Person {individual} should have exactly 2 parents, but has {len(parents)}: {parents}")
            parent1, parent2 = parents
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
    
    # Create Bayesian network and inference engine
    bn = DiscreteBayesianNetwork(bn_edges)
    bn.add_cpds(*all_cpds)
    assert bn.check_model(), "Bayesian network is invalid!"
    infer = VariableElimination(bn)
    
    if verbose:
        print(f"[DEBUG] Initial marginal probabilities:")
        for individual in individuals:
            p_values = initial_p[individual]
            p12 = p_values[1] + p_values[2]
            print(f"  {individual}: P(0)={p_values[0]:.6f}, P(1)={p_values[1]:.6f}, P(2)={p_values[2]:.6f}, P(1+2)={p12:.6f}")
            parents = pedigree.get_parents(individual)
            if parents:
                print(f"    Parents: {parents}")
            else:
                print(f"    Founder")
    
    # Initial z (one-hot indicators - all 0 since no one is tested initially)
    initial_z = {i: {g: 0.0 for g in gen_states} for i in individuals}
    
    # Call the main solver
    payload = solve_dual_dp_with_domain(
        I=individuals,
        gen_states=gen_states,
        mu0=mu0,
        a=config.a,
        b=config.b,
        c=config.c,
        delta=config.delta,
        x=x,
        allele_freq=config.allele_freq,
        child_cpds=child_cpds,
        pedigree=pedigree,
        p0=initial_p,
        z0=initial_z,
        infer=infer,
        role_groups=role_groups,
        verbose=verbose,
        debug_lp_path=debug_lp_path,
        fixed_cost=config.fixed_cost,
        variable_cost=config.variable_cost
    )
    
    Phi_sol, W_sol, belief, theta, master = payload[:5]
    phi_eval = payload[5] if len(payload) > 5 else None
    return DualDPResult(Phi_sol, W_sol, belief, theta, master, phi_eval)
