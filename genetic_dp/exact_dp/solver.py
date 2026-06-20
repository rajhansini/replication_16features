import os
import time
from pathlib import Path
from typing import Any, Dict
from ..models.belief import InferenceResult
from ..models.outcomes import project_state_by_gene, project_successor_by_gene
from .utils import (
    partial_states,
    build_full_joint,
    build_belief_map,
    lift_tuple_posteriors_to_genes,
    GENOTYPE_STATES,
)
from ..models.reward import r_reward, r_reward_test

try:  # PuLP is optional; fall back to primal DP when unavailable.
    import pulp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pulp = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _build_exact_dual_solver():
    backend = os.getenv("EXACT_DUAL_LP_SOLVER", "gurobi").strip().lower()
    log_enabled = _env_flag("EXACT_DUAL_LOG", True)
    log_path_raw = os.getenv("EXACT_DUAL_LOG_PATH", "").strip()
    time_limit_raw = os.getenv("EXACT_DUAL_TIME_LIMIT_SEC", "").strip()
    gap_rel_raw = os.getenv("EXACT_DUAL_REL_GAP", "").strip()

    log_path = None
    if log_path_raw:
        log_path = Path(log_path_raw).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    time_limit = float(time_limit_raw) if time_limit_raw else None
    gap_rel = float(gap_rel_raw) if gap_rel_raw else None

    if backend == "gurobi":
        if not hasattr(pulp, "GUROBI"):
            raise RuntimeError("PuLP GUROBI backend is unavailable in this environment.")
        solver = pulp.GUROBI(
            msg=log_enabled,
            timeLimit=time_limit,
            gapRel=gap_rel,
            logPath=str(log_path) if log_path is not None else None,
            Seed=int(os.getenv("GUROBI_SEED", "0")),
            Method=_env_int("EXACT_DUAL_GUROBI_METHOD", 2),
            Crossover=_env_int("EXACT_DUAL_GUROBI_CROSSOVER", 0),
            NumericFocus=_env_int("EXACT_DUAL_GUROBI_NUMERIC_FOCUS", 2),
            DualReductions=_env_int("EXACT_DUAL_GUROBI_DUAL_REDUCTIONS", 0),
            BarHomogeneous=_env_int("EXACT_DUAL_GUROBI_BAR_HOMOGENEOUS", 1),
        )
    elif backend == "cbc":
        solver = pulp.PULP_CBC_CMD(
            msg=log_enabled,
            timeLimit=time_limit,
            gapRel=gap_rel,
            logPath=str(log_path) if log_path is not None else None,
        )
    else:
        raise ValueError(
            f"Unknown EXACT_DUAL_LP_SOLVER={backend!r}; expected 'gurobi' or 'cbc'."
        )
    return solver, backend, str(log_path) if log_path is not None else None

# --- Exact dual DP via PuLP ---
def solve_exact_dual_pulp(
    individuals,
    gen_states,
    mu0,
    belief,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    *,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    base_gen_states=GENOTYPE_STATES,
    progress_callback=None,
    return_metadata: bool = False,
):
    total_started = time.perf_counter()

    def _emit_progress(event: str, status: str = "running", **payload: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "phase": "exact_dual_solve",
                "event": event,
                "status": status,
                "elapsed_sec": float(time.perf_counter() - total_started),
                **payload,
            }
        )

    if pulp is None:
        Phi_star, _ = solve_exact_dp_primal(
            individuals,
            gen_states,
            mu0,
            belief,
            a,
            b,
            c,
            delta,
            fixed_cost,
            variable_cost,
            genes=genes,
            a_gene=a_gene,
            b_gene=b_gene,
            c_gene=c_gene,
            delta_gene=delta_gene,
            base_gen_states=base_gen_states,
        )
        metadata = {
            "progress_status": "completed_primal_fallback",
            "backend": "primal_fallback",
            "state_count": len(belief),
            "lp_variable_count": None,
            "lp_constraint_count": None,
            "lp_build_elapsed_sec": 0.0,
            "lp_solve_elapsed_sec": float(time.perf_counter() - total_started),
            "total_elapsed_sec": float(time.perf_counter() - total_started),
            "status": "Optimal",
            "log_path": None,
        }
        return (Phi_star, metadata) if return_metadata else Phi_star
    # enumerate states
    states = list(belief.keys())
    gene_list = tuple(genes) if genes else tuple()
    per_gene_phi_active = bool(gene_list)

    # build LP
    lp_build_started = time.perf_counter()
    _emit_progress(
        "lp_build_start",
        state_count=len(states),
        per_gene_phi_active=per_gene_phi_active,
        gene_count=len(gene_list),
    )
    prob = pulp.LpProblem("D0_Single", pulp.LpMinimize)
    Phi = {s: pulp.LpVariable(f"Phi_{'_'.join(i+str(g) for i,g in sorted(s)) or 'root'}",
                              lowBound=None)
           for s in states}
    Phi_gene = {gene: {} for gene in gene_list} if per_gene_phi_active else {}
    state_projection_cache = {}
    successor_projection_cache = {}

    def _projection_label(gene, proj_state):
        parts = [f"{person}_{genotype}" for person, genotype in sorted(proj_state)]
        return f"{gene}_" + ("_".join(parts) if parts else "root")

    def _get_state_projection(state):
        if state in state_projection_cache:
            return state_projection_cache[state]
        proj = project_state_by_gene(state, gene_list)
        state_projection_cache[state] = proj
        return proj

    def _get_successor_projection(state, person, outcome):
        key = (state, person, outcome)
        if key in successor_projection_cache:
            return successor_projection_cache[key]
        proj = project_successor_by_gene(state, person, outcome, gene_list)
        successor_projection_cache[key] = proj
        return proj

    def _get_phi_gene(gene, proj_state):
        phi_map = Phi_gene[gene]
        if proj_state in phi_map:
            return phi_map[proj_state]
        label = _projection_label(gene, proj_state)
        var = pulp.LpVariable(f"Phi_{label}", lowBound=None)
        phi_map[proj_state] = var
        return var

    def _phi_sum(state):
        if not per_gene_phi_active:
            return Phi[state]
        proj_map = _get_state_projection(state)
        return pulp.lpSum(_get_phi_gene(gene, proj_map.get(gene, frozenset())) for gene in gene_list)

    # objective
    prob += pulp.lpSum(mu0.get(s, 0.0) * _phi_sum(s) for s in states)

    # constraints
    for s in states:
        belief_entry = belief[s]
        tuple_posteriors_state: Dict[str, Dict[Tuple[int, ...], float]] = {}
        if isinstance(belief_entry, InferenceResult):
            p_s = belief_entry.marginals
            tuple_posteriors_state = belief_entry.get_tuple_pmfs()
            per_gene_probs = (
                belief_entry.get_per_gene_probs() if genes else None
            )
        else:
            p_s = belief_entry
            per_gene_probs = None

        tested = {i for i,_ in s}

        if genes and not per_gene_probs:
            if tuple_posteriors_state:
                per_gene_probs = lift_tuple_posteriors_to_genes(tuple_posteriors_state, genes, base_gen_states)
            else:
                per_gene_probs = lift_tuple_posteriors_to_genes(p_s, genes, base_gen_states)

        # stopping
        Rstop = sum(
            r_reward(
                k,
                p_s,
                a,
                b,
                c,
                delta,
                per_gene_probs=per_gene_probs,
                a_gene=a_gene,
                b_gene=b_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )
            for k in individuals if k not in tested
        )
        prob += _phi_sum(s) >= Rstop, f"stop_{'_'.join(i+str(g) for i,g in sorted(s)) or 'root'}"

        # testing
        for i in individuals:
            if i in tested:
                continue
            rsi = r_reward_test(
                i,
                p_s,
                a,
                b,
                c,
                delta,
                fixed_cost,
                variable_cost,
                per_gene_probs=per_gene_probs,
                a_gene=a_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )
            # successor expectation
            succ_terms = []
            if tuple_posteriors_state and i in tuple_posteriors_state:
                for outcome, prob_g in tuple_posteriors_state.get(i, {}).items():
                    if prob_g <= 0.0:
                        continue
                    succ = frozenset(s | {(i, outcome)})
                    if per_gene_phi_active:
                        succ_proj = _get_successor_projection(s, i, outcome)
                        phi_sum = pulp.lpSum(
                            _get_phi_gene(gene, succ_proj.get(gene, frozenset()))
                            for gene in gene_list
                        )
                        succ_terms.append(prob_g * phi_sum)
                    else:
                        succ_terms.append(prob_g * Phi[succ])
            else:
                for g, prob_g in p_s[i].items():
                    if prob_g <= 0.0:
                        continue
                    succ = frozenset(s | {(i, g)})
                    if per_gene_phi_active:
                        succ_proj = _get_successor_projection(s, i, g)
                        phi_sum = pulp.lpSum(
                            _get_phi_gene(gene, succ_proj.get(gene, frozenset()))
                            for gene in gene_list
                        )
                        succ_terms.append(prob_g * phi_sum)
                    else:
                        succ_terms.append(prob_g * Phi[succ])
            if succ_terms:
                prob += _phi_sum(s) >= rsi + pulp.lpSum(succ_terms), \
                        f"test_{i}_{'_'.join(i+str(g) for i,g in sorted(s)) or 'root'}"
    lp_build_elapsed = time.perf_counter() - lp_build_started
    lp_variable_count = prob.numVariables()
    lp_constraint_count = len(prob.constraints)
    _emit_progress(
        "lp_build_complete",
        "completed",
        state_count=len(states),
        lp_variable_count=lp_variable_count,
        lp_constraint_count=lp_constraint_count,
        lp_build_elapsed_sec=float(lp_build_elapsed),
    )

    # solve
    solver, backend, log_path = _build_exact_dual_solver()
    print(f"Exact dual LP backend: {backend}")
    if log_path:
        print(f"Exact dual LP log: {log_path}")
    lp_solve_started = time.perf_counter()
    _emit_progress(
        "lp_solve_start",
        backend=backend,
        log_path=log_path,
        lp_variable_count=lp_variable_count,
        lp_constraint_count=lp_constraint_count,
    )
    try:
        prob.solve(solver)
    except Exception as exc:
        _emit_progress(
            "lp_solve_complete",
            "error",
            backend=backend,
            log_path=log_path,
            lp_solve_elapsed_sec=float(time.perf_counter() - lp_solve_started),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    lp_solve_elapsed = time.perf_counter() - lp_solve_started
    status = pulp.LpStatus.get(prob.status, str(prob.status))
    metadata = {
        "progress_status": "completed" if status == "Optimal" else "failed",
        "backend": backend,
        "state_count": len(states),
        "lp_variable_count": int(lp_variable_count),
        "lp_constraint_count": int(lp_constraint_count),
        "lp_build_elapsed_sec": float(lp_build_elapsed),
        "lp_solve_elapsed_sec": float(lp_solve_elapsed),
        "total_elapsed_sec": float(time.perf_counter() - total_started),
        "status": status,
        "log_path": log_path,
        "per_gene_phi_active": per_gene_phi_active,
        "gene_count": len(gene_list),
    }
    _emit_progress(
        "lp_solve_complete",
        metadata["progress_status"],
        backend=backend,
        log_path=log_path,
        lp_solve_elapsed_sec=float(lp_solve_elapsed),
        lp_status=status,
    )
    if status != "Optimal":
        raise RuntimeError(f"Exact dual LP did not solve to optimality (status={status}).")

    phi_sum_solution = {s: _phi_sum(s).value() if per_gene_phi_active else Phi[s].value() for s in states}
    if not per_gene_phi_active:
        return (phi_sum_solution, metadata) if return_metadata else phi_sum_solution

    phi_gene_solution = {
        gene: {proj: var.value() for proj, var in proj_map.items()}
        for gene, proj_map in Phi_gene.items()
    }
    result = (phi_sum_solution, phi_gene_solution)
    return (result, metadata) if return_metadata else result


def solve_exact_dp_primal(
    individuals,
    gen_states,
    mu0,
    belief,
    a,
    b,
    c,
    delta,
    fixed_cost,
    variable_cost,
    *,
    genes=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    base_gen_states=GENOTYPE_STATES,
):
    """
    Exact DP via backward induction on the belief map (acyclic testing graph).
    Returns (value_map, policy_map) over the provided belief states.
    """
    gene_list = tuple(genes) if genes else tuple()
    multi_gene = bool(gene_list)

    def _evidence_state(state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], frozenset):
            return state[0]
        return state

    def _state_size(state):
        return len(_evidence_state(state))

    # Sort states from most-tested to least-tested (so successors are ready).
    states = sorted(belief.keys(), key=_state_size, reverse=True)
    V = {}
    policy = {}

    for s in states:
        entry = belief[s]
        if isinstance(entry, tuple) and len(entry) == 2:
            posterior_entry = entry[0]
        else:
            posterior_entry = entry

        if isinstance(posterior_entry, InferenceResult):
            p_s = posterior_entry.marginals
            tuple_posteriors_state = posterior_entry.get_tuple_pmfs()
            per_gene_probs = posterior_entry.get_per_gene_probs() if multi_gene else None
        else:
            p_s = posterior_entry
            tuple_posteriors_state = {}
            per_gene_probs = None

        if multi_gene and not per_gene_probs:
            if tuple_posteriors_state:
                per_gene_probs = lift_tuple_posteriors_to_genes(
                    tuple_posteriors_state,
                    gene_list,
                    base_gen_states,
                )
            else:
                per_gene_probs = lift_tuple_posteriors_to_genes(
                    p_s,
                    gene_list,
                    base_gen_states,
                )

        evidence = _evidence_state(s)
        tested = {i for i, _ in evidence}

        # Stop action
        r_stop = sum(
            r_reward(
                k,
                p_s,
                a,
                b,
                c,
                delta,
                per_gene_probs=per_gene_probs,
                a_gene=a_gene,
                b_gene=b_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )
            for k in individuals if k not in tested
        )
        best_val = r_stop
        best_action = ("stop", None, r_stop)

        # Test actions
        if len(tested) < len(individuals):
            for i in individuals:
                if i in tested:
                    continue
                r_i = r_reward_test(
                    i,
                    p_s,
                    a,
                    b,
                    c,
                    delta,
                    fixed_cost,
                    variable_cost,
                    per_gene_probs=per_gene_probs,
                    a_gene=a_gene,
                    c_gene=c_gene,
                    delta_gene=delta_gene,
                )
                exp_succ = 0.0
                if tuple_posteriors_state and i in tuple_posteriors_state:
                    for outcome, prob in tuple_posteriors_state.get(i, {}).items():
                        if prob <= 0.0:
                            continue
                        succ = frozenset(evidence | {(i, outcome)})
                        if len(succ) >= len(individuals):
                            continue
                        exp_succ += prob * V.get(succ, 0.0)
                else:
                    for g, prob in p_s[i].items():
                        if prob <= 0.0:
                            continue
                        succ = frozenset(evidence | {(i, g)})
                        if len(succ) >= len(individuals):
                            continue
                        exp_succ += prob * V.get(succ, 0.0)

                val = r_i + exp_succ
                if val > best_val:
                    best_val = val
                    best_action = ("test", i, val)

        V[s] = best_val
        policy[s] = best_action

    return V, policy
