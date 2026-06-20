import gurobipy as gp
from gurobipy import GRB
from ..models.reward import r_reward_testp
from .constraints import add_domain_constraints
from ..optimisation.utils import compute_phi, _project_outcome
from .consistency import ConsistencyManager
from ..models.belief import propagate_all_marginals
from typing import Dict, Tuple, List, FrozenSet
import logging

def build_row_generation_stop_with_domain(
        I, S, gen_states,
        p_current, z_current,   # belief at state S
        a, b, c, delta,            # reward parameters
        x, allele_freq, child_cpds, pedigree,
        Phi_current):
    """
    Row‑gen‑stop @ state S:
      max_{p,z ∈ Domain(S)} [ Σ_{k∉S} r(k; p_current) ] – Φ(s)
    Uses add_domain_constraints for A–F.
    """
    m = gp.Model("row-gen-stop-domain")
    m.Params.OutputFlag = 0

    # create decision vars
    p = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="p")
    z = m.addVars(I, gen_states, vtype=GRB.BINARY,    name="z")
    m.update()

    # build x_state so add_domain_constraints knows who is “tested”
    x_state = {i: 1 if i in S else x.get(i,0) for i in I}

    # Extract observed genotypes from the current belief state
    observed_genotypes = {}
    if z_current:
        for i in I:
            if i in S and i in z_current:
                for outcome_key, val in z_current[i].items():
                    if val > 0.5:
                        observed_genotypes[i] = _project_outcome(outcome_key)
                        break

    # enforce your A–F in one shot
    add_domain_constraints(
        m, p, z, I, gen_states,
        x_state, allele_freq, child_cpds, pedigree,
        tested_individual=None,
        observed_genotype=None,
        observed_genotypes=observed_genotypes
    )

    # objective = Σ_{k∉S} r(k ; p_current)  – Φ(s)
    expr = gp.LinExpr()
    for k in I:
        if k not in S:
            p12 = p_current[k][1] + p_current[k][2]
            expr += a[k]*(p12 - delta[k]*p12*p12) \
                  + b[k]*(p12 - p12*p12) + c[k]
    m.setObjective(expr - Phi_current, GRB.MAXIMIZE)

    return m


def build_row_generation_test_with_domain(
        I, S, gen_states, W_snapshot,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        Phi_current, fixed_cost, variable_cost):
    """
    Row‑gen‑test @ state S:
      For each candidate individual i not in S, build a separate subproblem:
      max r(i; p_current) + [Φ(s∪{i}; p,z,W_snapshot) − Φ(s)]
      subject to A–F on (p,z) with the candidate treated as tested (successor semantics).
      
      Returns the best violation across all candidates.
    """
    import os
    use_expected = os.getenv("EXPECTED_ROWGEN", "0") == "1"

    best_violation = -float('inf')
    best_model = None
    best_individual = None

    consistency_manager = ConsistencyManager(I, pedigree)
    
    # Try testing each individual not already in S
    candidates = [i for i in I if i not in S]
    
    for test_individual in candidates:
        # # COMBINATORIAL CONSISTENCY CHECK - COMMENTED OUT (might use later)
        # # First, check for genetic consistency
        # hypothetical_state = S.copy()
        # # We don't know the outcome, so we check for all possible genotypes
        # is_feasible = False
        # for g in gen_states:
        #     hypothetical_state[test_individual] = g
        #     if consistency_manager.check_consistency(hypothetical_state):
        #         is_feasible = True
        #         break
        
        # if not is_feasible:
        #     continue

        if not use_expected:
            # Standard successor subproblem (candidate treated as tested)
            m = gp.Model(f"row-gen-test-{test_individual}")
            m.Params.OutputFlag = 0

            # decision vars
            p = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="p")
            z = m.addVars(I, gen_states, vtype=GRB.BINARY, name="z")
            m.update()

            # candidate tested: p = z
            for g in gen_states:
                m.addConstr(p[test_individual, g] == z[test_individual, g],
                           name=f"test_{test_individual}_{g}")

            # A–F with candidate tested
            x_state = {i: 1 if (i in S or i == test_individual) else x.get(i,0) for i in I}

            observed_genotypes = {}
            if z_current:
                for i in I:
                    if i in S and i in z_current:
                        for outcome_key, val in z_current[i].items():
                            if val > 0.5:
                                observed_genotypes[i] = _project_outcome(outcome_key)
                                break

            try:
                add_domain_constraints(
                    m, p, z, I, gen_states,
                    x_state, allele_freq, child_cpds, pedigree,
                    tested_individual=None,
                    observed_genotype=None,
                    observed_genotypes=observed_genotypes
                )
            except Exception:
                continue

            p12 = p_current[test_individual][1] + p_current[test_individual][2]
            r_i = r_reward_testp(test_individual, p12, a, b, c, delta, fixed_cost, variable_cost)

            phi_next = compute_phi(S | {test_individual}, p, z, W_snapshot, gen_states)
            obj = r_i + phi_next - Phi_current
            m.setObjective(obj, GRB.MAXIMIZE)
            m.optimize()
            if m.Status == gp.GRB.OPTIMAL and m.ObjVal > best_violation:
                best_violation = m.ObjVal
                best_model = m
                best_individual = test_individual
        else:
            # Expected-successor variant: separate solves per outcome g
            p12 = p_current[test_individual][1] + p_current[test_individual][2]
            r_i = r_reward_testp(test_individual, p12, a, b, c, delta, fixed_cost, variable_cost)

            expected_phi = 0.0
            for gfix in gen_states:
                w_g = p_current[test_individual][gfix]
                if w_g <= 1e-8:
                    continue
                m_g = gp.Model(f"row-gen-test-{test_individual}-g{gfix}")
                m_g.Params.OutputFlag = 0
                p_g = m_g.addVars(I, gen_states, lb=0.0, ub=1.0, name="p")
                z_g = m_g.addVars(I, gen_states, vtype=GRB.BINARY, name="z")
                m_g.update()
                # candidate tested and fixed to genotype gfix
                for gg in gen_states:
                    m_g.addConstr(p_g[test_individual, gg] == z_g[test_individual, gg])
                m_g.addConstr(z_g[test_individual, gfix] == 1.0)
                for gg in gen_states:
                    if gg != gfix:
                        m_g.addConstr(z_g[test_individual, gg] == 0.0)
                # A–F with candidate tested
                x_state = {i: 1 if (i in S or i == test_individual) else x.get(i,0) for i in I}
                observed_genotypes = {}
                if z_current:
                    for i in I:
                        if i in S and i in z_current:
                            for outcome_key, val in z_current[i].items():
                                if val > 0.5:
                                    observed_genotypes[i] = _project_outcome(outcome_key)
                                    break
                try:
                    add_domain_constraints(
                        m_g, p_g, z_g, I, gen_states,
                        x_state, allele_freq, child_cpds, pedigree,
                        tested_individual=None,
                        observed_genotype=None,
                        observed_genotypes=observed_genotypes
                    )
                except Exception:
                    continue
                # Maximize linearized successor value for this fixed outcome
                phi_expr = gp.LinExpr()
                tested_set = set(S) | {test_individual}
                for i_lin in I:
                    if i_lin in tested_set:
                        for gg in gen_states:
                            phi_expr += z_g[i_lin, gg] * W_snapshot[i_lin][gg]
                    else:
                        for gg in gen_states:
                            phi_expr += p_g[i_lin, gg] * W_snapshot[i_lin][gg]
                m_g.setObjective(phi_expr, GRB.MAXIMIZE)
                m_g.optimize()
                # Evaluate phi value numerically from solution
                if m_g.Status == gp.GRB.OPTIMAL:
                    phi_val = 0.0
                    for i_lin in I:
                        if i_lin in tested_set:
                            for gg in gen_states:
                                phi_val += z_g[i_lin, gg].X * W_snapshot[i_lin][gg]
                        else:
                            for gg in gen_states:
                                phi_val += p_g[i_lin, gg].X * W_snapshot[i_lin][gg]
                    expected_phi += w_g * phi_val
            obj_expected = r_i + expected_phi - Phi_current
            if obj_expected > best_violation:
                best_violation = obj_expected
                best_individual = test_individual
                # Build a lightweight return model carrying p,z,d so caller can extract P_out
                ret = gp.Model(f"row-gen-expected-{test_individual}")
                ret.Params.OutputFlag = 0
                p_ret = ret.addVars(I, gen_states, lb=0.0, ub=1.0, name="p")
                z_ret = ret.addVars(I, gen_states, vtype=GRB.BINARY, name="z")
                d = ret.addVars(I, vtype=gp.GRB.BINARY, name="d")
                ret.update()
                # Set p for chosen individual to p_current (so P_out equals p_current)
                for gg in gen_states:
                    ret.addConstr(p_ret[test_individual, gg] == p_current[test_individual][gg])
                # Fix d choices
                for i in I:
                    if i == test_individual:
                        ret.addConstr(d[i] == 1)
                    else:
                        ret.addConstr(d[i] == 0)
                ret.update()
                ret._violation = best_violation
                ret._best_individual = best_individual
                best_model = ret
    
    # Return a model with the best violation
    # For compatibility with existing code, we'll modify the best model to include d variables
    if best_model is not None:
        # Add d variables to the best model for compatibility
        # Only add d if not already present (expected path already added)
        if best_model.getVarByName("d[{}]".format(I[0])) is None:
            d = best_model.addVars(I, vtype=gp.GRB.BINARY, name="d")
            best_model.update()
            for i in I:
                if i == best_individual:
                    best_model.addConstr(d[i] == 1, name=f"fix_d_{i}")
                else:
                    best_model.addConstr(d[i] == 0, name=f"fix_d_{i}")
            best_model.update()
        # Store violation
        best_model._violation = best_violation
        best_model._best_individual = best_individual
    
    return best_model if best_model is not None else _build_neutral_model(I)


def _build_neutral_model(I):
    """Helper function to return a neutral model when no test candidate works."""
    m = gp.Model("neutral-test")
    m.Params.OutputFlag = 0
    
    # Add d variables for compatibility with expected interface
    d = m.addVars(I, vtype=gp.GRB.BINARY, name="d")
    
    # Set objective to zero (no benefit from testing when all candidates are infeasible)
    m.setObjective(0.0, GRB.MAXIMIZE)
    
    # Ensure exactly one d variable is selected (required by interface)
    m.addConstr(gp.quicksum(d[i] for i in I) == 1, name="select_one")
    
    m.update()
    m.optimize()
    
    # Store neutral violation value
    m._violation = 0.0
    m._best_individual = None
    
    return m
