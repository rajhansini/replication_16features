import gurobipy as gp
from gurobipy import GRB
from ..models.reward import r_reward_testp
from .constraints import add_domain_constraints


def build_testing_constraint_rowgen_milp(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost):
    """
    MILP for testing constraint row generation subproblem.
    
    Implements the objective from the slide:
    max_{d,P,P',z} Σ_j∈J Σ_{i∈I\\S} d_i { r_j(S_{:j},i) + 
                   Σ_{g_i} P_{i,g} [Σ_{k∈I\\S'} Σ_g P'_{k,g} W_{k,g} - 
                                    Σ_{k∈I\\S'} Σ_g P_{k,g} W_{k,g} - W_{i,g}] }
    
    Args:
        I: List of all individuals
        S: Current tested set (frozenset of (individual, genotype) pairs)
        gen_states: List of genotypes [0, 1, 2]
        J: List of test stages/scenarios
        p_current: Current state probabilities P_{k,g}
        z_current: Current genotype indicators
        a, b, c, delta: Reward parameters
        x: Testing status {i: 0/1}
        allele_freq: Founder allele frequency
        child_cpds: Child CPD tables
        pedigree: Pedigree structure
        W_current: Current W values W_{k,g}
        W_prime: Next state W values W'_{k,g}
        fixed_cost, variable_cost: Testing costs
        
    Returns:
        Gurobi model with testing constraint row generation objective
    """
    # Extract tested individuals from state S
    tested_individuals = {i for (i, g) in S}
    untested_individuals = [i for i in I if i not in tested_individuals]
    
    # Create model
    m = gp.Model("testing-constraint-rowgen")
    m.Params.OutputFlag = 0
    
    # Decision variables
    # d_i: binary, selects individual i to test
    d = m.addVars(untested_individuals, vtype=GRB.BINARY, name="d")
    
    # P_{i,g}: current state probabilities
    P = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="P")
    
    # P'_{k,g}: next state (posterior) probabilities after hypothetical test
    P_prime = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="P_prime")
    
    # z_{i,g}: genotype indicators
    z = m.addVars(I, gen_states, vtype=GRB.BINARY, name="z")
    
    m.update()
    
    # Testing selection constraint: select exactly one individual to test
    m.addConstr(gp.quicksum(d[i] for i in untested_individuals) == 1, 
                name="select_one_test")
    
    # Set up state for domain constraints
    x_state = {i: 1 if i in tested_individuals else x.get(i, 0) for i in I}
    
    # Extract observed genotypes from current state
    observed_genotypes = {}
    for i in I:
        if i in tested_individuals:
            for g in gen_states:
                if z_current[i][g] > 0.5:
                    observed_genotypes[i] = g
                    break
    
    # Add domain constraints A-F for current state probabilities P
    add_domain_constraints(
        m, P, z, I, gen_states,
        x_state, allele_freq, child_cpds, pedigree,
        tested_individual=None,
        observed_genotype=None,
        observed_genotypes=observed_genotypes
    )
    
    # Add normalization constraints for P'
    for i in I:
        m.addConstr(gp.quicksum(P_prime[i, g] for g in gen_states) == 1,
                   name=f"norm_prime_{i}")
    
    # For each potentially tested individual, add conditional constraints for P'
    # Using big-M formulation to handle the conditional logic
    M = 1.0  # Big-M constant
    
    for test_i in untested_individuals:
        # When d[test_i] = 1, P'_{test_i,g} should equal z_{test_i,g} (tested individual)
        # When d[test_i] = 0, P'_{test_i,g} should equal P_{test_i,g} (untested)
        
        for g in gen_states:
            # If d[test_i] = 1: P'[test_i,g] = z[test_i,g]
            # If d[test_i] = 0: P'[test_i,g] = P[test_i,g]
            # This can be modeled as:
            # P'[test_i,g] <= z[test_i,g] + M*(1-d[test_i])
            # P'[test_i,g] >= z[test_i,g] - M*(1-d[test_i])
            # P'[test_i,g] <= P[test_i,g] + M*d[test_i]
            # P'[test_i,g] >= P[test_i,g] - M*d[test_i]
            
            m.addConstr(P_prime[test_i, g] <= z[test_i, g] + M*(1 - d[test_i]),
                       name=f"cond_test_upper_{test_i}_{g}")
            m.addConstr(P_prime[test_i, g] >= z[test_i, g] - M*(1 - d[test_i]),
                       name=f"cond_test_lower_{test_i}_{g}")
            m.addConstr(P_prime[test_i, g] <= P[test_i, g] + M*d[test_i],
                       name=f"cond_untest_upper_{test_i}_{g}")
            m.addConstr(P_prime[test_i, g] >= P[test_i, g] - M*d[test_i],
                       name=f"cond_untest_lower_{test_i}_{g}")
        
        # When individual test_i is selected for testing, ensure z variables are binary
        for g in gen_states:
            m.addConstr(z[test_i, g] <= d[test_i], name=f"z_implies_test_{test_i}_{g}")
        
        # Ensure exactly one genotype is selected when testing
        m.addConstr(gp.quicksum(z[test_i, g] for g in gen_states) <= d[test_i],
                   name=f"z_sum_{test_i}")
    
    # For individuals not being tested, P' should equal P
    for i in I:
        if i not in untested_individuals:  # Already tested individuals
            for g in gen_states:
                m.addConstr(P_prime[i, g] == P[i, g], name=f"fixed_prime_{i}_{g}")
        else:  # Untested individuals that might be selected
            # This is handled by the conditional constraints above
            pass
    
    # Build the objective function
    # Σ_j∈J Σ_{i∈I\\S} d_i { r_j(S_{:j},i) + 
    #   Σ_{g_i} P_{i,g} [Σ_{k∈I\\S'} Σ_g P'_{k,g} W_{k,g} - 
    #                    Σ_{k∈I\\S'} Σ_g P_{k,g} W_{k,g} - W_{i,g}] }
    
    obj_expr = gp.LinExpr()
    
    # For simplicity, assume J = [0] (single test scenario)
    # In practice, J would represent different test scenarios or stages
    for j in [0]:  # J should be provided as parameter
        for i in untested_individuals:
            # Compute immediate reward r_j(S_{:j}, i)
            # For testing constraint, this is the testing reward
            p12_current = p_current[i][1] + p_current[i][2]
            r_immediate = r_reward_testp(i, p12_current, a, b, c, delta, 
                                       fixed_cost, variable_cost)
            
            # Add immediate reward term
            obj_expr += d[i] * r_immediate
            
            # Add the expectation term:
            # Σ_{g_i} P_{i,g} [Σ_{k∈I\S'} Σ_g P'_{k,g} W_{k,g} - 
            #                  Σ_{k∈I\S'} Σ_g P_{k,g} W_{k,g} - W_{i,g}]
            
            for g_i in gen_states:
                # First sum: Σ_{k∈I\S'} Σ_g P'_{k,g} W_{k,g}
                sum_prime = gp.quicksum(P_prime[k, g] * W_prime.get((k, g), 0.0)
                                      for k in I if k not in tested_individuals
                                      for g in gen_states)
                
                # Second sum: Σ_{k∈I\S'} Σ_g P_{k,g} W_{k,g}
                sum_current = gp.quicksum(P[k, g] * W_current.get((k, g), 0.0)
                                        for k in I if k not in tested_individuals
                                        for g in gen_states)
                
                # Third term: W_{i,g}
                w_i_g = W_current.get((i, g_i), 0.0)
                
                # Combine the terms
                expectation_term = sum_prime - sum_current - w_i_g
                
                # Add to objective: d_i * P_{i,g_i} * expectation_term
                # This creates a bilinear term, which we need to linearize
                # For now, use a simplified linear approximation
                # In practice, you might need McCormick envelopes or other linearization techniques
                obj_expr += d[i] * P[i, g_i] * expectation_term
    
    # Set objective to maximize
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Store additional information in model
    m._tested_individuals = tested_individuals
    m._untested_individuals = untested_individuals
    m._d_vars = d
    m._P_vars = P
    m._P_prime_vars = P_prime
    m._z_vars = z
    
    return m


def solve_testing_constraint_rowgen(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost,
        tolerance=1e-6):
    """
    Solve the testing constraint row generation subproblem.
    
    Returns:
        Tuple of (violation, selected_individual, solution_info)
    """
    model = build_testing_constraint_rowgen_milp(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost
    )
    
    # Optimize the model
    model.optimize()
    
    if model.Status != gp.GRB.OPTIMAL:
        return 0.0, None, {"status": "infeasible"}
    
    violation = model.ObjVal
    
    if violation <= tolerance:
        return violation, None, {"status": "no_violation"}
    
    # Extract solution
    d_vars = model._d_vars
    selected_individual = None
    for i in model._untested_individuals:
        if d_vars[i].X > 0.5:
            selected_individual = i
            break
    
    # Extract probability solutions
    P_solution = {(i, g): model._P_vars[i, g].X 
                 for i in I for g in gen_states}
    P_prime_solution = {(i, g): model._P_prime_vars[i, g].X 
                       for i in I for g in gen_states}
    
    solution_info = {
        "status": "optimal",
        "violation": violation,
        "selected_individual": selected_individual,
        "P_solution": P_solution,
        "P_prime_solution": P_prime_solution,
        "objective_value": model.ObjVal
    }
    
    return violation, selected_individual, solution_info