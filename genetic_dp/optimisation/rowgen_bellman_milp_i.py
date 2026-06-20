import gurobipy as gp
from gurobipy import GRB
from ..models.reward import r_reward_testp
from .constraints import add_domain_constraints


def build_testing_constraint_rowgen_milp_binary(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost):
    """
    Binary reformulation MILP for testing constraint row generation subproblem.
    
    Solves |I| separate linear subproblems instead of one bilinear problem.
    Since d_i ∈ {0,1}, we can enumerate all possible individual selections
    and solve a linear program for each, then take the maximum.
    
    This avoids the bilinear terms d_i * P_{i,g} that cause optimization issues.
    
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
        Dictionary with best solution across all individual selections
    """
    # Extract tested individuals from state S
    tested_individuals = {i for (i, g) in S}
    untested_individuals = [i for i in I if i not in tested_individuals]
    
    if not untested_individuals:
        return {"status": "no_untested", "violation": 0.0, "selected_individual": None}
    
    best_violation = float('-inf')
    best_solution = None
    best_individual = None
    
    # Enumerate each possible individual to test
    for test_individual in untested_individuals:
        # Solve linear subproblem with test_individual fixed as selected
        violation, solution_info = solve_linear_subproblem(
            I, S, gen_states, J,
            p_current, z_current,
            a, b, c, delta,
            x, allele_freq, child_cpds, pedigree,
            W_current, W_prime,
            fixed_cost, variable_cost,
            test_individual
        )
        
        if violation > best_violation:
            best_violation = violation
            best_solution = solution_info
            best_individual = test_individual
    
    return {
        "status": "optimal" if best_solution else "infeasible",
        "violation": best_violation,
        "selected_individual": best_individual,
        "solution_info": best_solution
    }


def solve_linear_subproblem(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost,
        test_individual):
    """
    Solve linear subproblem with test_individual fixed as selected.
    
    Since d_i is fixed (d_{test_individual} = 1, d_j = 0 for j ≠ test_individual),
    all bilinear terms become linear.
    
    Returns:
        Tuple of (objective_value, solution_info)
    """
    # Extract tested individuals from state S
    tested_individuals = {i for (i, g) in S}
    
    # Create model
    m = gp.Model(f"linear-subproblem-{test_individual}")
    m.Params.OutputFlag = 0
    
    # Decision variables (all continuous now since d is fixed)
    # P_{i,g}: current state probabilities
    P = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="P")
    
    # P'_{k,g}: next state (posterior) probabilities after testing test_individual
    P_prime = m.addVars(I, gen_states, lb=0.0, ub=1.0, name="P_prime")
    
    # z_{i,g}: genotype indicators for test_individual
    z = m.addVars(I, gen_states, vtype=GRB.BINARY, name="z")
    
    m.update()
    
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
    
    # CRITICAL FIX: Set P variables to correct posterior probabilities
    # The key insight is that P should equal the Bayesian posterior probabilities,
    # not be optimized variables. This is what bellman_rowgen.py does correctly.
    for i in I:
        for g in gen_states:
            # Fix P to the correct posterior probability from p_current
            m.addConstr(P[i, g] == p_current[i][g],
                       name=f"fix_posterior_{i}_{g}")
    
    # Add only the necessary z constraints (normalization, evidence linking)
    # Skip founder priors for P since we fixed them above
    for i in I:
        if i in tested_individuals:
            # Link z to known genotypes for tested individuals
            m.addConstr(gp.quicksum(z[i, g] for g in gen_states) == 1,
                       name=f"z_norm_{i}")
            if i in observed_genotypes:
                observed_g = observed_genotypes[i]
                for g in gen_states:
                    val = 1 if g == observed_g else 0
                    m.addConstr(z[i, g] == val, name=f"z_observed_{i}_{g}")
    
    # Create z' variables for the successor state (needed for domain constraints)
    z_prime = m.addVars(I, gen_states, vtype=GRB.BINARY, name="z_prime")
    m.update()
    
    # Set up the successor state: test_individual will be observed
    x_state_prime = x_state.copy()
    x_state_prime[test_individual] = 1  # test_individual becomes tested
    
    # Update observed genotypes for successor state
    observed_genotypes_prime = observed_genotypes.copy()
    # Note: we don't know which genotype yet - that's what z[test_individual] determines
    
    # Apply domain constraints to P' and z' variables for the successor state
    # The key insight: P' should satisfy all the same domain constraints as P,
    # but with the additional evidence that test_individual is now tested
    # We don't pass tested_individual/observed_genotype because the optimization
    # needs to determine which genotype outcome occurs
    add_domain_constraints(
        m, P_prime, z_prime, I, gen_states,
        x_state_prime, allele_freq, child_cpds, pedigree,
        tested_individual=None,  # Don't fix the genotype - let optimization choose
        observed_genotype=None,  
        observed_genotypes=observed_genotypes  # Keep existing observations
    )
    
    # Link z' to z for consistency (z' represents the same state as z for already tested)
    for i in I:
        if i in tested_individuals:
            for g in gen_states:
                m.addConstr(z_prime[i, g] == z[i, g], name=f"z_prime_consistency_{i}_{g}")
        elif i == test_individual:
            # For the test individual, z_prime should equal z (the test outcome)
            for g in gen_states:
                m.addConstr(z_prime[i, g] == z[i, g], name=f"z_prime_test_{i}_{g}")
    
    # Ensure exactly one genotype is selected for test_individual
    m.addConstr(gp.quicksum(z[test_individual, g] for g in gen_states) == 1,
               name=f"one_genotype_{test_individual}")
    
    # Build the objective function (now linear since d is fixed)
    # For test_individual: d = 1, for others: d = 0
    # So we only consider the term for test_individual
    
    obj_expr = gp.LinExpr()
    
    # Immediate reward for testing test_individual
    p12_current = p_current[test_individual][1] + p_current[test_individual][2]
    r_immediate = r_reward_testp(test_individual, p12_current, a, b, c, delta, 
                               fixed_cost, variable_cost)
    obj_expr += r_immediate
    
    # Add the expectation term (now linear):
    # Σ_{g_i} P_{i,g_i} [Σ_{k∈I\S'} Σ_g P'_{k,g} W_{k,g} - 
    #                    Σ_{k∈I\S'} Σ_g P_{k,g} W_{k,g} - W_{i,g_i}]
    
    for g_i in gen_states:
        # First sum: Σ_{k∈I\S'} Σ_g P'_{k,g} W_{k,g}
        sum_prime = gp.quicksum(P_prime[k, g] * W_prime.get((k, g), 0.0)
                              for k in I if k not in tested_individuals
                              for g in gen_states)
        
        # Second sum: Σ_{k∈I\S'} Σ_g P_{k,g} W_{k,g}
        sum_current = gp.quicksum(P[k, g] * W_current.get((k, g), 0.0)
                                for k in I if k not in tested_individuals
                                for g in gen_states)
        
        # Third term: W_{test_individual,g_i}
        w_i_g = W_current.get((test_individual, g_i), 0.0)
        
        # Combine the terms
        expectation_term = sum_prime - sum_current - w_i_g
        
        # Add to objective: P_{test_individual,g_i} * expectation_term
        # This is linear since P is a variable and expectation_term is linear in variables
        obj_expr += P[test_individual, g_i] * expectation_term
    
    # Set objective to maximize
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Optimize the model
    m.optimize()
    
    if m.Status != gp.GRB.OPTIMAL:
        return float('-inf'), {"status": f"failed_status_{m.Status}"}
    
    # Extract solution
    P_solution = {(i, g): P[i, g].X for i in I for g in gen_states}
    P_prime_solution = {(i, g): P_prime[i, g].X for i in I for g in gen_states}
    z_solution = {(i, g): z[i, g].X for i in I for g in gen_states}
    
    solution_info = {
        "status": "optimal",
        "objective_value": m.ObjVal,
        "P_solution": P_solution,
        "P_prime_solution": P_prime_solution,
        "z_solution": z_solution,
        "test_individual": test_individual
    }
    
    return m.ObjVal, solution_info


def solve_testing_constraint_rowgen_binary(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost,
        tolerance=1e-6):
    """
    Solve the testing constraint row generation subproblem using binary reformulation.
    
    Returns:
        Tuple of (violation, selected_individual, solution_info)
    """
    result = build_testing_constraint_rowgen_milp_binary(
        I, S, gen_states, J,
        p_current, z_current,
        a, b, c, delta,
        x, allele_freq, child_cpds, pedigree,
        W_current, W_prime,
        fixed_cost, variable_cost
    )
    
    if result["status"] != "optimal":
        return 0.0, None, {"status": result["status"]}
    
    violation = result["violation"]
    
    if violation <= tolerance:
        return violation, None, {"status": "no_violation"}
    
    return violation, result["selected_individual"], result["solution_info"]