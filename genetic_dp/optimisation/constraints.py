import gurobipy as gp

def add_domain_constraints(
        model,
        p, z,                    # tupledicts:  p[i,g]  z[i,g]
        I,                       # list of individuals
        gen_states,              # [0,1,2]
        x,                       # {i:0/1}  (1 ⇒ already tested)
        allele_freq,             # founder D-allele frequency
        child_cpds,              # dict of {child_name: cpd_table}
        pedigree,                # pedigree object
        tested_individual=None,  # name of the one whose genotype is observed
        observed_genotype=None,  # 0/1/2  (only if tested_individual given)
        observed_genotypes=None  # dict {individual: genotype} for all observed genotypes
):
    """
    Adds constraints A–F to `model`:

      A  Normalisation for every p_{i,*}
      B  Linking p = z for already-tested individuals
      C  Founder priors
      D  Evidence: fix z for the currently tested person
      E  CPD inheritance
      F  Logical Mendelian constraints on z
    """
    # ---------- A  normalisation ----------
    for i in I:
        model.addConstr(
            gp.quicksum(p[i, g] for g in gen_states) == 1,
            name=f"norm_{i}"
        )

    # ---------- B  linking for tested history ----------
    for i in I:
        if x.get(i, 0) == 1:
            model.addConstr(
                gp.quicksum(z[i, g] for g in gen_states) == 1,
                name=f"linkSum_{i}"
            )
            for g in gen_states:
                model.addConstr(p[i, g] == z[i, g],
                                name=f"link_{i}_{g}")
            
            # If we know the specific genotype, fix z accordingly
            if observed_genotypes and i in observed_genotypes:
                observed_g = observed_genotypes[i]
                for g in gen_states:
                    val = 1 if g == observed_g else 0
                    model.addConstr(z[i, g] == val,
                                    name=f"observed_{i}_{g}")

    # ---------- C  founder priors ----------
    def founder_prior(pD):
        return [(1 - pD)**2, 2 * pD * (1 - pD), pD**2]

    for founder in pedigree.get_founders():
        if x.get(founder, 0) == 0:          # only if still untested
            prior = founder_prior(allele_freq)
            for g in gen_states:
                model.addConstr(p[founder, g] == prior[g],
                                name=f"prior_{founder}_{g}")

    # ---------- D  evidence for *currently* tested person ----------
    if tested_individual is not None:
        gi = observed_genotype
        # force exactly one z-entry to 1, others 0
        for g in gen_states:
            val = 1 if g == gi else 0
            model.addConstr(z[tested_individual, g] == val,
                            name=f"evid_{tested_individual}_{g}")
        # and link p = z
        for g in gen_states:
            model.addConstr(p[tested_individual, g] == z[tested_individual, g],
                            name=f"evid_link_{g}")

    # ---------- E  CPD for Child ----------
    for j in pedigree.get_offspring():
        parents = pedigree.get_parents(j)
        if len(parents) == 2:
            p1, p2 = parents
            child_cpd = child_cpds[j]
            for g in gen_states:
                rhs = gp.quicksum(
                    child_cpd[g, u * 3 + v] * p[p1, u] * p[p2, v]
                    for u in gen_states for v in gen_states
                )
                model.addQConstr(p[j, g] == rhs, name=f"inherit_{j}_{g}")

    # ---------- F  logical Mendelian z-constraints ----------
    for j in pedigree.get_offspring():
        parents = pedigree.get_parents(j)
        if len(parents) == 2:
            p1, p2 = parents
            # 1) no diseased allele in parents ⇒ none in child
            model.addConstr(z[j, 0] >= z[p1, 0] + z[p2, 0] - 1, name=f"mendel1a_{j}")
            model.addConstr(z[j, 1] <= 2 - (z[p1, 0] + z[p2, 0]), name=f"mendel1b_{j}")
            model.addConstr(z[j, 2] <= 2 - (z[p1, 0] + z[p2, 0]), name=f"mendel1c_{j}")

            # 2) two diseased alleles in parents ⇒ two in child
            model.addConstr(z[j, 2] >= z[p1, 2] + z[p2, 2] - 1, name=f"mendel2a_{j}")
            model.addConstr(z[j, 0] <= 2 - (z[p1, 2] + z[p2, 2]), name=f"mendel2b_{j}")
            model.addConstr(z[j, 1] <= 2 - (z[p1, 2] + z[p2, 2]), name=f"mendel2c_{j}")

            # 3) one parent 2, other 0 ⇒ child must be 1
            model.addConstr(z[j, 1] >= z[p1, 2] + z[p2, 0] - 1, name=f"mendel3a_{j}")
            model.addConstr(z[j, 1] >= z[p1, 0] + z[p2, 2] - 1, name=f"mendel3b_{j}")
            model.addConstr(z[j, 0] <= 2 - (z[p1, 2] + z[p2, 0]), name=f"mendel3c_{j}")
            model.addConstr(z[j, 0] <= 2 - (z[p1, 0] + z[p2, 2]), name=f"mendel3d_{j}")
            model.addConstr(z[j, 2] <= 2 - (z[p1, 2] + z[p2, 0]), name=f"mendel3e_{j}")
            model.addConstr(z[j, 2] <= 2 - (z[p1, 0] + z[p2, 2]), name=f"mendel3f_{j}")

            # 4) parent (2) + parent (1) ⇒ child ≥ 1
            model.addConstr(z[j, 0] <= 2 - (z[p1, 2] + z[p2, 1]), name=f"mendel4a_{j}")
            model.addConstr(z[j, 0] <= 2 - (z[p1, 1] + z[p2, 2]), name=f"mendel4b_{j}")

            # 5) parent (1) + parent (0) ⇒ child ≤ 1
            model.addConstr(z[j, 2] <= 2 - (z[p1, 1] + z[p2, 0]), name=f"mendel5a_{j}")
            model.addConstr(z[j, 2] <= 2 - (z[p1, 0] + z[p2, 1]), name=f"mendel5b_{j}")
