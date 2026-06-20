import itertools
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple

GENOTYPE_STATES = (0, 1, 2)

# --- Helper: enumerate all partial-assignment states ---
def partial_states(individuals, gen_states):
    """
    Return a list of frozenset-of-(i,g) tuples representing 
    every partial assignment s ⊆ I×G (observed tests).
    """
    states = []
    for r in range(len(individuals)+1):
        for subset in itertools.combinations(individuals, r):
            # for each choice of r tested individuals, assign every r-tuple of genotypes
            for genotype_tuple in itertools.product(gen_states, repeat=r):
                # create frozenset of (i,g) pairs
                state = frozenset(zip(subset, genotype_tuple))
                states.append(state)
    return states

# --- Build full joint distribution over all genotypes ---
def build_full_joint(
    pedigree,
    gen_states,
    allele_freq,
    child_cpds,
    *,
    genes=None,
    allele_freqs=None,
    base_gen_states=GENOTYPE_STATES,
):
    """
    Returns dict mapping (g_i1, g_i2, ...) → P(g_i1, g_i2, ...)
    for all individuals in the pedigree.
    """
    founders = pedigree.get_founders()
    offspring = pedigree.get_offspring()
    all_individuals = pedigree.to_list()

    multi_gene = genes is not None and len(tuple(genes)) > 0

    # Backwards compatibility: single-gene case
    if not multi_gene:
        joint = {}
        def hw(p): return {(0):(1-p)**2, (1):2*p*(1-p), (2):p**2}
        founder_probs = {f: hw(allele_freq) for f in founders}

        for genotype_tuple in itertools.product(gen_states, repeat=len(all_individuals)):
            genotype_map = dict(zip(all_individuals, genotype_tuple))
            prob = 1.0

            for f in founders:
                prob *= founder_probs[f][genotype_map[f]]

            for child in offspring:
                parent1, parent2 = pedigree.get_parents(child)
                child_cpd = child_cpds[child]
                idx = genotype_map[parent1] * 3 + genotype_map[parent2]
                prob *= child_cpd[genotype_map[child], idx]
        
            joint[tuple(genotype_map[i] for i in all_individuals)] = prob

        return joint

    gene_list = tuple(genes)
    tuple_states: Sequence[Tuple[int, ...]] = gen_states
    if not isinstance(tuple_states[0], tuple):
        tuple_states = list(itertools.product(base_gen_states, repeat=len(gene_list)))

    def hardy_weinberg(p: float) -> Mapping[int, float]:
        return {
            0: (1 - p) ** 2,
            1: 2 * p * (1 - p),
            2: p ** 2,
        }

    hw_cache = {
        gene: hardy_weinberg(allele_freqs.get(gene, allele_freq) if allele_freqs else allele_freq)
        for gene in gene_list
    }

    joint = {}
    for assignment in itertools.product(tuple_states, repeat=len(all_individuals)):
        genotype_map = dict(zip(all_individuals, assignment))
        prob = 1.0

        for founder in founders:
            tuple_value = genotype_map[founder]
            for idx, gene in enumerate(gene_list):
                prob *= hw_cache[gene][tuple_value[idx]]

        for child in offspring:
            parent1, parent2 = pedigree.get_parents(child)
            child_cpd = child_cpds[child]
            child_tuple = genotype_map[child]
            father_tuple = genotype_map[parent1]
            mother_tuple = genotype_map[parent2]
            for idx, _ in enumerate(gene_list):
                g_child = child_tuple[idx]
                g_father = father_tuple[idx]
                g_mother = mother_tuple[idx]
                table_index = g_father * 3 + g_mother
                prob *= child_cpd[g_child, table_index]

        joint[tuple(genotype_map[i] for i in all_individuals)] = prob

    return joint

def build_belief_map(pedigree, gen_states, joint):
    """
    Returns belief: state → p_s, where p_s[i][g] = P(genotype_i=g | observations s)
    and state is frozenset of (i,g) observed.
    """
    belief = {}
    all_individuals = pedigree.to_list()
    states = partial_states(all_individuals, gen_states)

    for s in states:
        total = 0.0
        p_s = {i: {g: 0.0 for g in gen_states} for i in all_individuals}

        for genotype_tuple, prob in joint.items():
            genotype_map = dict(zip(all_individuals, genotype_tuple))
            ok = True
            for (i_obs, g_obs) in s:
                if genotype_map[i_obs] != g_obs:
                    ok = False
                    break
            if not ok:
                continue
            
            total += prob
            for i in all_individuals:
                p_s[i][genotype_map[i]] += prob
        
        if total > 0:
            for i in all_individuals:
                for g in gen_states:
                    p_s[i][g] /= total
        else:
            for i in all_individuals:
                for g in gen_states:
                    p_s[i][g] = 0.0
        belief[s] = p_s
    return belief


def lift_tuple_posteriors_to_genes(
    p_state: Mapping[Hashable, Mapping[Hashable, float]],
    gene_list: Iterable[str],
    base_states: Iterable[int] = GENOTYPE_STATES,
) -> Dict[str, Dict[Hashable, Dict[int, float]]]:
    """
    Convert per-person tuple distributions into per-gene marginals.
    """
    gene_list = tuple(gene_list)
    if not gene_list:
        return {}

    gene_probs: Dict[str, Dict[Hashable, Dict[int, float]]] = {
        gene: {} for gene in gene_list
    }

    for person, dist in p_state.items():
        for gene in gene_list:
            gene_probs[gene][person] = {g: 0.0 for g in base_states}

        for outcome, prob in dist.items():
            if prob <= 0.0:
                continue
            if isinstance(outcome, tuple):
                for idx, gene in enumerate(gene_list):
                    gene_probs[gene][person][outcome[idx]] += prob
            else:
                # Scalar outcome replicated across all genes
                for gene in gene_list:
                    gene_probs[gene][person][outcome] += prob

    return gene_probs
