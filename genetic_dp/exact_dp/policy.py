from ..models.reward import r_reward, r_reward_test
from .utils import lift_tuple_posteriors_to_genes, GENOTYPE_STATES
from ..models.belief import InferenceResult
from ..models.outcomes import project_state_by_gene

# --- Extract policy from Φ* ---
def extract_policy(
    individuals,
    gen_states,
    a,
    b,
    c,
    delta,
    Phi_star,
    belief,
    fixed_cost,
    variable_cost,
    *,
    genes=None,
    Phi_star_gene=None,
    a_gene=None,
    b_gene=None,
    c_gene=None,
    delta_gene=None,
    base_gen_states=GENOTYPE_STATES,
):
    gene_list = tuple(genes) if genes else tuple()
    per_gene_active = bool(gene_list) and Phi_star_gene is not None

    def _phi_sum(state):
        if not per_gene_active:
            return Phi_star.get(state, 0.0)
        proj = project_state_by_gene(state, gene_list)
        return sum(Phi_star_gene.get(g, {}).get(proj.get(g, frozenset()), 0.0) for g in gene_list)

    policy = {}
    for s, entry in belief.items():
        if isinstance(entry, InferenceResult):
            tuple_pmfs_state = entry.get_tuple_pmfs()
            p_s = entry.marginals
            per_gene_probs = entry.get_per_gene_probs() if genes else None
        else:
            tuple_pmfs_state = {}
            p_s = entry
            per_gene_probs = None
        tested = {i for i,_ in s}
        if genes and not per_gene_probs:
            if tuple_pmfs_state:
                per_gene_probs = lift_tuple_posteriors_to_genes(tuple_pmfs_state, genes, base_gen_states)
            else:
                per_gene_probs = lift_tuple_posteriors_to_genes(p_s, genes, base_gen_states)
        # stopping value
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
        
        # test values
        best_i, best_Q = None, -1e9
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
            V = 0.0
            if tuple_pmfs_state and i in tuple_pmfs_state:
                for outcome, prob_g in tuple_pmfs_state.get(i, {}).items():
                    succ = frozenset(s | {(i, outcome)})
                    V += prob_g * _phi_sum(succ)
            else:
                for g, prob_g in p_s[i].items():
                    succ = frozenset(s | {(i, g)})
                    V += prob_g * _phi_sum(succ)
            Q = rsi + V
            if Q > best_Q:
                best_Q, best_i = Q, i
        if Rstop >= best_Q:
            policy[s] = ("stop", None, Rstop)
        else:
            policy[s] = ("test", best_i, best_Q)
    return policy
