import numpy as np
from typing import Dict, Iterable, Tuple

GENOTYPE_STATES = (0, 1, 2)

try:  # pgmpy is optional (some environments lack pandas/dateutil)
    from pgmpy.factors.discrete import TabularCPD as _PgmpyTabularCPD  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _PgmpyTabularCPD = None


if _PgmpyTabularCPD is not None:
    TabularCPD = _PgmpyTabularCPD
else:

    class TabularCPD:  # noqa: D101 - lightweight fallback
        def __init__(
            self,
            *,
            variable: str,
            variable_card: int,
            values,
            evidence=None,
            evidence_card=None,
        ):
            self.variable = variable
            self.variable_card = int(variable_card)
            self.evidence = list(evidence) if evidence else []
            self.evidence_card = list(evidence_card) if evidence_card else []
            self.values = np.asarray(values, dtype=float)

        def get_evidence(self):
            return list(self.evidence)

        def get_value(self, **assignments):
            values = self.values.reshape(self.variable_card, -1)
            var_value = int(assignments[self.variable])
            if not self.evidence:
                return float(values[var_value, 0])
            col = 0
            for parent in self.evidence:
                col = col * 3 + int(assignments[parent])
            return float(values[var_value, col])

def genotype_node_name(individual: str, gene: str = None) -> str:
    """
    Consistently name genotype variables. Multi-gene models append the gene label.
    """
    if gene:
        return f"{individual}::{gene}"
    return individual

def founder_prior_distribution(allele_freq: float) -> Dict[int, float]:
    """
    Helper for converting allele frequency to genotype mass function.
    """
    p = allele_freq
    return {
        0: (1 - p) ** 2,
        1: 2 * p * (1 - p),
        2: p ** 2,
    }

def make_founder_genotype_cpd(individual_name: str, allele_freq=0.01, gene: str = None):
    """
    Creates a TabularCPD for a founder's genotype: {0=NN, 1=ND, 2=DD}.
    allele_freq = p(D) in the population.
    """
    node = genotype_node_name(individual_name, gene)
    prior = founder_prior_distribution(allele_freq)
    return TabularCPD(
        variable=node,
        variable_card=3,
        values=[[prior[0]], [prior[1]], [prior[2]]]
    )

def make_inheritance_genotype_cpd_with_table(child_name: str, father_name: str, mother_name: str, gene: str = None):
    """
    Creates a TabularCPD for a child's genotype given the father and mother's genotypes,
    using Mendelian inheritance, and returns both the CPD and the raw CPD table.
    The table is 3 x 9: 3 outcomes (0=NN, 1=ND, 2=DD) and 9 columns (one per parent combination).
    """
    child_cond = np.zeros((3, 9))  # 3 outcomes x 9 parent-combinations
    col = 0
    for f_state in GENOTYPE_STATES:
        if f_state == 0:
            f_pass = 0.0
        elif f_state == 1:
            f_pass = 0.5
        else:
            f_pass = 1.0
        for m_state in GENOTYPE_STATES:
            if m_state == 0:
                m_pass = 0.0
            elif m_state == 1:
                m_pass = 0.5
            else:
                m_pass = 1.0
            # Mendelian probabilities:
            p_nn = (1 - f_pass) * (1 - m_pass)
            p_dd = f_pass * m_pass
            p_nd = 1.0 - (p_nn + p_dd)
            child_cond[0, col] = p_nn
            child_cond[1, col] = p_nd
            child_cond[2, col] = p_dd
            col += 1

    variable = genotype_node_name(child_name, gene)
    father_var = genotype_node_name(father_name, gene)
    mother_var = genotype_node_name(mother_name, gene)

    cpd = TabularCPD(
        variable=variable,
        variable_card=3,
        values=child_cond,
        evidence=[father_var, mother_var],
        evidence_card=[3, 3]
    )
    return cpd, child_cond

def make_multigene_founder_cpds(
    individual_name: str,
    genes: Iterable[str],
    allele_freq_lookup: Dict[str, float],
) -> Dict[str, TabularCPD]:
    """
    Build founder CPDs for every gene, returning a mapping gene → CPD.
    """
    cpds = {}
    for gene in genes:
        freq = allele_freq_lookup.get(gene)
        if freq is None:
            raise KeyError(f"Missing allele frequency for gene '{gene}'")
        cpds[gene] = make_founder_genotype_cpd(individual_name, allele_freq=freq, gene=gene)
    return cpds

def make_multigene_inheritance_cpds_with_tables(
    child_name: str,
    father_name: str,
    mother_name: str,
    genes: Iterable[str],
) -> Dict[str, Tuple[TabularCPD, np.ndarray]]:
    """
    Produce CPDs and raw tables for each gene independently.
    """
    cpds = {}
    for gene in genes:
        cpd, table = make_inheritance_genotype_cpd_with_table(
            child_name,
            father_name,
            mother_name,
            gene=gene,
        )
        cpds[gene] = (cpd, table)
    return cpds
