from dataclasses import dataclass, field
from typing import Dict, Mapping, Iterable, Tuple, Optional

@dataclass
class Config:
    """A configuration class for the genetic testing problem."""
    fixed_cost: float = 0.01
    variable_cost: float = 0.02
    allele_freq: float = 0.1
    a: Dict[str, float] = field(default_factory=dict)
    b: Dict[str, float] = field(default_factory=dict)
    c: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    genes: Tuple[str, ...] = field(default_factory=tuple)
    allele_freqs: Dict[str, float] = field(default_factory=dict)
    a_gene: Dict[str, Dict[str, float]] = field(default_factory=dict)
    b_gene: Dict[str, Dict[str, float]] = field(default_factory=dict)
    delta_gene: Dict[str, Dict[str, float]] = field(default_factory=dict)
    c_gene: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def gene_list(self) -> Tuple[str, ...]:
        if self.genes:
            return self.genes
        return ("gene",)

    def get_allele_freq(self, gene: Optional[str] = None) -> float:
        if gene is None or not self.allele_freqs:
            return self.allele_freq
        return self.allele_freqs.get(gene, self.allele_freq)

    def get_reward_coeff(self, coeff: str, individual: str, gene: Optional[str] = None) -> float:
        """
        Retrieve reward coefficients, supporting optional per-gene overrides.
        coeff ∈ {'a','b','c','delta'}
        """
        if coeff not in {"a", "b", "c", "delta"}:
            raise ValueError(f"Unknown coefficient '{coeff}'")

        gene_maps = {
            "a": self.a_gene,
            "b": self.b_gene,
            "c": self.c_gene,
            "delta": self.delta_gene
        }
        aggregate = getattr(self, coeff)

        if gene is None or not gene_maps[coeff]:
            return aggregate.get(individual, 0.0)

        per_gene = gene_maps[coeff].get(gene, {})
        if individual in per_gene:
            return per_gene[individual]
        return aggregate.get(individual, 0.0)

def get_config(
    individuals,
    a_base=-0.1,
    b_base=-0.1,
    c_base=0.0,
    delta_base=0.7,
    child_multiplier=2.0,
    pedigree=None,
    genes: Optional[Iterable[str]] = None,
    allele_freq=0.1,
    allele_freqs: Optional[Mapping[str, float]] = None,
    per_gene_a: Optional[Mapping[str, float]] = None,
    per_gene_b: Optional[Mapping[str, float]] = None,
    per_gene_c: Optional[Mapping[str, float]] = None,
    per_gene_delta: Optional[Mapping[str, float]] = None,
):
    """
    Creates a configuration object with dynamically generated reward parameters.

    Args:
        individuals (list): A list of individual names.
        a_base (float): The base value for the 'a' reward parameter.
        b_base (float): The base value for the 'b' reward parameter.
        c_base (float): The base value for the 'c' reward parameter.
        delta_base (float): The base value for the 'delta' reward parameter.
        child_multiplier (float): The multiplier for reward parameters for younger generation individuals.
        pedigree (Pedigree, optional): The pedigree object to determine generational structure.
        genes (Iterable[str], optional): Names of genes to include in the additive model.
        allele_freqs (Mapping[str, float], optional): Per-gene allele frequencies. Falls back to allele_freq if missing.
        per_gene_* (Mapping[str, float], optional): Overrides for base reward parameters per gene.

    Returns:
        Config: A configuration object.
    """
    genes_tuple: Tuple[str, ...] = tuple(genes) if genes else tuple()
    allele_freq_map: Dict[str, float] = {}
    if genes_tuple:
        for gene in genes_tuple:
            if allele_freqs and gene in allele_freqs:
                allele_freq_map[gene] = allele_freqs[gene]
            else:
                pedigree_map = getattr(pedigree, "allele_freq_by_gene", {}) if pedigree else {}
                allele_freq_map[gene] = pedigree_map.get(gene, allele_freq)

    a = {i: a_base for i in individuals}
    b = {i: b_base for i in individuals}
    c = {i: c_base for i in individuals}
    delta = {i: delta_base for i in individuals}

    a_gene: Dict[str, Dict[str, float]] = {gene: {} for gene in genes_tuple}
    b_gene: Dict[str, Dict[str, float]] = {gene: {} for gene in genes_tuple}
    c_gene: Dict[str, Dict[str, float]] = {gene: {} for gene in genes_tuple}
    delta_gene: Dict[str, Dict[str, float]] = {gene: {} for gene in genes_tuple}

    # Apply multiplier based on generational structure
    def get_generation_level(individual, pedigree_obj):
        """
        Calculate generation level (0 = founders, 1 = their children, 2 = grandchildren, etc.)
        """
        if not pedigree_obj:
            # Fallback to name-based detection if no pedigree provided
            return 1 if "Child" in individual else 0
        
        # Recursive calculation of generation level
        
        # If individual is a founder (no parents), generation = 0
        parents = pedigree_obj.get_parents(individual)
        if not parents:
            return 0
            
        # Find the maximum generation of parents + 1
        max_parent_gen = 0
        for parent in parents:
            parent_gen = get_generation_level(parent, pedigree_obj)
            max_parent_gen = max(max_parent_gen, parent_gen)
        
        return max_parent_gen + 1

    # Apply child_multiplier to non-founder generations (generation > 0)
    for ind in individuals:
        generation = get_generation_level(ind, pedigree)
        if generation > 0:  # Apply multiplier to children, grandchildren, etc.
            a[ind] *= child_multiplier
            b[ind] *= child_multiplier
            c[ind] *= child_multiplier
            for gene in genes_tuple:
                scale = child_multiplier
                a_gene[gene][ind] = scale * (per_gene_a.get(gene, a_base) if per_gene_a else a_base)
                b_gene[gene][ind] = scale * (per_gene_b.get(gene, b_base) if per_gene_b else b_base)
                c_gene[gene][ind] = scale * (per_gene_c.get(gene, c_base) if per_gene_c else c_base)
                delta_gene[gene][ind] = per_gene_delta.get(gene, delta_base) if per_gene_delta else delta_base
        else:
            for gene in genes_tuple:
                a_gene[gene][ind] = per_gene_a.get(gene, a_base) if per_gene_a else a_base
                b_gene[gene][ind] = per_gene_b.get(gene, b_base) if per_gene_b else b_base
                c_gene[gene][ind] = per_gene_c.get(gene, c_base) if per_gene_c else c_base
                delta_gene[gene][ind] = per_gene_delta.get(gene, delta_base) if per_gene_delta else delta_base

    # Ensure aggregate coefficients reflect per-gene data if provided
    if genes_tuple:
        for ind in individuals:
            a[ind] = sum(a_gene[gene][ind] for gene in genes_tuple)
            b[ind] = sum(b_gene[gene][ind] for gene in genes_tuple)
            c[ind] = sum(c_gene[gene][ind] for gene in genes_tuple)
            delta[ind] = sum(delta_gene[gene][ind] for gene in genes_tuple) / len(genes_tuple)

    config = Config(
        a=a,
        b=b,
        c=c,
        delta=delta,
        genes=genes_tuple,
        allele_freqs=allele_freq_map,
        a_gene=a_gene,
        b_gene=b_gene,
        c_gene=c_gene,
        delta_gene=delta_gene,
    )

    if not genes_tuple:
        if pedigree and hasattr(pedigree, "allele_freq"):
            config.allele_freq = pedigree.allele_freq
        else:
            config.allele_freq = allele_freq
    else:
        config.allele_freq = config.allele_freqs.get(genes_tuple[0], config.allele_freq)

    return config
