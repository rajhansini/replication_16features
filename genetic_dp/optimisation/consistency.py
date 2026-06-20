
import itertools

def get_alleles(genotype):
    if genotype == 0: return {0}
    if genotype == 1: return {0, 1}
    if genotype == 2: return {1}
    raise ValueError(f"Invalid genotype: {genotype}")

def get_genotype(allele1, allele2):
    return allele1 + allele2

class ConsistencyManager:
    """
    Manages genetic consistency within a pedigree.
    """
    def __init__(self, individuals, pedigree):
        self.individuals = individuals
        self.pedigree = pedigree
        self.possible_genotypes = {i: {0, 1, 2} for i in self.individuals}

    def set_known_genotype(self, individual, genotype):
        """Fix the genotype for an individual."""
        self.possible_genotypes[individual] = {genotype}

    def check_consistency(self, state):
        """
        Checks if the given state is genetically consistent by using constraint propagation.

        Args:
            state (dict): A dictionary of known genotypes, e.g., {'Child': 0}.

        Returns:
            bool: True if the state is consistent, False otherwise.
        """
        # Reset possibilities and apply the current known state
        self.possible_genotypes = {i: {0, 1, 2} for i in self.individuals}
        for person, genotype in state.items():
            if genotype not in self.possible_genotypes[person]:
                return False # Initial state is already inconsistent
            self.possible_genotypes[person] = {genotype}

        # Iteratively propagate constraints until no more changes occur
        changed = True
        while changed:
            changed = False
            # Propagate from parents to children
            if self._propagate_down():
                changed = True
            # Propagate from children to parents
            if self._propagate_up():
                changed = True
            
            # If any individual has no possible genotypes, the state is inconsistent
            if any(not s for s in self.possible_genotypes.values()):
                return False

        return True

    def _propagate_down(self):
        """Rule out child genotypes based on parent genotypes."""
        changed = False
        for child in self.individuals:
            parents = self.pedigree.get_parents(child)
            if not parents:
                continue
            
            father, mother = parents
            
            possible_child_genotypes = set()
            parent_combinations = itertools.product(
                self.possible_genotypes[father], 
                self.possible_genotypes[mother]
            )
            
            for f_g, m_g in parent_combinations:
                f_alleles = get_alleles(f_g)
                m_alleles = get_alleles(m_g)
                for f_a in f_alleles:
                    for m_a in m_alleles:
                        possible_child_genotypes.add(get_genotype(f_a, m_a))
            
            if self.possible_genotypes[child] != self.possible_genotypes[child].intersection(possible_child_genotypes):
                self.possible_genotypes[child] = self.possible_genotypes[child].intersection(possible_child_genotypes)
                changed = True
        
        return changed

    def _propagate_up(self):
        """Rule out parent genotypes based on child genotypes."""
        changed = False
        for child in self.individuals:
            parents = self.pedigree.get_parents(child)
            if not parents:
                continue

            father, mother = parents
            
            # Check father
            original_father_options = set(self.possible_genotypes[father])
            for f_g in list(self.possible_genotypes[father]):
                is_possible = False
                # Is there at least one mother genotype that makes this father genotype possible?
                for m_g in self.possible_genotypes[mother]:
                    offspring_genotypes = {get_genotype(f_a, m_a) for f_a in get_alleles(f_g) for m_a in get_alleles(m_g)}
                    if offspring_genotypes.intersection(self.possible_genotypes[child]):
                        is_possible = True
                        break
                if not is_possible:
                    self.possible_genotypes[father].discard(f_g)

            if original_father_options != self.possible_genotypes[father]:
                changed = True

            # Check mother
            original_mother_options = set(self.possible_genotypes[mother])
            for m_g in list(self.possible_genotypes[mother]):
                is_possible = False
                for f_g in self.possible_genotypes[father]:
                    offspring_genotypes = {get_genotype(f_a, m_a) for f_a in get_alleles(f_g) for m_a in get_alleles(m_g)}
                    if offspring_genotypes.intersection(self.possible_genotypes[child]):
                        is_possible = True
                        break
                if not is_possible:
                    self.possible_genotypes[mother].discard(m_g)
            
            if original_mother_options != self.possible_genotypes[mother]:
                changed = True

        return changed
