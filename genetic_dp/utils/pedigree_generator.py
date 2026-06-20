import random
from ..models.pedigree import Pedigree

def generate_random_pedigree(num_generations, mean_offspring=2):
    pedigree = Pedigree()
    if num_generations < 1:
        raise ValueError("Cannot generate a pedigree with fewer than 1 generation.")

    # Generation 0: Founders
    founders = ["F1", "F2"]
    pedigree.add_individual(founders[0])
    pedigree.add_individual(founders[1])

    last_generation = list(founders)
    all_individuals = list(founders)

    for gen in range(1, num_generations):
        next_generation = []
        # Pair up individuals from the last generation to produce offspring
        random.shuffle(last_generation)
        for i in range(0, len(last_generation), 2):
            if i + 1 < len(last_generation):
                parent1 = last_generation[i]
                parent2 = last_generation[i+1]
                num_offspring = random.randint(1, max(1, int(mean_offspring * 2)))
                for j in range(num_offspring):
                    child_name = f"G{gen}_C{len(next_generation) + 1}"
                    pedigree.add_individual(child_name, parents=(parent1, parent2))
                    next_generation.append(child_name)
                    all_individuals.append(child_name)
        last_generation = next_generation

    return pedigree

def generate_deterministic_pedigree(relationships):
    """
    Generates a pedigree based on a list of relationships.

    Args:
        relationships (list): A list of tuples, where each tuple represents
                              (child_name, parent1_name, parent2_name).
                              Founders are implicitly defined as individuals who are parents
                              but not children in any relationship.

    Returns:
        Pedigree: The generated pedigree object.
    """
    pedigree = Pedigree()
    all_individuals = set()
    children = set()

    # First pass: add all individuals and establish parent-child relationships
    for child, parent1, parent2 in relationships:
        pedigree.add_individual(child, parents=(parent1, parent2))
        all_individuals.add(child)
        all_individuals.add(parent1)
        all_individuals.add(parent2)
        children.add(child)

    # Add founders (individuals who are parents but not children)
    for ind in all_individuals:
        if ind not in children:
            pedigree.add_individual(ind) # Add as founder if not already added as a child

    return pedigree

def reconstruct_pedigree_from_edges(edges):
    """
    Reconstructs a Pedigree object from a list of edges.

    Args:
        edges (list): A list of lists, where each inner list is [parent, child].

    Returns:
        Pedigree: The reconstructed pedigree object.
    """
    reconstructed_pedigree = Pedigree()
    all_individuals = set()
    child_parent_map = {}

    for parent, child in edges:
        all_individuals.add(parent)
        all_individuals.add(child)
        if child not in child_parent_map:
            child_parent_map[child] = []
        child_parent_map[child].append(parent)

    for individual in all_individuals:
        reconstructed_pedigree.add_individual(individual)

    for child, parents in child_parent_map.items():
        if len(parents) == 2:
            reconstructed_pedigree.add_individual(child, parents=tuple(parents))
    
    return reconstructed_pedigree
