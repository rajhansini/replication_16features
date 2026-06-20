import networkx as nx

class Pedigree:
    """A class to represent a pedigree using a directed graph."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_individual(self, name, parents=None):
        """Adds an individual to the pedigree.

        Args:
            name (str): The name of the individual.
            parents (tuple, optional): A tuple containing the names of the parents. 
                                     Defaults to None for founders.
        """
        self.graph.add_node(name)
        if parents and len(parents) == 2:
            self.graph.add_edge(parents[0], name)
            self.graph.add_edge(parents[1], name)

    def get_founders(self):
        """Returns a list of founders in the pedigree."""
        return [n for n, d in self.graph.in_degree() if d == 0]

    def get_offspring(self):
        """Returns a list of all non-founder individuals."""
        return [n for n, d in self.graph.in_degree() if d > 0]

    def get_parents(self, name):
        """Returns the parents of an individual."""
        return list(self.graph.predecessors(name))

    def to_list(self):
        """Returns a list of all individuals in the pedigree."""
        return list(nx.topological_sort(self.graph))
