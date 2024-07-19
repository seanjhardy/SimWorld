import numpy as np
from scipy.spatial import KDTree

class VectorMemory:
    def __init__(self, code_dim, vector_dim):
        self.code_dim = code_dim
        self.vector_dim = vector_dim
        self.memory = []
        self.codes = []
        self.tree = None

    def add_memory(self, code, vector):
        if len(code) != self.code_dim or len(vector) != self.vector_dim:
            raise ValueError("Dimension mismatch.")
        self.codes.append(code)
        self.memory.append(vector)
        self.tree = KDTree(np.array(self.codes))  # Rebuild the KDTree with the new code.

    def retrieve_memory(self, code, tolerance=0.05):
        if self.tree is None:
            return []

        indices = self.tree.query(code, tolerance)
        return [self.memory[i] for i in indices]