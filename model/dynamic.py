import numpy as np

class Dynamic:
    def __init__(self, k):
        self.similarity_measure = lambda x1, x2: np.linalg.norm(x1 - x2)
        self.k = k

    def find_similar_samples(self, val_X, xt):
        similarities = []
        for x in xt:
            similarity = self.similarity_measure(val_X, x)
            similarities.append(similarity)
        similarities = np.array(similarities)
        indices = np.argsort(similarities)[:self.k]
        subset = xt[indices]
        return subset, indices
