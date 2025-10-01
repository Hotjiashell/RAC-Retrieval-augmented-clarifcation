import numpy as np



class BaseScoreEvaluator:
    fields = []

    def __init__(self, name):
        self.name = name

    def setup(self):
        pass

    def score(self):
        pass

    def compute_mean(self, scores):
        return {k: np.mean(scores[k]) for k in self.fields}

