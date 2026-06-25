import numpy as np

from .base_search import BaseSearch


class RandomSearch(BaseSearch):
    def __init__(self, params):
        self.params = params

    def range(self, n, seed):
        self.rng = np.random.default_rng()
        for _ in range(n):
            yield self.get()

    def get(self):
        out = {}
        for name, param in self.params.get_items():
            out[name] = self._get_random(param)

    def _get_random(self, param):
        if param["type"] == "category":
            return self.rng.choice(param["values"])
        if param["type"] == "continuous":
            if param["distribution"] == "uniform":
                return self.rng.uniform(param["min"], param["max"])
