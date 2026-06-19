import pdb
from .base_experiment import BaseExperiment
from uplift.models import *
from uplift.config.registry import MODEL_REGISTRY, SAMPLER_REGISTRY

class Experiment(BaseExperiment):
    def __init__(self, params):
        super().__init__(params)
        self.model_class = MODEL_REGISTRY[params.architecture]
        # self.search_space = SearchSpace()
        # self.tune_algo = SEARCH_ALGO_REGISTRY[params.tune_algo.name](params.tune_algo)
        

    def run(self, data, metadata):
        for hp in ???:
            model = self.model_class(hp)
            results = model.tune(data, metadata)

            


if __name__ == "__main__":
    exp = Experiment()
    exp.run()
    print("Experiment.py class file compiles.")
