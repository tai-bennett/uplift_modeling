from .tlearner import MyTLearner


class ModelFactory():
    def create(self, name):
        if name == 'tlearner':
            return MyTLearner
        raise ValueError(f"Unknown causal model of type {name}.")
