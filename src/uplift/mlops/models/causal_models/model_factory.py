from .tlearner import MyTLearner
from .drlearner import MyDRLearner


class ModelFactory():
    def create(self, name):
        if name == 'tlearner':
            return MyTLearner
        if name == 'drlearner':
            return MyDRLearner
        raise ValueError(f"Unknown causal model of type {name}.")
