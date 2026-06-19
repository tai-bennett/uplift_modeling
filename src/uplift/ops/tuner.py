from abc import ABC, abstractmethod
from easydict import EasyDict as edict


class BaseTuner(ABC):
    def __init__(self, params):
        self.params = edict(params)

    def tune(self, data, metadata, model_name):
        self.model = MODEL_REGISTRY[model_name]
