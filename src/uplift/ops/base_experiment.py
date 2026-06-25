from abc import ABC, abstractmethod

from easydict import EasyDict as edict


class BaseExperiment(ABC):
    def __init__(self, params):
        self.params = edict(params)

    @abstractmethod
    def run(self):
        return True
