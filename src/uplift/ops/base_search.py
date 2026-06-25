from abc import ABC, abstractmethod


class BaseSearch(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def range(self, n):
        pass
