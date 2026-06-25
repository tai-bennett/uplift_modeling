from abc import ABC, abstractmethod

from uplift.mlops.spec import *


class ParameterGenerator(ABC):
    @abstractmethod
    def __iter__(self):
        pass


class ChoiceGenerator(ParameterGenerator):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values


class IntRangeGenerator(ParameterGenerator):
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step

    def __iter__(self):
        current = self.min

        while current <= self.max:
            yield current
            current += self.step
