"""
================================================================================
TITLE: pipeline.py
AUTHOR: Duncan Bennett
DESCRIPTION: The Pipeline object is a major piece of the OptunaExperiment class
================================================================================
"""
from abc import ABC, abstractmethod


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
