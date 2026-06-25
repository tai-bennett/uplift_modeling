from abc import ABC, abstractmethod

from easydict import EasyDict as edict

from uplift.mlops.sampler import SamplerFactory
from uplift.mlops.splitter import SplitterFactory


class BasePipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build(self):
        pass


class PipelineProto(BasePipeline):
    def __init__(self, params):
        super().__init__()
        self.params = edict(params)
        self.splitter = SplitterFactory().create(self.params.splitter.name)(
            **self.params.splitter.parameters
        )
        if "sampler" in self.params:
            self.sampler = SamplerFactory().create(self.params.sampler.name)(
                **self.params.sampler.parameters
            )
        self.indices = None

    def build(self, data):
        indices = self.splitter.get_splits(data)
        if "sampler" in self.params:
            indices = self.sampler.get_splits(indices, data)
        self.indices = indices

    def build_splits(self, data):
        for fold, split in self.indices.items():
            train_data = data[split["train"]]
            test_data = data[split["test"]]
            yield fold, train_data, test_data


class PipelineFactory:
    def create(self, name):
        if name == "proto":
            return PipelineProto
        else:
            raise ValueError(f"Unknown pipeline type {name}")
