"""
================================================================================
TITLE: pipeline.py
AUTHOR: Duncan Bennett
DESCRIPTION: The Pipeline object is a major piece of the OptunaExperiment class
and handles all data related preprocessing. In contrast to preprocess outside
the OptunaExperiment object, the pipeline preprocessing is dependent on hyper-
parameters and is involved in the Optuna parameter suggestion process. E.g.
we may want to tune the model selection according to undersampling ratio.
================================================================================
"""
from abc import ABC, abstractmethod

import numpy as np

from .sampler import SamplerFactory
from .splitter import SplitterFactory


class BasePipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build(self):
        pass


class PipelineProto(BasePipeline):
    def __init__(self, params):
        super().__init__()
        self.components = []
        for k, v, in params.items():
            self.components.append(ComponentFactory().create(k, v))

    def build(self, data, mode='tune'):
        if mode == 'tune':
            indices = self._init_indices(data)
            for comp in self.components:
                indices = comp.get_indices(indices, data)
            # indices = self.splitter.get_splits(data)
            # if "sampler" in self.params:
            #     indices = self.sampler.get_splits(indices, data)
            self.indices = indices
        if mode == 'train':
            indices = self._init_indices(data)
            indices = {'train': {0: {'train': indices['train']}}}
            self.indices = indices


    def build_splits(self, data):
        for fold, split in self.indices['train'].items():
            train_data = data[split["train"]]
            if 'test' in split.keys():
                test_data = data[split["test"]]
            else:
                test_data = None
            yield fold, train_data, test_data

    def _init_indices(self, data):
        n = len(data)
        nums = np.arange(0, n, 1, dtype=np.int_)
        return {'train': nums}

class PipelineFactory:
    def create(self, name):
        if name == "proto":
            return PipelineProto
        else:
            raise ValueError(f"Unknown pipeline type {name}")

class ComponentFactory:
    def create(self, name, config):
        if name == 'splitter':
            return SplitterFactory().create(config['name'])(**config['parameters'])
        if name == 'sampler':
            return SamplerFactory().create(config['name'])(**config['parameters'])
        if name == 'indexer':
            return IndexerFactory().create(config['name'])(**config['parameters'])
        raise ValueError(f"Unknown component type {name}")

