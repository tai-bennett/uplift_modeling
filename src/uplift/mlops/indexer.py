from abc import ABC, abstractmethod

from uplift.mlops.training_data import PolarsData
from uplift.mlops.utils import ArtifactStore

from .sampler import Undersampler


class BaseIndexer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class Indexer(BaseIndexer):
    def __init__(self, inputs: dict(str, float)):
        super().__init__()
        self.inputs = inputs
        self._check_inputs

    def get_indices(self, indices, data):
        store = ArtifactStore()
        # look for stored indices
        result = store.get(data.get_hash(), self.inputs, artifact_codec="fold_indices")
        # if no stored indices, then make them
        if result is None:
            result = self._generate_indices(data)
            store.save(
                data.get_hash(), self.inputs, result, artifact_codec="fold_indices"
            )
        return result

    def _generate_indices(self, data):
        if type(data) != PolarsData:
            raise NotImplementedError(
                f"Undersampler not implemented for data type {type(data)}"
            )

    def _check_inputs(self, inputs):
        total = 0
        for k, v in inputs:
            total += v
        if total != 1.0:
            raise ValueError(f"Input for indexer needs value that sum to 1. Got {total} from inputs {inputs}")


class SamplerFactory:
    def create(self, name):
        if name == "undersampler":
            return Undersampler
        else:
            raise ValueError(f"Unknown sampler type {name}")
