from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from uplift.mlops.training_data import PolarsData
from uplift.mlops.utils import *


class BaseSampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class Undersampler(BaseSampler):
    def __init__(self, ratio, seed):
        super().__init__()
        self.inputs = {"ratio": ratio, "seed": seed, "sampler": "undersampler"}
        self.ratio = ratio
        self.seed = seed
        # self.class_col_name = params.class_col_name
        # self.minority_class_name = params.minority_class_name
        # self.majority_class_name = params.minority_class_name

    def sample(self):
        pass

    def get_indices(self, indices, data):
        store = ArtifactStore()
        # look for stored indices
        result = store.get(data.get_hash(), self.inputs, artifact_codec="fold_indices")
        # if no stored indices, then make them
        if result is None:
            result = self._generate_splits(indices, data)
            store.save(
                data.get_hash(), self.inputs, result['train'], artifact_codec="fold_indices"
            )
            return result
        indices['train'] = result
        return indices

    def _generate_splits(self, indices, data):
        input_indices = indices.copy()
        indices = indices['train']
        if type(data) != PolarsData:
            raise NotImplementedError(
                f"Undersampler not implemented for data type {type(data)}"
            )
        data.data = data.data.with_row_index("row_idx")
        for fold_idx, data_idx in indices.items():
            idx = data_idx["train"]
            current_df = data[idx]
            value_counts = data.value_counts(data.metadata["target_name"])
            # compute major class sample probability
            prob, m, major_class, minor_class = self._compute_sampling_prob(
                value_counts, data.metadata["target_name"]
            )
            # compute subset of indices
            subindices = self._subsample_indices(
                idx, data, major_class, minor_class, count=m
            )
            indices[fold_idx]["train"] = subindices.flatten()
        data.data = data.data.drop("row_idx")
        input_indices['train'] = indices
        return input_indices

    def _compute_sampling_prob(self, value_counts, target_col):
        majority = {target_col: None, "count": -100}
        for dd in value_counts:
            if dd["count"] > majority["count"]:
                majority = dd
        minority = {target_col: None, "count": np.inf}
        for dd in value_counts:
            if dd["count"] < minority["count"]:
                minority = dd
        old_ratio = minority["count"] / majority["count"]
        p = old_ratio / self.ratio
        # m = np.floor(majority['count'] * p)
        m = np.floor(minority["count"] / self.ratio)
        return p, m, majority[target_col], minority[target_col]

    def _subsample_indices(self, idx, data, major, minor, count=None, p=None):
        if p is not None:
            raise NotImplementedError(
                "Subsampling with probability not implemented for Undersampler."
            )
        if count is not None:
            indices_maj = (
                data.data.filter(pl.col(data.metadata["target_name"]) == major)
                .sample(n=count, shuffle=True, seed=100)
                .select("row_idx")
                .to_numpy()
            )
            indices_min = (
                data.data.filter(pl.col(data.metadata["target_name"]) == minor)
                .select("row_idx")
                .to_numpy()
            )
            indices = np.concatenate([indices_maj, indices_min], axis=0)
            # pl.col("row_idx").is_in(idx)
            return indices
        else:
            raise ValueError(
                "count or p needs to be not None for method _subsample_indices"
            )


class SamplerFactory:
    def create(self, name):
        if name == "undersampler":
            return Undersampler
        else:
            raise ValueError(f"Unknown sampler type {name}")
