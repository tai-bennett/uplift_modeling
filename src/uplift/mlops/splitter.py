from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import StratifiedKFold as SKStratifiedKFold

from uplift.mlops.utils import ArtifactStore


class BaseSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_splits(self):
        pass


class DetectSplitter(BaseSplitter):
    def __init__(self, indices):
        super().__init__()
        # params gives name of split colum
        self.indices = indices
        # splitter needs to know the sampling method
        self._generate_splits()

    def get_splits(self, data):
        """
        returns a dictionary with the indices for each fold
        """
        pass

    def _generate_splits(self):
        pass


class StratifiedKFold(BaseSplitter):
    def __init__(self, num_folds):
        super().__init__()
        self.inputs = {"splitter": "stratified_k_fold", "num_folds": num_folds}
        # params gives name of split colum
        self.num_folds = num_folds
        self.skf = SKStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    def get_splits(self, data):
        store = ArtifactStore()
        # look for stored indices
        result = store.get(data.get_hash(), self.inputs, artifact_codec="fold_indices")
        # if no stored indices, then make them
        if result is None:
            result = self._generate_splits(data)
            store.save(
                data.get_hash(), self.inputs, result, artifact_codec="fold_indices"
            )
        else:
            print(
                "==================== loading splits: stratified_k_fold ============================"
            )
        return result

    def _generate_splits(self, data):
        print(
            "==================== generating splits: stratified_k_fold ============================"
        )
        indices = {}
        labels = data.get_column(data.metadata["target_name"])
        dummy = np.zeros(len(data), dtype=np.int32)
        for fold_idx, (train_idx, test_idx) in enumerate(self.skf.split(dummy, labels)):
            indices[fold_idx] = {"train": train_idx, "test": test_idx}
        return indices


class SplitterFactory:
    def create(self, name):
        if name == "stratified_k_fold":
            return StratifiedKFold
        elif name == "detect_splitter":
            return DetectSplitter
        else:
            raise ValueError(f"Unknown splitter type {name}")
