from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import StratifiedKFold as SKStratifiedKFold
from uplift.mlops.utils import ArtifactStore


class BaseSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_indices(self):
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
    def __init__(self, num_splits):
        super().__init__()
        self.inputs = {"splitter": "stratified_k_fold", "num_splits": num_splits}
        # params gives name of split colum
        self.num_splits = num_splits
        self.skf = SKStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

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
        indices_new = {}
        data.data = data.data.with_row_index("row_idx")
        df = data[indices['train']]
        labels = df.get_column(data.metadata["target_name"])
        dummy = df.get_column('row_idx')
        for fold_idx, (train_idx, test_idx) in enumerate(self.skf.split(dummy, labels)):
            indices_new[fold_idx] = {"train": dummy[train_idx].flatten(), "test": dummy[test_idx].flatten()}
        data.data = data.data.drop("row_idx")
        indices['train'] = indices_new
        return indices


class SplitterFactory:
    def create(self, name):
        if name == "stratified_k_fold":
            return StratifiedKFold
        elif name == "detect_splitter":
            return DetectSplitter
        else:
            raise ValueError(f"Unknown splitter type {name}")
