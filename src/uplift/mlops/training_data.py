import hashlib
from abc import ABC, abstractmethod

import polars as pl


class TrainingData:
    def __init__(self, data, fold_data):
        self.data = data
        self.fold_data = fold_data

    def get_fold(self, n):
        """
        returns dictionary with train data and valid data
        """
        out = {}
        out["train"] = self.data[self.fold_data[n]["train"]]
        out["valid"] = self.data[self.fold_data[n]["valid"]]
        return out


class Data(ABC):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_column(self):
        pass


class PolarsData(Data):
    def __init__(self, data, metadata):
        super().__init__(data, metadata)

    def get_features(self, as_type=None):
        cols = pl.col(self.metadata["feature_names"])
        out = self.data.select(cols)
        out = self._convert_output(out, as_type=as_type)
        return out

    def get_target(self, as_type=None):
        cols = pl.col(self.metadata["target_name"])
        out = self.data.select(cols)
        out = self._convert_output(out, as_type=as_type).flatten()
        return out

    def __getitem__(self, indices):
        subset = self.data[indices]
        return type(self)(subset, self.metadata)

    def __len__(self):
        return self.data.height

    def get_hash(self):
        return hashlib.sha256(self.data.hash_rows().to_numpy().tobytes()).hexdigest()

    def get_column(self, col_name):
        return self.data[col_name].to_numpy()

    def value_counts(self, col_name):
        return self.data[col_name].value_counts().to_dicts()

    def _convert_output(self, out, as_type=None):
        if as_type is None:
            return out
        if as_type == "numpy":
            return out.to_numpy()


class DataFactory:
    def create(self, data, meta):
        # load metadata (always as yml)
        if config.type == "polars":
            pass
            # load data
