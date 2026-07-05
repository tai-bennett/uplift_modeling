from abc import ABC, abstractmethod

from uplift.mlops.training_data import Data


class CausalModel(ABC):
    def __init__(self):
        self.calibration = None
        self.model = None
    def _process_train_data(self, data: Data):
        X = data.get_features(as_type="numpy")
        y = data.get_target(as_type="numpy").flatten()
        T = data.get_column(data.metadata["treatment_name"]).flatten()

        return X, y, T

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def __call__(self, X):
        return X
