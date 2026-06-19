from abc import ABC, abstractmethod
import pdb
from uplift.mlops.training_data import Data

class CausalModel(ABC):
    def _process_train_data(self, data: Data):
        X = data.get_features()
        y = data.get_target()
        T = data.get_column(data.metadata['treatment_name'])

        return X, y, T
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def eval(self):
        pass



