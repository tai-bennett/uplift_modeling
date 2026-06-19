from easydict import EasyDict as edict
from uplift.config.registry import register_model
from abc import ABC, abstractmethod

@register_model('base')
class BaseModel(ABC):
    def __init__(self, params):
        self.params = edict(params)

    def tune(self, df, metadata):
        # tuner = TUNER_REGISTRY[params.tuner.name]
        pass

    def fit(self, df, metadata):
        """
        This method should,
        - Filter the data based on the sampling scheme
        - set X, T, y to their appropriate variables
        - use cross validation for parameter selection
        - save the model self.model as the best cross validation selected

        """
        print("Code not implemented for method 'fit'")

    def infer(self, X):
        print("Code not implemented for method 'infer'")

    def get_info(self):
        print("Code not implemented for method 'get_info'")
