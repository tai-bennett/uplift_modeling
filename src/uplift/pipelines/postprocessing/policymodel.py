import pandas as pd
import numpy as np
import mlflow


class PolicyModel:
    def __init__(self, uplift_model):
        self.uplift_model = uplift_model
        self.calibrator = lambda x: x
        self.policy = None

    def predict(self,
                model_input: pd.DataFrame
                # model_input: pd.DataFrame,
                # context: mlflow.pyfunc.PythonModelContext = None,
                # params: dict | None = None
                ) -> np.ndarray:
        return self.__call__(model_input)
        
    def __call__(self,
                 model_input: pd.DataFrame
                 ) -> np.ndarray:
                 # ) -> np.ndarray:
                 # model_input: pd.DataFrame,
                 # context: mlflow.pyfunc.PythonModelContext = None,
                 # params: dict | None = None
                 # ) -> np.ndarray:
        if self.policy is None:
            raise ValueError("Policy is note set. Set policy or use PolicyModel.scores method to get (calibrated) uplift scores.")
        features, treatment, outcome = self._parse_input(model_input.copy())
        scores = self.scores(features)
        mask = self.policy.predict(scores, treatment, outcome)
        return mask

    def scores(self, features):
        uplift = self.uplift_model.predict(features)
        scores = self.calibrator.predict(uplift)
        return scores


    def set_calibrator(self, calibrator):
        self.calibrator = calibrator

    def set_policy(self, policy):
        self.policy = policy

    def _parse_input(self, model_input):
        treatment = model_input.pop('treatment').to_numpy()
        outcome = model_input.pop('outcome').to_numpy()
        features = model_input
        return features, treatment, outcome
