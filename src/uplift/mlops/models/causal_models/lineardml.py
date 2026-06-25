import warnings

from econml.dml import LinearDML

from .base_causal_model import CausalModel


class LinearDML(CausalModel):
    def __init__(self):
        self.model = LinearDML()
        self.fitted = False

    def fit(self, train_data):
        X, y, T = self._process_train_data(train_data)
        self.model(y, T, X=X)
        self.fitted = True

    def eval(self, X):
        if not self.fitted:
            warnings.warn(
                "This model has not been fit to data, behavior can be unexpected.",
                UserWarning,
            )
        return self.model.const_marginal_effect(X)
