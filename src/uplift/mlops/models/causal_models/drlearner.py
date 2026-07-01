# Main imports
# Helper imports
from econml.dr import DRLearner

from .base_causal_model import CausalModel
from .submodel_factory import SubmodelFactory


class MyDRLearner(CausalModel):
    def __init__(
        self,
        model_effect,
        effect_parameters,
        model_propensity,
        propensity_parameters,
        model_final,
        final_parameters,
        parameters,
    ):
        model_effect = SubmodelFactory().create(model_effect)(**effect_parameters)
        model_propensity = SubmodelFactory().create(model_propensity)(
            **propensity_parameters
        )
        model_final = SubmodelFactory().create(model_final)(**final_parameters)
        self.model = DRLearner(
            model_regression=model_effect,
            model_propensity=model_propensity,
            model_final=model_final,
            discrete_outcome=True,
            **parameters,
        )

    def fit(self, data):
        X, y, T = self._process_train_data(data)
        self.model.fit(y, T, X=X)

    def __call__(self, X):
        return self.effect(X)


    def effect(self, X):
        """
        returns the uplift
        """
        return self.model.effect(X)

    def eval(self, X):
        """
        returns the predict class
        Note: this might compute differences between f_treat.eval(X) - f_control.eval(X)
        but f.eval for certain submodels behave differently, could be predicted classed or class probability?
        """
        return self.model.effect(X)
