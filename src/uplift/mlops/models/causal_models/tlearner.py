# Main imports
import pdb
from .base_causal_model import CausalModel
from .submodel_factory import SubmodelFactory
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

# Helper imports
import numpy as np
from numpy.random import binomial, multivariate_normal, normal, uniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class MyTLearner(CausalModel):
    def __init__(self, model_name, parameters):
        # models = CausalModelFactory.create(model_name)(parameters)
        # models = GradientBoostingRegressor(**parameters)
        models = SubmodelFactory().create(model_name)(**parameters)
        self.model = TLearner(models=models)

    def fit(self, data):
        X, y, T = self._process_train_data(data)
        self.model.fit(y, T, X=X)

    def effect(self, X):
        """
        returns the uplift
        """
        p_treat = self.model.models[1].predict_proba(X)[:, 1]
        p_control = self.model.models[0].predict_proba(X)[:, 1]
        return p_treat - p_control

    def eval(self, X):
        """
        returns the predict class
        Note: this might compute differences between f_treat.eval(X) - f_control.eval(X)
        but f.eval for certain submodels behave differently, could be predicted classed or class probability?
        """
        return self.model.effect(X)


