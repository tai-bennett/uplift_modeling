# Main imports
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
        print("Training TLearner model ...")
        #self.model.fit(y, T, X=X)

    def eval(self, X):
        return self.model.effect(X)


