from .models import *
from .metric import EvaluationData

class Evaluator():
    def __init__(self, value_names):
        self.value_names = value_names

    def __call__(self, model, data):
        out = {}
        for name in self.value_names:
            # compute value called name
            out[name] = self._compute(name, model, data)
        return EvaluationData(**out)

    def _compute(self, name, model, data):
        if name == 'y_true':
            return data.get_target(as_type='numpy')
        if name == 'y_pred':
            return model.eval(data.get_features(as_type='numpy'))
        if name == 'uplift':
            return model.eval(data.get_features(as_type='numpy'))
        if name == 'treatment':
            return data.get_column(data.metadata['treatment_name'])
        if name == 'conversion':
            return data.get_column(data.metadata['effect_name'])
        raise ValueError(f"Unknown value name for Evaluator called {name}.")
