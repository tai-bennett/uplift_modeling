import pdb
import numpy as np
from abc import ABC, abstractmethod
METHODS = ['all', 'none', 'positive_uplift', 'positive_profit', 'topkpercent']


class PolicySelection:
    def __init__(self, metadata, methods=METHODS):
        self.methods = methods
        self.metadata = metadata

    def fit(self, model, data):
        treatment = data[self.metadata['treatment_name']].to_numpy()
        conversion = data[self.metadata['target_name']].to_numpy()
        X = data[self.metadata['feature_names']].to_pandas()
        R = data[self.metadata['revenue_name']].to_numpy()
        C = data[self.metadata['cost_name']].to_numpy()
        scores = model.scores(X)
        # outcome is observed profit
        outcome = treatment * conversion * (R - C) + treatment * (1 - conversion) * (- C) + (1 - treatment) * conversion * R
        results = {}
        for method in self.methods:
            policy = PolicyFactory().create(method)
            mask = policy().predict(scores, treatment, outcome)
            v = ipw(mask, treatment, outcome)
            results[method] = v
        # get max method
        best_method = max(results, key=results.get)
        model.set_policy(PolicyFactory().create(best_method)())
        return model


class PolicyFactory:
    def create(self, name):
        if name == 'all':
            return PolicyAll
        if name == 'none':
            return PolicyNone
        if name == 'positive_uplift':
            return PolicyPositiveUplift
        if name == 'positive_profit':
            return PolicyPositiveProfit
        if name == 'topkpercent':
            return PolicyTopPercent


class PolicyBase(ABC):
    def __init__(self):
        pass


class PolicyAll:
    def __init__(self):
        pass

    def predict(self, scores, treatment, outcome):
        mask = self._make_mask(scores)
        return mask

    def _make_mask(self, scores):
        n = len(scores)
        mask = np.ones(n)
        return mask


class PolicyNone:
    def __init__(self):
        pass

    def run(self):
        pass

    def predict(self, scores, treatment, outcome):
        mask = self._make_mask(scores)
        return mask

    def _make_mask(self, scores):
        n = len(scores)
        mask = np.zeros(n)
        return mask

class PolicyPositiveUplift:
    def __init__(self):
        pass

    def predict(self, scores, treatment, outcome):
        mask = self._make_mask(scores)
        return mask

    def _make_mask(self, scores):
        mask = np.where(scores > 0, 0, 1)
        return mask

class PolicyPositiveProfit:
    def __init__(self):
        pass

    def predict(self, scores, treatment, outcome):
        mask = self._make_mask(scores, outcome)
        return mask

    def _make_mask(self, scores, outcome):
        mask = np.where((scores * outcome) > 0, 0, 1)
        return mask

class PolicyTopPercent:
    def __init__(self, k=20):
        self.k = k
        if k > 1:
            self.k = k/100

    def predict(self, scores, treatment, outcome):
        mask = self._make_mask(scores, outcome)
        return mask

    def _make_mask(self, scores, outcome):
        n = len(scores)
        m = int(np.ceil(n * self.k))
        sorted_idx = np.argsort(scores)[::-1]
        top_idx = sorted_idx[:m]
        mask = np.zeros(n)
        mask[top_idx] = 1
        return mask

# ========================== helper functions ========================
def ipw(policy, treatment, outcome):
    n = len(treatment)
    one = np.ones(n)
    # compute propensity
    e = sum(treatment)/len(treatment)

    v = (policy * treatment * outcome) / e + (one - policy) * (one - treatment) * outcome / (one - e)
    return (1/n) * sum(v)

