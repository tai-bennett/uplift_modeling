"""
================================================================================
TITLE: metric.py

AUTHOR: Duncan Bennett

DESCRIPTION: Metrics to be evaluated on tuning and training data. Users primary
interact with the EvaluationData and MetricReport objects.
================================================================================
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklift.metrics import qini_auc_score, uplift_auc_score


@dataclass
class EvaluationData:
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None

    uplift: np.ndarray | None = None
    treatment: np.ndarray | None = None
    conversion: np.ndarray | None = None


class Metric(ABC):
    @abstractmethod
    def eval(self, y, y_hat):
        pass


class MetricReport:
    def __init__(self, metric_names):
        self.metrics = {}
        for name in metric_names:
            self.metrics[name] = MetricFactory().create(name)()
        self.required_inputs = set()
        for name, metric in self.metrics.items():
            self.required_inputs = self.required_inputs | metric.required_inputs

    def eval(self, data: EvaluationData):
        results = {}
        for name, metric in self.metrics.items():
            args = self._get_required_inputs(metric, data)
            results[name] = metric.eval(**args)
        return results

    def _get_required_inputs(self, metric, data):
        out = {}
        for name in metric.required_inputs:
            out[name] = getattr(data, name)
        return out


class QiniMetric(Metric):
    def __init__(self):
        self.required_inputs = {"treatment", "conversion", "uplift"}

    def eval(self, uplift, treatment, conversion):
        return qini_auc_score(conversion, uplift, treatment)


class AUUCMetric(Metric):
    def __init__(self):
        self.required_inputs = {"treatment", "conversion", "uplift"}

    def eval(self, uplift, treatment, conversion):
        return uplift_auc_score(conversion, uplift, treatment)


class MSE(Metric):
    def __init__(self):
        self.required_inputs = {"y_true", "y_pred"}

    def eval(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


class MAE(Metric):
    def __init__(self):
        self.required_inputs = {"y_true", "y_pred"}

    def eval(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


class Precision(Metric):
    def eval(self, y, y_hat):
        pass


class Recall(Metric):
    def eval(self, y, y_hat):
        pass


class F1Score(Metric):
    def eval(self, y, y_hat):
        pass


class MetricFactory:
    def create(self, name):
        if name == "qini":
            return QiniMetric
        if name == "auuc":
            return AUUCMetric
        if name == "mse":
            return MSE
        if name == "mae":
            return MAE
        raise ValueError(f"Unknown metric {name}")
