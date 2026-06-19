from abc import ABC, abstractmethod
from typing import List

class Metric(ABC):
    @abstractmethod
    def eval(self, y, y_hat):
        pass

class MetricReport():
    def __init__(self, metric_names):
        self.metrics = {}
        for name in metric_names:
            self.metrics[name] = MetricFactory().create(name)()

    def eval(self, y, y_hat):
        if len(y) != len(y_hat):
            raise ValueError(f"len({y}) is not equal to len({y_hat})")
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.eval(y, y_hat)
        return results

    def __iter__(self):
        pass

class QiniMetric(Metric):
    def eval(y, y_hat):
        pass
            
class MSE(Metric):
    def eval(y, y_hat):
        pass

class Precision(Metric):
    def eval(y, y_hat):
        pass

class Recall(Metric):
    def eval(y, y_hat):
        pass

class F1Score(Metric):
    def eval(y, y_hat):
        pass

class MetricFactory():
    def create(self, name):
        if name == 'qini':
            return QiniMetric
        if name == 'mse':
            return MSE
        raise ValueError(f"Unknown metric {name}")
