import pytest
import numpy as np
from uplift.mlops.metric import *

def test_metric_report():
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([1, 1.3, 2.9, 3, 3.9, 5.5])
    data = {'y_true': y_true, 'y_pred': y_pred}
    data = EvaluationData(**data)

    metric_names = ["mse", "mae"]
    reporter = MetricReport(metric_names)
    results = reporter.eval(data)

