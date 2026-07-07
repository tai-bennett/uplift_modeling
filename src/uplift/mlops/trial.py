"""
================================================================================
TITLE: trial.py
AUTHOR: Duncan Bennett
DESCRIPTION: The SingleTrial object faciliates the evaluation of a model under
a fixed set of hyperparameters. It is meant to be used in tandem with the
OptunaExperiment object and is complimentary to the Optuna Trial object
================================================================================
"""

import polars as pl

from .evaluator import Evaluator
from .metric import MetricReport
from .models import ModelFactory


class SingleTrial:
    def __init__(self, study_config, general_config):
        # set configs
        self.study_config = study_config
        self.general_config = general_config

        # call factories
        self.model_class = ModelFactory().create(self.study_config['model'])
        self.model = None
        # self.pipeline_class = PipelineFactory().create(pipeline_config['type'])

        # metric reporter
        self.metric_report = MetricReport(self.general_config['metrics'])
        self.evaluator = Evaluator(self.metric_report.required_inputs)

    def run(self, data, pipeline):
        out = []
        for split_num, train_data, valid_data in pipeline.build_splits(data):
            # instantiate and train model
            self.model = self.model_class(**self.study_config["parameters"])
            self.model.fit(train_data)
            # build evaluation data
            if valid_data is not None:
                eval_data = self.evaluator(self.model, valid_data)
                results = self.metric_report.eval(eval_data)
                results['split'] = split_num
                out.append(results)
            else:
                eval_data = self.evaluator(self.model, train_data)
                results = self.metric_report.eval(eval_data)
                results['split'] = split_num
                out.append(results)
        out = pl.from_dicts(out)
        metric_name = self.general_config['metrics'][0]
        value = out.mean()[metric_name].item(0)
        # mlflow.log_metric(metric_name, value)
        return value

    def train(self, data, pipeline):
        out = []
        for split_num, train_data, valid_data in pipeline.build_splits(data):
            # instantiate and train model
            model = self.model_class(**self.study_config["parameters"])
            model.fit(train_data)
            # build evaluation data
            eval_data = self.evaluator(model, valid_data)
            results = self.metric_report.eval(eval_data)
            results['split'] = split_num
            out.append(results)
        out = pl.from_dicts(out)
        metric_name = self.general_config['metrics'][0]
        value = out.mean()[metric_name].item(0)
        # mlflow.log_metric(metric_name, value)
        return value

