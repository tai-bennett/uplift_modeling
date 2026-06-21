from uplift.mlops.models import ModelFactory
from uplift.mlops.search_space import *
from uplift.mlops.metric import *
from uplift.mlops.evaluator import *
import polars as pl

class Trial():
    def __init__(self, general_config, trial_config):
        self.general_config = general_config
        self.config = trial_config
        # create factories, tuner etc for one trial
        self.model_class = ModelFactory().create(self.config.model)
        # self.hp_space = self._make_hp_space()
        self.hp_space = HPBuilder(trial_config)
        self.metric_report = MetricReport(general_config.metrics)
        self.evaluator = Evaluator(self.metric_report.required_inputs)

    def run(self, data, pipeline):
        for hp in self.hp_space:
            # make new model
            metrics = []
            print("running trial for model " + str(self.config.model) + " with " + str(hp))
            for split_num, train_data, valid_data in pipeline.build_splits(data):
                split_result = {'split': split_num}
                print(" ==== split num:" + str(split_num) + " ====")
                # instantiate and train model
                model = self.model_class(**hp['parameters'])
                model.fit(train_data)
                # build evaluation data
                eval_data = self.evaluator(model, valid_data)
                # evaluate on validation set
                # y = valid_data.get_target(as_type='numpy')
                # y_hat = model.eval(valid_data.get_features())
                # compute metrics and append to results
                metrics.append(split_result | self.metric_report.eval(eval_data))

            metric_df = pl.DataFrame(metrics)
