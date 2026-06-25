import polars as pl

from uplift.mlops.evaluator import *
from uplift.mlops.metric import *
from uplift.mlops.models import ModelFactory
from uplift.mlops.results import TrialResults
from uplift.mlops.search_space import *


class Trial:
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
        results_aggregate_list = []
        results_per_split_list = []
        hp_list = []
        n = -1
        for hp in self.hp_space:
            n += 1
            # make new model
            metrics = []
            print(
                "running trial for model " + str(self.config.model) + " with " + str(hp)
            )
            for split_num, train_data, valid_data in pipeline.build_splits(data):
                split_result = {"split": split_num}
                print(" ==== split num:" + str(split_num) + " ====")
                # instantiate and train model
                model = self.model_class(**hp["parameters"])
                model.fit(train_data)
                # build evaluation data
                eval_data = self.evaluator(model, valid_data)
                # compute metrics and append to results
                metrics.append(split_result | self.metric_report.eval(eval_data))
            metric_per_split = pl.DataFrame(metrics)
            metric_df = (
                pl.DataFrame(metrics)
                .drop("split")
                .describe()
                .filter(pl.col("statistic").is_in(["mean", "std", "min", "max"]))
                .with_columns(pl.lit(n).alias("idx"))
            )
            results_per_split_list.append(metric_per_split)
            results_aggregate_list.append(metric_df)
            hp_list.append(hp)
        result = pl.concat(results_aggregate_list).sort(["idx", "statistic"])
        result_splits = pl.concat(results_per_split_list)
        out = TrialResults(
            self.config.name, self.config, result_splits, result, hp_list
        )
        return out
