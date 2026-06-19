from uplift.mlops.models import ModelFactory
from uplift.mlops.search_space import *
from uplift.mlops.metric import *
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
                # evaluate on validation set
                y = valid_data.get_target()
                y_hat = model.eval(valid_data.get_features())
                # compute metrics and append to results
                metrics.append(split_result | self.metric_report.eval(y, y_hat))

            metric_df = pl.DataFrame(metrics)



                

    def _make_hp_space(self):
        # validate the hyperparameter config
        config = SearchSpaceConfig.model_validate({'hyperparameters': self.config.parameters})

        # turn each spec into its corresponding generator
        generators = {}
        for k, v, in config.hyperparameters.items():
            generators[k] = GeneratorFactory().create(v)
        # change this later to a factor so other search space algo can be used
        hp_space = GridSearchSpace(generators)
        return hp_space

