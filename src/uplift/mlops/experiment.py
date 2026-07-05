import pdb
import pandas as pd
import mlflow
import optuna
from easydict import EasyDict as edict
from pydantic import TypeAdapter

from uplift.config.loaders import get_paths
from uplift.mlops.data_source import *
from uplift.mlops.optuna_utils import OptunaHPBuilder, optuna_to_config
from uplift.mlops.pipeline import PipelineFactory
from uplift.mlops.results import TuningResults, ExperimentResults
from uplift.mlops.search_space import *
from uplift.mlops.spec import *
from uplift.mlops.training_data import *
from uplift.mlops.trial import *
from uplift.mlops.trial import SingleTrial, Trial
from uplift.mlops.utils import ArtifactStore


class OptunaExperiment:
    def __init__(self, config):
        self.config = edict(config)
        self.current_study_config = None
        self.data = self._make_dataset(self.config.data)
        self.experiment_name = self.config.experiment.name
        tracking_db = get_paths()['root'] / 'mlflow.db'
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
        mlflow.set_experiment(self.experiment_name)
        self.check_mlflow_connection()

    def run(self):
        # run tuning selecting winners from each study
        winners = []
        with mlflow.start_run(run_name=f"studies_{self.experiment_name}") as run:
            for study_config in self.config.studies:
                self.current_config = {'study': study_config, 'pipeline': self.config.pipeline}
                study_winner = self.run_study()
                winners.append(study_winner)
            # get winner from study winners
            winner = max(winners, key=lambda result: result.metric)
            mlflow.log_param("winner_config", winner.config)

            # train the best config on the whole dataset
            model_info = self.train_winner(winner)
            results = ExperimentResults(
                run.info.run_id,
                model_info.model_uri,
                winner.metric,
                winner.config
                )
        return results

    def train_winner(self, winner):
        with mlflow.start_run(run_name="final_training_run", nested=True):
            hp = winner.config
            # make pipeline
            pipeline_class = PipelineFactory().create(hp['pipeline']['type'])
            pipeline = pipeline_class(hp['pipeline']['components'])
            pipeline.build(self.data, mode='train')
            # make single trial
            st = SingleTrial(hp['study'], self.config.trial_config)
            eval_metric = st.run(self.data, pipeline)
            # log metrics and parameters
            input_ex = pd.DataFrame(
                self.data[0:10].get_features(as_type='numpy'),
                columns=self.data.metadata['feature_names']
            )
            model_info = mlflow.pyfunc.log_model(
                name=f"model_{self.experiment_name}_winner",
                python_model=st.model,
                input_example=input_ex,
                registered_model_name=f"self.model_{self.experiment_name}_winner"
            )
            return model_info


    def run_study(self):
        study = optuna.create_study(
            study_name=self.current_config['study']['name'],
            direction='maximize'
        )
        study.optimize(self.mlflow_objective, n_trials=3)
        best_params = study.best_trial.params

        # best_model_config = optuna_to_config(self.current_study_config.copy(), best_params)
        # best_pipeline_config = optuna_to_config(self.config.pipeline.copy(), best_params)
        best_config = optuna_to_config(self.current_config.copy(), best_params)
        out = TuningResults(
            study.best_trial,
            best_config,
            study.best_trial.value
            )
        # mlflow.log_params(study.best_trial.params)
        return out

    def mlflow_objective(self, trial):
        with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
            score = self.objective(trial)
            mlflow.log_metric(f"score_{self.config.trial_config.metrics[0]}", score)
            return score


    def objective(self, trial):
        # generate pipeline parameters via optuna
        hp_builder = OptunaHPBuilder(self.current_config)
        hp = hp_builder.get_parameters(trial)
        # hp_pipeline = OptunaHPBuilder(self.config.pipeline)
        # pipeline_params = hp_pipeline.get_parameters(trial)
        pipeline_class = PipelineFactory().create(hp['pipeline']['type'])
        pipeline = pipeline_class(hp['pipeline']['components'])
        pipeline.build(self.data)
        # generate model config via optuna
        # hp_study = OptunaHPBuilder(self.current_study_config)
        # study_params = hp_study.get_parameters(trial)
        # mlflow log params
        # mlflow.log_params({"pipeline": pipeline_params, "study": study_params})
        mlflow.log_param(f"trial_{trial.number}_hp", hp)
        # make trial and train
        st = SingleTrial(hp['study'], self.config.trial_config)
        eval_metric = st.run(self.data, pipeline)
        return eval_metric

    def _make_dataset(self, config):
        config_copy = config.copy()
        adapter = TypeAdapter(DataSourceSpec)
        spec = adapter.validate_python(config_copy)
        data_source_class = DataSourceFactory().create(spec)
        config_copy.pop("type")
        data_source = data_source_class(**config_copy)
        return data_source.load()


    def check_mlflow_connection(self):
        # Print connection information
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Active Experiment: {mlflow.get_experiment_by_name(self.experiment_name)}")
