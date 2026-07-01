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
from uplift.mlops.results import *
from uplift.mlops.search_space import *
from uplift.mlops.spec import *
from uplift.mlops.training_data import *
from uplift.mlops.trial import *
from uplift.mlops.trial import SingleTrial, Trial
from uplift.mlops.utils import ArtifactStore


class Experiment:
    def __init__(self, config):
        self.config = edict(config)
        self.store = ArtifactStore(get_paths()["experiment_results"])
        self.store.clear_root()
        # create search space for pipeline hp
        # self.pipeline_hp = self._make_hp_search_space(self.config.pipelines.parameters)

    def run(self):
        data = self._make_dataset(self.config.data)
        trial_results = []
        hp_list = []
        for pipeline_config in self.config.pipelines:
            print("Running pipeline: " + pipeline_config.name)
            # make pipeline class from factory
            pipeline_class = PipelineFactory().create(pipeline_config.type)
            # make hp space for pipeline
            pipeline_hp_space = self._make_hp_search_space(pipeline_config.components)
            # for hp in hpspace
            for hp in pipeline_hp_space:
                hp_list.append(hp)
                print("Pipeline parameters: " + str(hp))
                # make pipeline from factory
                pipeline = pipeline_class(hp)
                pipeline.build(data)
                for trial_config in self.config.trials:
                    trial = Trial(self.config.trial_config, trial_config)
                    trial_results.append(trial.run(data, pipeline))
        final_result = ExperimentResults(
            self.config.experiment.name, self.config, trial_results, hp_list
        )
        self.store.save_direct(
            self.config.experiment.name,
            final_result,
            artifact_codec="experiment_results",
        )

    def _make_dataset(self, config):
        adapter = TypeAdapter(DataSourceSpec)
        spec = adapter.validate_python(config)
        data_source_class = DataSourceFactory().create(spec)
        config.pop("type")
        data_source = data_source_class(**config)
        return data_source.load()

    def _make_hp_search_space(self, config):
        # validate the hyperparameter config
        adapter = TypeAdapter(ParameterSpec)
        # for each subcomponent of pipeline
        # generators = config
        generators = {}
        for component_name, component_config in config.items():
            # for each parameter of subcomponent
            subgenerators = {}
            for param_name, param_config in component_config.parameters.items():
                spec = adapter.validate_python(param_config)
                subgenerators[param_name] = GeneratorFactory().create(spec)
            generators[component_name] = GridSearchSpace(
                subgenerators, name=component_config.name
            )
        hp_space = GridSearchSpace(generators)
        return hp_space

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
        with mlflow.start_run(run_name=f"studies_{self.experiment_name}"):
            for study_config in self.config.studies:
                self.current_config = {'study': study_config, 'pipeline': self.config.pipeline}
                study_winner = self.run_study()
                winners.append(study_winner)
            # get winner from study winners
            winner = max(winners, key=lambda result: result.metric)
            mlflow.log_param("winner_config", winner.config)

            # train the best config on the whole dataset
            model_info = self.train_winner(winner)
        return model_info

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
