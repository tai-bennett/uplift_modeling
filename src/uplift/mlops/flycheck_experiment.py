import optuna
from easydict import EasyDict as edict
from pydantic import TypeAdapter

from uplift.mlops import utils
from uplift.mlops.data_source import *
from uplift.mlops.pipeline import PipelineFactory
from uplift.mlops.results import *
from uplift.mlops.search_space import *
from uplift.mlops.spec import *
from uplift.mlops.training_data import *
from uplift.mlops.trial import *
from uplift.mlops.trial import Trial, SingleTrial
from uplift.mlops.utils import ArtifactStore
from uplift.mlops.optuna_utils import OptunaHPBuilder, optuna_to_config


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

    def run(self):
        for study_config in self.config.studies:
            self.current_study_config = study_config
            study_winner = self.run_study()

    def run_study(self):
        study = optuna.create_study(
            study_name=self.current_study_config['name'],
            direction='maximize'
        )
        study.optimize(self.objective, n_trials=3)
        best_params = study.best_trial.params
        best_pipeline_config = optuna_to_config(self.config.pipeline.copy(), best_params)
        best_model_config = optuna_to_config(self.current_study_config.copy(), best_params)
        # translate back to original config structure
        out = {
            'pipeline': best_pipeline_config,
            'study': best_model_config
            }
        return out

    def objective(self, trial):
        # generate pipeline parameters via optuna
        hp_pipeline = OptunaHPBuilder(self.config.pipeline)
        pipeline_params = hp_pipeline.get_parameters(trial)
        pipeline_class = PipelineFactory().create(pipeline_params['type'])
        pipeline = pipeline_class(pipeline_params['components'])
        pipeline.build(self.data)
        # generate model config via optuna
        hp_study = OptunaHPBuilder(self.current_study_config)
        study_params = hp_study.get_parameters(trial)
        # make trial and train
        st = SingleTrial(study_params, self.config.trial_config)
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
        
