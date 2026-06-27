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
from uplift.mlops.trial import Trial
from uplift.mlops.utils import *


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

    def run(self):
        pass
