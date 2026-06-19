import yaml
import pdb
from easydict import EasyDict as edict
from uplift.mlops.spec import *
from uplift.mlops.search_space import *
from uplift.mlops.trial import *
from uplift.mlops.training_data import *
from uplift.mlops.pipeline import PipelineFactory
from uplift.mlops.trial import Trial
from uplift.mlops.data_source import *
import uplift.mlops.utils as utils
from pydantic import TypeAdapter

class Experiment():
    def __init__(self, config):
        self.config = edict(config)
        # create search space for pipeline hp
        # self.pipeline_hp = self._make_hp_search_space(self.config.pipelines.parameters)

    def run(self):
        data = self._make_dataset(self.config.data)
        for pipeline_config in self.config.pipelines:
            print("Running pipeline: " + pipeline_config.name)
            # make pipeline class from factory
            pipeline_class = PipelineFactory().create(pipeline_config.type)
            # make hp space for pipeline
            pipeline_hp_space = self._make_hp_search_space(pipeline_config.components)
            # for hp in hpspace
            for hp in pipeline_hp_space:
                print("Pipeline parameters: " + str(hp))
                # make pipeline from factory
                pipeline = pipeline_class(hp)
                pipeline.build(data)
                for trial_config in self.config.trials:
                    trial = Trial(self.config.trial_config, trial_config)
                    trial.run(data, pipeline)

    def _make_dataset(self, config):
        adapter = TypeAdapter(DataSourceSpec)
        spec = adapter.validate_python(config)
        data_source_class = DataSourceFactory().create(spec)
        config.pop('type')
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
            generators[component_name]= GridSearchSpace(subgenerators, name=component_config.name)
        hp_space = GridSearchSpace(generators)
        return hp_space
        
if __name__ == "__main__":
    config = utils.load_yml('sample_config.yml')
#    with open('sample_config.yml', 'r') as f:
#        config = yaml.safe_load(f)

    e = Experiment(config)
    e.run()
