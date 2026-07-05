from uplift.mlops import utils
from uplift.mlops.experiment import OptunaExperiment
from uplift.config.loaders import get_paths

def train_from_config(status, config_path):
    if not status:
        raise ValueError("An error with the previous node Preprocessing has occured")
    config_path = get_paths()['config'] / config_path
    config = utils.load_yml(config_path)

    e = OptunaExperiment(config)
    model_info = e.run()
    return model_info

