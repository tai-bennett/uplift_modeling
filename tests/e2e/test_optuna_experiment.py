import pytest
import pdb
import uplift.mlops.utils as utils
from uplift.config.loaders import get_paths
from uplift.mlops.experiment import OptunaExperiment

def test_sample_optuna_experiment(tmp_path):
    config = utils.load_yml('tests/e2e/sample_config_optuna.yml')
    e = OptunaExperiment(config)
    e.run()
