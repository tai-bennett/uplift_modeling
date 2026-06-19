import pytest
import pdb
import uplift.mlops.utils as utils
from uplift.config.loaders import get_paths
from uplift.mlops.experiment import Experiment

def test_blank_experiment():
    config = utils.load_yml('tests/e2e/sample_config.yml')

    e = Experiment(config)
    e.run()
