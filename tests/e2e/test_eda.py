import pytest
import pdb
import uplift.mlops.utils as utils
from uplift.config.loaders import get_paths
from uplift.mlops.eda import EDA

def test_sample_eda():
    config = utils.load_yml('tests/e2e/sample_config_eda.yml')

    e = EDA(config)
    e.run()
