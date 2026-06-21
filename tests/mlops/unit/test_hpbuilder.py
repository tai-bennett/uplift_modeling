import pytest
import pdb
from easydict import EasyDict as edict
from uplift.mlops.search_space import HPBuilder
from uplift.mlops.trial import Trial
import uplift.mlops.utils as utils


def test_hpbuilder():
    config = utils.load_yml('tests/mlops/unit/sample_config.yml')

    builder = HPBuilder(config)
    for item in builder:
        pass

    
