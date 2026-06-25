from uplift.mlops import utils
from uplift.mlops.search_space import HPBuilder


def test_hpbuilder():
    config = utils.load_yml("tests/mlops/unit/sample_config.yml")

    builder = HPBuilder(config)
    for item in builder:
        pass
