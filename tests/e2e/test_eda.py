from uplift.mlops import utils
from uplift.mlops.eda import EDA


def test_sample_eda():
    config = utils.load_yml("tests/e2e/sample_config_eda.yml")

    e = EDA(config)
    e.run()
