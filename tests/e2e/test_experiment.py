from uplift.mlops import utils
from uplift.mlops.experiment import Experiment


def test_blank_experiment():
    config = utils.load_yml("tests/e2e/sample_config.yml")

    e = Experiment(config)
    e.run()
