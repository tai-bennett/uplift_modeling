from uplift.mlops import utils
from uplift.mlops.experiment import OptunaExperiment


def test_sample_optuna_experiment(tmp_path):
    config = utils.load_yml('tests/e2e/sample_config_optuna.yml')
    e = OptunaExperiment(config)
    e.run()
