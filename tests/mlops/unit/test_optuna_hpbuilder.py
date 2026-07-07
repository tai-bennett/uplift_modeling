import optuna

from uplift.mlops import utils
from uplift.mlops.optuna_utils import OptunaHPBuilder


def test_optuna_hpbuilder():
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

def objective(trial):
    config = utils.load_yml("tests/mlops/unit/sample_config_optuna.yml")
    config = config['studies'][0]
    hp = OptunaHPBuilder(config)
    out = hp.get_parameters(trial)
    return len(out)
