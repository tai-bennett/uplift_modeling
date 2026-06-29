import pprint

from uplift.mlops import utils
from uplift.mlops.optuna_utils import optuna_to_config


def test_optuna_to_config():
    config = {
        'name': "hello",
        'age': "hello",
        'parameter':
        {'type': 'uniform', 'name': 'testing', 'low': 3, 'high': 100}
        }
    params = {'testing': 4}
    answer = {
        'name': "hello",
        'age': "hello",
        'parameter': 4
        }
    out = optuna_to_config(config, params)
    assert out == answer

def test_optuna_to_config2():
    config = utils.load_yml("tests/mlops/unit/sample_config_optuna.yml")
    config = config['studies'][0]
    params = {
        'n_estimators': 5,
        'max_depth': 5,
        'learning_rate': 1,
        'objective': 'binary:logistic',
        'extra': 'hello'
    }

    out = optuna_to_config(config, params)
    pprint.pp(out)
    pprint.pp(config)



