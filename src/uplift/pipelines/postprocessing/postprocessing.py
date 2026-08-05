import pdb
import mlflow
import polars as pl

from .calibration import Calibrator
from .policy import PolicySelection
from .policymodel import PolicyModel



def load_model(training_results):
    model = mlflow.pyfunc.load_model(training_results.model_uri)
    model = PolicyModel(model)
    return model

def calibration(model, data_path_dict, metadata):
    try:
        data_path = data_path_dict['validate_calibration']
        data = pl.read_parquet(data_path)
    except KeyError:
        data = None
    if data is None:
        return model
    else:
        cal = Calibrator(metadata)
        model = cal.run(model, data)
        return model

def policy_method(model, data_path_dict, metadata):
    try:
        data_path = data_path_dict['validate_policy']
        data = pl.read_parquet(data_path)
    except KeyError:
        data = None
    if data is None:
        return model
    else:
        policy = PolicySelection(metadata)
        model = policy.fit(model, data)
        return model


