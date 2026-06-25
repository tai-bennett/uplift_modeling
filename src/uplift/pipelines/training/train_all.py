import logging

from easydict import EasyDict as edict

from uplift.config.registry import MODEL_REGISTRY
from uplift.models import *
from uplift.ops.experiment import Experiment


def train_all(data, metadata, experiments_list):
    logger = logging.getLogger(__name__)
    results = {}
    for exp in experiments_list:
        exp = edict(exp)
        current = Experiment(exp)
        # define model
        logger.info(12 * "=" + " " + exp.name + " " + 12 * "=")
        model = MODEL_REGISTRY[exp.architecture](exp.model_params)
        # train/select model
        model.fit(data, metadata)
        # prepare model results
        results[exp.name] = model.get_info()

    return results
