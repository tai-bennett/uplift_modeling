"""
================================================================================
TITLE: results.py
AUTHOR: Duncan Bennett
DESCRIPTION: These are simple @dataclass objects that hold results from tuning
training etc
================================================================================
"""
from dataclasses import dataclass

from optuna import Trial


@dataclass
class TrialResults:
    trial_id: str
    config: dict
    split_results: list
    aggregate_metrics: dict
    hp_list: list


@dataclass
class ExperimentResults:
    run_id: str
    model_uri: str
    metrics: float
    config: dict

@dataclass
class TuningResults:
    trial: Trial
    config: dict
    metric: float
