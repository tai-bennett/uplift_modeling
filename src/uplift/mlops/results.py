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
    experiment_id: str
    config: dict
    trials: list
    hp_list: list

@dataclass
class TuningResults:
    trial: Trial
    config: dict
    metric: float
