from dataclasses import dataclass


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
