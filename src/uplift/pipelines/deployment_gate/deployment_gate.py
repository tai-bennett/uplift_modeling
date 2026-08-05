import pdb
from dataclasses import dataclass
from typing import Any
import mlflow
import polars as pl

from .promote import promote
from .comparison import ChampionChallengerEvaluator


def champion_candidate_comparison(candidate_policy_model, data_path_dict, metadata):
    # load data
    data_path = data_path_dict['test']
    data = pl.read_parquet(data_path)

    # compare models
    evaluator = ChampionChallengerEvaluator(data, metadata)
    
    comparison_results = evaluator.run(candidate_policy_model)

    return comparison_results

def model_promotion(comparison_results):
    if not comparison_results.promote:
        print("candidate model did not beat current champion, no promotion")
        return False
    else:
        print("candidate model beats current champion, model promotion starting ...")
        # promote model
        promote(comparison_results)
        return True
