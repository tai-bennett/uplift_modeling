import mlflow
import pdb
import numpy as np
from dataclasses import dataclass
from uplift.config.loaders import get_paths

class ChampionChallengerEvaluator():
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.name = "champion_challeger"
        self.candidate_uri = None
        self.candidate_run_id = None
        tracking_db = get_paths()['root'] / 'mlflow.db'
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
        mlflow.set_experiment(self.name)
        # mlflow client
        self.client = mlflow.MlflowClient()

    def run(self, candidate_model):
        with mlflow.start_run(run_name=self.name) as run:
            # log challenger model
            model_info = self._log_challenger(candidate_model)
            self.candidate_uri = model_info.model_uri
            self.candidate_run_id = run.info.run_id
            # get champion if exists
            champion_model = self._fetch_champion()
            # comparison logic
            results = self._run_model_comparison(candidate_model, champion_model)
        return results

    def _log_challenger(self, model):
        # make example input
        X = self.data[self.metadata['feature_names']].to_pandas()
        X['treatment'] = self.data['treatment'].to_numpy()
        X['outcome'] = self.data[self.metadata['revenue_name']].to_numpy()
        X = X[0:3]

        model(X)
        # log challenger
        print("logging policy model now...")
        model_info = mlflow.pyfunc.log_model(
            name="policy_model",
            python_model=model,
            input_example=X
        )
        return model_info

    def _fetch_champion(self):
        try:
            # champion = self.client.get_model_version_by_alias(
            #     'policy_model',
            #     'champion'
            #     )
            # pdb.set_trace()
            champion = mlflow.pyfunc.load_model(
                "models:/policy_model@champion"
                )
            return champion
        except Exception:
            return None
        

    def _run_model_comparison(self, challenger, champion):
        # get model_input, treatment and outcome
        X = self.data[self.metadata['feature_names']].to_pandas()
        treatment = self.data[self.metadata['treatment_name']].to_numpy()
        conversion = self.data[self.metadata['target_name']].to_numpy()
        R = self.data[self.metadata['revenue_name']].to_numpy()
        C = self.data[self.metadata['cost_name']].to_numpy()
        outcome = treatment * conversion * (R - C) + treatment * (1 - conversion) * (- C) + (1 - treatment) * conversion * R

        X['treatment'] = treatment
        X['outcome'] = outcome

        # evaluate challenger
        epsilon = 0.005
        mask = challenger.predict(X)
        challenger_ipw = ipw(mask, treatment, outcome)

        if champion is None:
            champion_ipw = - np.inf
        else:
            mask = champion.predict(X)
            champion_ipw = ipw(mask, treatment, outcome)
        promote = (champion_ipw + epsilon) < challenger_ipw

        if promote:
            winner_uri = self.candidate_uri
        else:
            winner_uri = "none"
        
        results = promotionDecision(
            winner_uri,
            promote,
            {'ipw': champion_ipw},
            {'ipw': challenger_ipw},
            self.candidate_run_id
            )
        
        return results


def ipw(policy, treatment, outcome):
    n = len(treatment)
    one = np.ones(n)
    # compute propensity
    e = sum(treatment)/len(treatment)

    v = (policy * treatment * outcome) / e + (one - policy) * (one - treatment) * outcome / (one - e)
    return (1/n) * sum(v)

@dataclass
class promotionDecision:
    winner_uri: str
    promote: bool
    champion_metrics: dict[str, float]
    candidate_metrics: dict[str, float] | None
    run_id: str

# @dataclass
# class promotionDecision:
#     winner_uri: str
#     winner_version: str | None
#     champion_version: str | None
#     candidate_version: str
#     promote: bool
#     champion_metrics: dict[str, float]
#     candidate_metrics: dict[str, float] | None
