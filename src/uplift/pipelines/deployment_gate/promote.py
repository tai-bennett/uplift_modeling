import mlflow
import pdb
from uplift.config.loaders import get_paths


def promote(comparison_results):
    tracking_db = get_paths()['root'] / 'mlflow.db'
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    client = mlflow.MlflowClient()
    try:
        client.get_registered_model("policy_model")
    except mlflow.exceptions.MlflowException:
        print("policy_model is not a registered model, creating...")
        client.create_registered_model("policy_model")
    version = client.create_model_version(
        name="policy_model",
        source=comparison_results.winner_uri,
        run_id=comparison_results.run_id
    )

    # set model alias
    client.set_registered_model_alias(
        name="policy_model",
        alias='champion',
        version=version.version
        )
