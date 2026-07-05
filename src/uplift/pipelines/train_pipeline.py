from kedro.pipeline import Pipeline, node
from .training.train import train_from_config

def create_train_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=train_from_config,
                inputs=["status", "params:train.config"],
                outputs="training_results",
            ),
        ]
    )
    return out
