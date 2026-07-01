from kedro.pipeline import Pipeline, node
from .postprocessing.postprocessing import load_model

def create_postprocessing_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=load_model,
                inputs="model_info",
                outputs="model"
            ),
        ]
    )
    return out
