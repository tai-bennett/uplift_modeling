from kedro.pipeline import Pipeline, node

from .postprocessing.postprocessing import calibration, load_model, policy_method


def create_postprocessing_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=load_model,
                inputs="training_results",
                outputs="model",
                name="postprocessing_load_model"
            ),
            node(
                func=calibration,
                inputs=["model", "data_path_dict", 'metadata'],
                outputs="calibrated_model",
                name="postprocessing_calibration"
            ),
            node(
                func=policy_method,
                inputs=["calibrated_model", "data_path_dict", 'metadata'],
                outputs="candidate_policy_model",
                name="postprocessing_policy"
            ),
        ]
    )
    return out
