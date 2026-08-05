from kedro.pipeline import Pipeline, node

from .deployment_gate.deployment_gate import champion_candidate_comparison, model_promotion


def create_deployment_gate_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=champion_candidate_comparison,
                inputs=["candidate_policy_model", "data_path_dict", "metadata"],
                outputs="comparison_results",
                name="champion_candidate_comparision"
            ),
            node(
                func=model_promotion,
                inputs="comparison_results",
                outputs="promotion_status",
                name="model_promotion"
            ),
        ]
    )
    return out
