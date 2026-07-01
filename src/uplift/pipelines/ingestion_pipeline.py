from kedro.pipeline import Pipeline, node
from .ingestion.ingestion import get_dataset, save_snapshot

def create_ingestion_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=get_dataset,
                inputs="params:ingestion.dataset_name",
                outputs="ds",
            ),
            node(
                func=save_snapshot,
                inputs=["ds", "params:ingestion.dataset_name"],
                outputs="path",
            ),
            # node(
            #    func=preprocessing,
            #    inputs=[
            #        "ds_dict_raw",
            #        "params:preprocessing"
            #    ],
            #    outputs=["ds_dict", "criteo_schema"]
            # )
        ]
    )
    return out
