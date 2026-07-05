from kedro.pipeline import Pipeline, node
from .ingestion.ingestion import get_dataset, save_snapshot

def create_ingestion_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=get_dataset,
                inputs="params:ingestion",
                outputs="ds",
                name="ingestion"
            ),
            node(
                func=save_snapshot,
                inputs=["ds", "params:ingestion.dataset_name"],
                outputs="path",
                name="ingestion_save_snapshot"
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
