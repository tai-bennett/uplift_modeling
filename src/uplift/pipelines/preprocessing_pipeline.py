from kedro.pipeline import Pipeline, node
from .preprocessing.preprocessing import fetch_snapshot, split

def create_preprocessing_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=fetch_snapshot,
                inputs="params:ingestion.dataset_name",
                outputs=["ds_preprocessed", "metadata"],
            ),
            node(
                func=split,
                inputs=["params:ingestion.dataset_name", "ds_preprocessed", "metadata", "params:preprocess.split"],
                outputs=["path_dict", "status"],
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
