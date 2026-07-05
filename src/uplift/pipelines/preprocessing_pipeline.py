from kedro.pipeline import Pipeline, node
from .preprocessing.preprocessing import fetch_snapshot, split, simulate_monetary

def create_preprocessing_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=fetch_snapshot,
                inputs="params:ingestion.dataset_name",
                outputs=["ds_preprocessed", "metadata_raw"],
                name="preprocessing_fetch_snapshot"
            ),
            node(
                func=simulate_monetary,
                inputs=["ds_preprocessed", "metadata_raw", "params:preprocess.simulate"],
                outputs=["ds_simulated", "metadata"],
                name="preprocessing_simulation"
            ),
            node(
                func=split,
                inputs=["params:ingestion.dataset_name", "ds_preprocessed", "metadata", "params:preprocess.split"],
                outputs=["data_path_dict", "status"],
                name="preprocessing_split"
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
