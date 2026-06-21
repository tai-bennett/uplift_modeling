from kedro.pipeline import Pipeline, node
from .data.ingestion import get_dataset, test_hf_load, make_hf_split, preprocessing, preprocess_small, data_ingestion, data_ingestion_small

#def create_data_pipeline() -> Pipeline:
#    out = Pipeline(
#        [
#        node(
#            func=get_dataset,
#            inputs=["params:hf_data.criteo"],
#            outputs="ds_dict"
#        )
#    ])
#    return out

def create_data_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=make_hf_split,
            inputs=[
                "criteo_uplift",
                "criteo_schema",
                "params:data_ingestion.split.size",
                "params:data_ingestion.split.shuffle",
                "params:data_ingestion.test_mode"
            ],
            outputs=["ds_dict_raw", "schema"]
        ),
        node(
            func=preprocess_small,
            inputs=[
                "ds_dict_raw",
                "params:data_ingestion"
            ],
            outputs='ds_dict'
        )
        #node(
        #    func=preprocessing,
        #    inputs=[
        #        "ds_dict_raw",
        #        "params:preprocessing"
        #    ],
        #    outputs=["ds_dict", "criteo_schema"]
            #)
    ])
    return out

def create_data_parquet_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=data_ingestion,
            inputs=["criteo_uplift", "criteo_metadata", "params:data_ingestion"],
            outputs=["data_train", "data_test", "metadata"]
        )
    ])
    return out

def create_data_small_parquet_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=data_ingestion_small,
            inputs=["criteo_uplift", "criteo_metadata", "params:data_ingestion_small"],
            outputs=["data_train_small", "data_test_small", "metadata"]
        )
    ])
    return out

def create_hf_data_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=test_hf_load,
            inputs=["criteo_uplift"],
            outputs="status"
        )
    ])
    return out
