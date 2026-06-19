from kedro.pipeline import Pipeline, node
from .eda.eda import main
from .eda.explore import eda

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

def create_eda_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=main,
            inputs=[
                "ds_dict",
                "schema",
                "params:eda"
            ],
            outputs="eda_results"
        )
    ])
    return out

def create_explore_pipeline() -> Pipeline:
    out = Pipeline(
        [
        node(
            func=eda,
            inputs=[
                "criteo_train",
                "params:eda"
            ],
            outputs="eda_results"
        )
    ])
    return out
