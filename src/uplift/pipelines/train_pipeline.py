from kedro.pipeline import Pipeline, node

from .training import tree
from .training import xlearner as txl
from .training.train_all import train_all

# def create_data_pipeline() -> Pipeline:
#    out = Pipeline(
#        [
#        node(
#            func=get_dataset,
#            inputs=["params:hf_data.criteo"],
#            outputs="ds_dict"
#        )
#    ])
#    return out


def create_train_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=txl.train,
                inputs=["ds_dict", "params:train.xlearner.train_params"],
                outputs="x_model",
            ),
            node(
                func=tree.train,
                inputs=["ds_dict", "params:train.tree.train_params"],
                outputs="tree_model",
            ),
        ]
    )
    return out


def create_train_all_pipeline() -> Pipeline:
    out = Pipeline(
        [
            node(
                func=train_all,
                inputs=["data_train", "metadata", "params:experiment_list"],
                outputs="experiment_results",
            )
        ]
    )
    return out
