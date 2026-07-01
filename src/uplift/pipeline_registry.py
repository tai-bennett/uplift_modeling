"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipelines.ingestion_pipeline import create_ingestion_pipeline
from .pipelines.preprocessing_pipeline import create_preprocessing_pipeline
from .pipelines.train_pipeline import create_train_pipeline
from .pipelines.postprocessing_pipeline import create_postprocessing_pipeline

from .pipelines.data_pipeline import (
    create_data_parquet_pipeline,
    create_data_pipeline,
    create_data_small_parquet_pipeline,
)

# from .pipelines.train_pipeline import create_train_pipeline
# from .pipelines.train_pipeline import create_train_all_pipeline
from .pipelines.eda_pipeline import create_eda_pipeline, create_explore_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines(raise_errors=True)
    # pipelines["__default__"] = sum(pipelines.values())
    ingestion = create_ingestion_pipeline()
    preprocessing = create_preprocessing_pipeline()
    training = create_train_pipeline()
    postprocessing = create_postprocessing_pipeline()
    data = create_data_pipeline()
    data_parquet = create_data_parquet_pipeline()
    data_parquet_small = create_data_small_parquet_pipeline()
    # train_all = create_train_all_pipeline()
    # train = create_train_pipeline()
    eda = create_eda_pipeline()
    explore = create_explore_pipeline()
    pipelines = {}
    # pipelines["__default__"] = data + eda + train
    pipelines["__default__"] = ingestion + preprocessing + training + postprocessing
    pipelines["ingestion"] = ingestion
    pipelines["preprocessing"] = preprocessing
    pipelines["training"] = training
    pipelines["parquet_small"] = data_parquet_small
    pipelines["eda"] = data + eda
    pipelines["parquet"] = data_parquet + explore
    pipelines["explore"] = explore
    # pipelines["hf"] = data_hf
    # pipelines["ingestion_train_all"] = data_parquet + train_all
    # pipelines["train_all"] = train_all
    return pipelines
