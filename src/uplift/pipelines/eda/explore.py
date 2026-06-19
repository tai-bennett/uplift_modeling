import pandas as pd
import polars as pl
import plotly.express as px
from uplift.config.loaders import get_paths
from .utils import get_numeric_columns, get_category_columns, save_figure
#from umap.umap_ import umap
import umap
import pdb
import pprint
from easydict import EasyDict as edict

def eda(data, params):
    params = edict(params)
    summary = data.describe()
    sample = (
        data
        .collect(streaming=True)
        .sample(
            n=params.sample_size,
            seed=3993,
            shuffle=True
        )
    )
    # sample = (
    #     data
    #     .with_columns(
    #         pl.int_range(len(data)).shuffle(seed=2929).alias("_rand")
    #         )
    #     .filter(pl.col("_rand") < params.sample_size)
    #     .drop("_rand")
    # )
    pdb.set_trace()
    results = {}
    return results
