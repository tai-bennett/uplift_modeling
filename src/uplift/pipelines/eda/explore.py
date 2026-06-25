# from umap.umap_ import umap
import pdb

from easydict import EasyDict as edict


def eda(data, params):
    params = edict(params)
    summary = data.describe()
    sample = data.collect(streaming=True).sample(
        n=params.sample_size, seed=3993, shuffle=True
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
