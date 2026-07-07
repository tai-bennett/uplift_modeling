# from umap.umap_ import umap

from easydict import EasyDict as edict


def eda(data, params):
    params = edict(params)
    _ = data.describe()
    _ = data.collect(streaming=True).sample(
        n=params.sample_size, seed=3993, shuffle=True
    )
    results = {}
    return results
