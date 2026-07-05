import pdb
import polars as pl
from .splitter import SplitterFactory
from .simulator import Simulator
from uplift.config.loaders import get_paths
from uplift.utils import load_yml

def fetch_snapshot(dataset_name):
    root = get_paths()['raw']
    path = root / (dataset_name + ".parquet")
    root = get_paths()['meta']
    ds = pl.read_parquet(path)
    path = root / (dataset_name + ".yml")
    meta = load_yml(path)
    return ds, meta

def simulate_monetary(ds, meta, params):
    sim = Simulator(**params)
    ds, meta = sim.run(ds, meta)
    return ds, meta

def split(dataset_name, ds, meta, params):
    splitter = SplitterFactory().create(params['type'])(ds, meta)
    splits = splitter.run(ds, params['split_dict'])
    # save
    root = get_paths()['data_primary']
    out = {}
    for name, data in splits.items():
        # math path with k
        path = root / (dataset_name + f"_{name}" + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(path)
        out[name] = path
    return out, True

