import pdb
import numpy as np
import polars as pl
from datasets import DatasetDict, load_dataset
from easydict import EasyDict as edict
from sklearn.model_selection import StratifiedKFold

from uplift.config.loaders import get_paths

def get_dataset(data_name):
    ds = load_dataset(data_name)
    return ds

def save_snapshot(ds, name):
    ds = pl.from_arrow(ds['train'].data.table)
    path = get_paths()['raw'] / (name + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_parquet(path)
    return path
