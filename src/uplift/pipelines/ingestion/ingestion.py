import polars as pl
from datasets import load_dataset

from uplift.config.loaders import get_paths


def get_dataset(params):
    try:
        test_status = params['test']
    except KeyError:
        test_status = False
    data_name = params['dataset_name']
    if test_status:
        ds = load_dataset(data_name, split='train[:7000]')
        ds = {'train': ds}
    else:
        ds = load_dataset(data_name)
    return ds

def save_snapshot(ds, name):
    ds = pl.from_arrow(ds['train'].data.table)
    path = get_paths()['raw'] / (name + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_parquet(path)
    return path
