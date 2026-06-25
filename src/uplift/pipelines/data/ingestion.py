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


def test_hf_load(hf_data):
    pdb.set_trace()
    return True


def make_hf_split(ds, schema, split_size=0.1, shuffle=True, test_mode=False):
    data = ds["train"].train_test_split(test_size=split_size, shuffle=shuffle)
    if test_mode:
        out = DatasetDict(
            {split: ds.select(range(0, 10000)) for split, ds in data.items()}
        )
    else:
        out = data

    return [out, schema]


def preprocess_small(ds, params):
    return ds


def preprocessing(ds, params):
    pass
    # data = ds['train'].to_pandas()
    # pp = Preprocessor(data, params)
    # data, schema = pp.run()
    ## ds = synthetic_over_sampling(ds, params['smote'])
    ## insert others here if necessary
    # return data, schema


def apply_schema(data, schema):
    data[schema.category] = data[schema.category].astype("category")
    data[schema.continuous] = data[schema.continuous].astype("float")
    return data


def data_ingestion(ds, meta, params):
    params = edict(params)
    ds = ds["train"].train_test_split(
        test_size=params.split.size, shuffle=params.split.shuffle
    )
    train = pl.from_arrow(ds["train"].data.table)
    test = pl.from_arrow(ds["test"].data.table)
    # make directories
    # make stratify labels
    if params.get("stratify_columns") is not None:
        train = stratify(train, params.stratify_columns)
    # create n fold for validation
    train = make_folds(train, params.nfolds)
    # save to parquet
    train.write_parquet(get_paths()["data_primary"] / "criteo_train.parquet")
    test.write_parquet(get_paths()["data_primary"] / "criteo_test.parquet")
    return [train, test, meta]


def data_ingestion_small(ds, meta, params):
    params = edict(params)
    ds = ds["train"].train_test_split(
        test_size=params.split.size, shuffle=params.split.shuffle
    )
    train = pl.from_arrow(ds["train"].data.table)
    test = pl.from_arrow(ds["test"].data.table)
    majority = train.filter(pl.col("conversion") == 0)
    minority = train.filter(pl.col("conversion") == 1)
    majority_sample = majority.sample(n=2 * len(minority), shuffle=True, seed=42069)

    train_balanced = pl.concat([majority_sample, minority]).sample(
        fraction=1, shuffle=True, seed=49393
    )
    # make directories
    # make stratify labels
    # if params.get('stratify_columns') is not None:
    #     train = stratify(train, params.stratify_columns)
    # create n fold for validation
    # train = make_folds(train, params.nfolds)
    # save to parquet
    train = train_balanced.sample(n=params.n_train, seed=params.seed)
    test = test.sample(n=params.n_test, seed=params.seed)
    train.write_parquet(get_paths()["data_primary"] / "criteo_train_small.parquet")
    test.write_parquet(get_paths()["data_primary"] / "criteo_test_small.parquet")
    return [train, test, meta]


def make_folds(data, n):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    folds = np.zeros(len(data), dtype=np.int32)
    strat_labels = data["stratify_label"].to_numpy()
    dummy = folds
    for fold_idx, (_, val_idx) in enumerate(skf.split(dummy, strat_labels)):
        folds[val_idx] = fold_idx

    data = data.with_columns(pl.Series(name="cv_fold", values=folds))
    data = data.drop("stratify_label")
    return data


def stratify(data, strat_cols):
    data = data.with_columns(
        pl.concat_str(
            [pl.col(col_name).cast(pl.Utf8) for col_name in strat_cols], separator="_"
        ).alias("stratify_label")
    )
    return data
