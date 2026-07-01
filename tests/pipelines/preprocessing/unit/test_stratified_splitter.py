import pdb
import polars as pl
import numpy as np
from uplift.pipelines.preprocessing.splitter import StratifiedSplitter
def test_stratified_splitter():
    n = 1000
    # make sample data
    data = {
        'f1': np.random.rand(n),
        'f2': np.random.rand(n),
        'f3': np.random.rand(n),
        'target': np.random.binomial(1, 0.3, size=n)
    }
    data = pl.from_dict(data)

    # sample meta data
    meta = {
        'target_name': 'target'
    }
    split_dict = {
        'train': 0.7,
        'valid': 0.2,
        'test': 0.1
        }
    obj = StratifiedSplitter(data, meta)
    out = obj.run(data, split_dict)


    
