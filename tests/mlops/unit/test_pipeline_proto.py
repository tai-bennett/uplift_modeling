import numpy as np
import polars as pl
from uplift.mlops import utils
from uplift.mlops.pipeline import PipelineProto
from uplift.mlops.training_data import PolarsData

def test_pipeline_proto():
    config = {
        'splitter': {
            'name': 'stratified_k_fold',
            'parameters': {
                'num_splits': 5
                }
        },
        'sampler': {
            'name': 'undersampler',
            'parameters': {
                'ratio': 0.5,
                'seed': 123123
                }
            }
        }

    n = 100
    f1 = np.random.rand(n)
    f2 = np.random.rand(n)
    f3 = np.random.rand(n)
    labels = np.random.binomial(1, 0.2, size=n)
    data = {'f1': f1, 'f2': f2, 'f3': f3, 'labels': labels}
    data = pl.from_dict(data)
    meta = {'feature_names': ['f1', 'f2', 'f3'], 'target_name': 'labels'}

    data = PolarsData(data, meta)
    pipe = PipelineProto(config)
    pipe.build(data)
