from .base_model import BaseModel
from easydict import EasyDict as edict
from uplift.config.registry import register_model
import polars as pl

@register_model('slearner')
class SLearner(BaseModel):
    def __init__(self, params):
        super().__init__(params)

    def fit(self, df, metadata):
        metadata = edict(metadata)
        # set X, T, y using metadata
        X = pl.col(metadata.feature_names)
        # X= df. ...
        T = pl.col(metadata.treatment_col)
        y = pl.col(metadata.effect_col)

        
if __name__ == "__main__":
    params = {
        'feature_names': ['a', 'b', 'c'],
        'treatment_col': 'treat',
        'effect_col': 'conversion'
    }
    model = SLearner(params)
    print("slearner file compiles")
