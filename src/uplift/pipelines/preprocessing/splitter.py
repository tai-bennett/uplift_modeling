import pdb
import polars as pl
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class BaseSplitter(ABC):
    def __init__(self):
        pass

    def run(self, ds):
        pass


class StratifiedSplitter(BaseSplitter):
    def __init__(self, ds, meta):
        self.ds = ds
        self.meta = meta

    def run(self, ds, split_dict):
        data = ds
        proportion = 1.0
        out = {}
        for k, v in split_dict.items():
            split, remaining = self._make_split(data, v, proportion)
            out[k] = split
            data = remaining
            proportion -= v
            if remaining is None:
                break
        return out

    def _make_split(self, data, v, proportion):
        # compute split proportions
        phi = v/proportion
        if (1-phi)*len(data) < 1:
            return data, None
        y = data[self.meta['target_name']]
        X = data.drop(self.meta['target_name'])

        split_X, remain_X, split_y, remain_y = train_test_split(
            X, y,
            train_size=phi,
            stratify=y,
            random_state=1919
            )
        # combine splits back to pl dataframe
        split = split_X.with_columns(split_y.alias(self.meta['target_name']))
        remain = remain_X.with_columns(remain_y.alias(self.meta['target_name']))
        return split, remain

class SplitterFactory:
    def create(self, name):
        if name == 'stratified':
            return StratifiedSplitter
        raise ValueError(f"Unknown splitter of type {name}.")
