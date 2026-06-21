from abc import ABC, abstractmethod
import numpy as np
from uplift.mlops.serializer import *
import pdb
"""
A codec tells us how certain objects are saved and loaded. For example, indices for folds should be saved as a collection of files name {i}_fold which is a npz dictionary where the dictionary looks like
{'train_idx': np.array(...), 'test_idx': np.array(...)}
which is a flat dictionary for fold_i. 
Metadata dictionaries are saved differently, model weights are saved differently etc.
"""

class Codec(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save(self, path, obj):
        pass

    @abstractmethod
    def load(self, path):
        pass

class FoldIndicesCodec(Codec):
    def __init__(self):
        super().__init__()
        self.serializer = NumpySerializer()

    def save(self, path, obj, meta=None):
        # save metadata
        if meta is not None:
            ss = YamlSerializer()
            ss.save(path / "metadata.yml", meta)
            

        # save indices
        for fold_idx, fold in obj.items():
            current_name = f"{fold_idx}_fold.npz"
            current_path = path / current_name
            np.savez(current_path, **fold)

    def load(self, path):
        out = {}
        for f in path.glob("*.npz"):
            idx = f.name[0]
            ff = np.load(f)
            out[int(idx)] = {
                'train': ff['train'],
                'test': ff['test']
                }
        return out

class ConfigCodec(Codec):
    def __init__(self):
        super().__init__()
        self.serializer = YamlSerializer()
        
    def save(self, path, obj, meta=None):
        path = path / "config.yml"
        self.serializer.save(path, obj)

    def load(self, path):
        path = path / "config.yml"
        return self.serializer.load(path)

class CodecFactory():
    def create(self, name):
        if name == "fold_indices":
            return FoldIndicesCodec
        if name == "config":
            return ConfigCodec
        
        raise ValueError(f"Unknown codec type {name}")
    
