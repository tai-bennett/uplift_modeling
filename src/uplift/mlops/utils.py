from abc import ABC, abstractmethod
import os
import shutil
import pdb
import yaml
import polars as pl
from uplift.config.loaders import get_paths
from pathlib import Path
from pydantic import BaseModel
from uplift.config.loaders import get_paths
from uplift.mlops.serializer import *
from uplift.mlops.codec import *
from typing import Literal
import json
import hashlib
import pickle

def load_yml(path):
    with open(path, "r") as f:
        out = yaml.safe_load(f)
    return out
    
def load_data_from_path(root, name, format=None):
    paths = get_paths
    ext = Path(name).suffix

    path = paths[root] / name

    if format == "polars":
        df = pl.read_parquet(path)
        return df
    elif ext.casefold() == ".yml".casefold():
        with open(path, "r") as f:
            out = yaml.safe_load(f)
        return out
    else:
        raise ValueError(f"File extension {ext} not supported")

# def ArtifactSpec(BaseModel):
#     id: str

class ArtifactStore():
    def __init__(self, root=None):
        if root is None:
            self.root = get_paths()['artifacts']
        else:
            self.root = root

    def save_direct(self, path, obj, artifact_codec=None):
        path = self.root / path
        path.mkdir(parents=True, exist_ok=True)
        if artifact_codec is None:
            with open(path / "artifact.pkl", "wb") as f:
                pickle.dump(artifact, f)
        else:
            codec = CodecFactory().create(artifact_codec)()
            codec.save(path, obj)

    def save(self, super_hash, params, artifact, artifact_codec=None):
        hash_new = self._combine_hash_dict(super_hash, params)
        path = self.root / hash_new
        path.mkdir(parents=True, exist_ok=True)
        if artifact_codec is None:
            with open(path / "artifact.pkl", "wb") as f:
                pickle.dump(artifact, f)
            with open(path / "metadata.yml", "wb") as f:
                pickle.dump(params, f)
        else:
            codec = CodecFactory().create(artifact_codec)()
            codec.save(path, artifact, meta=params)

    def clear_root(self):
        for name in os.listdir(self.root):
            full_path = os.path.join(self.root, name)
            try:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.unlink(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            except Exception as e:
                print(f"Failed to delete file {full_path} due to {e}")
            
            

    def get(self, super_hash, params, artifact_codec=None):
        # make combined hash
        hash_new = self._combine_hash_dict(super_hash, params)
        path = self.root / hash_new
        # test if directory exists
        if path.exists():
            return self._load(path, artifact_codec)
        else:
            return None

    def _load(self, path, codec):
        if codec is None:
            with open(path / "artifact.pkl", "rb") as f:
                obj = pickle.load(f)
        else:
            codec = CodecFactory().create(codec)()
            obj = codec.load(path)
        return obj

    def _combine_hash_dict(self, super_hash, dictionary):
        payload = (
            super_hash + 
            json.dumps(dictionary, sort_keys=True, separators=(",", ":"))
            ).encode("utf-8")
        hash_new = hashlib.sha256(payload).hexdigest()
        return hash_new


