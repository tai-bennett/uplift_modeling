from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
from pydantic import BaseModel, Field

from uplift.config.loaders import get_paths
from uplift.mlops import utils
from uplift.mlops.training_data import *


class LocalParquet(BaseModel):
    type: Literal["parquet"]
    format: str
    path: str
    metapath: str
    root: str = None
    metaroot: str = None


class LocalCSV(BaseModel):
    type: Literal["csv"]
    format: str
    path: str
    metapath: str
    root: str = None
    metaroot: str = None


class Huggingface(BaseModel):
    type: Literal["huggingface"]
    format: str
    path: str
    metapath: str
    root: str = None
    metaroot: str = None


DataSourceSpec = Annotated[
    LocalParquet | LocalCSV | Huggingface, Field(discriminator="type")
]


class DataSource(ABC):
    def __init__(self):
        self.get_paths = get_paths()

    @abstractmethod
    def load(self):
        pass


class LocalParquetDataSource(DataSource):
    def __init__(self, format, path, metapath, root, metaroot):
        super().__init__()
        self.format = format
        if root is None:
            self.path = Path(path)
        else:
            self.path = self.get_paths[root] / path

        if metaroot is None:
            self.metapath = Path(metapath)
        else:
            self.metapath = self.get_paths[metaroot] / metapath

    def load(self):
        if self.format == "polars":
            df = pl.read_parquet(self.path)
            metadata = utils.load_yml(self.metapath)
            return PolarsData(df, metadata)
        else:
            raise ValueError(f"LocalParquetDataSource doesn't support format {format}")


class LocalCSVDataSource(DataSource):
    def __init__(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class HuggingfaceDataSource(DataSource):
    def __init__(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class DataSourceFactory:
    def create(self, spec: DataSourceSpec):
        match spec:
            case LocalParquet():
                return LocalParquetDataSource
            case LocalCSV():
                return LocalCSVDataSource
            case LocalParquet():
                return HuggingfaceDataSource
