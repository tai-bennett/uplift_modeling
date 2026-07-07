"""
================================================================================
TITLE: serializer.py
AUTHOR: Duncan Bennett
DESCRIPTION: Serializers are strategies for saving certain file formats. It is
used by the Codec class which decides how to save certain objects and
structures. For example, the FoldIndicesCodec uses the NumpySerializer to save
collections of indices (such as cv fold indices) in the appropriate way.
================================================================================
"""
import pickle
from abc import ABC, abstractmethod

import yaml


class Serializer(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass


class PickleSerializer(Serializer):
    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save(self, path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)


class NumpySerializer(Serializer):
    def load(self, path):
        pass

    def save(self, path, obj):
        pass


class YamlSerializer(Serializer):
    def load(self, path):
        with open(path) as f:
            obj = yaml.safe_load(f)
        return obj

    def save(self, path, obj):
        with open(path, "w+") as f:
            yaml.dump(obj, f)


class PlotlySerializer(Serializer):
    def load(self, path):
        pass

    def save(self, path, obj):
        if isinstance(obj, list):
            n = 0
            for item in obj:
                self._save_single_obj(path / f"plot_{n}.html", item)
                n += 1
        else:
            self._save_single_obj(path / "plot.html", obj)

    def _save_single_obj(self, path, obj):
        obj.write_html(path)


class SerializerFactory:
    def create(self, name):
        if name == "pickle":
            return PickleSerializer
        if name == "numpy":
            return NumpySerializer
        if name == "yaml":
            return YamlSerializer
        if name == 'plotly':
            return PlotlySerializer
        raise ValueError(f"Unknown Serializer type {name}")
