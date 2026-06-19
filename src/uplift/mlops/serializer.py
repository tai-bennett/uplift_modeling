from abc import ABC, abstractmethod
from pathlib import Path
import pickle
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
        with open(path, "r") as f:
            obj = yaml.safe_load(f)
        return obj

    def save(self, path, obj):
        with open(path, 'w+') as f:
            yaml.dump(obj, f)


class SerializerFactory():
    def create(self, name):
        if name == "pickle":
            return PickleSerializer
        if name == "numpy":
            return NumpySerializer
        if name == "yaml":
            return YamlSerializer
        raise ValueError(f"Unknown Serializer type {name}")
    
        
