import json
from abc import ABC, abstractmethod
from itertools import product

from easydict import EasyDict
from pydantic import TypeAdapter, ValidationError

from uplift.mlops.parameters import *
from uplift.mlops.spec import *


class GeneratorFactory:
    def create(self, spec: ParameterSpec):
        match spec:
            case ChoiceSpec():
                return ChoiceGenerator(spec.values)
            case IntRangeSpec():
                return IntRangeGenerator(spec.min, spec.max, spec.step)


class SearchSpace(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class GridSearchSpace(SearchSpace):
    """ """

    def __init__(self, generators: dict[str, ParameterSpec], name=None):
        self.generators = generators
        self.name = name

    def __iter__(self):
        names = list(self.generators.keys())
        values = [list(generator) for generator in self.generators.values()]
        for combo in product(*values):
            if self.name is None:
                yield dict(zip(names, combo))
            else:
                out = {}
                out["name"] = self.name
                out["parameters"] = dict(zip(names, combo))
                yield out


class SearchSpaceFactory:
    def create(self, name):
        if name == "grid":
            return GridSearchSpace
        else:
            raise ValueError(f"Unknown search space type {name}")


class HPBuilder:
    def __init__(self, config):
        if type(config) == EasyDict:
            config = json.loads(json.dumps(config))
        self.built = False
        self.config = config
        self.config_bones = config.copy()
        self.path = []
        self.spec_paths = []
        self.generators = {}
        self.adapter = TypeAdapter(ParameterSpec)
        self.build()

    def __iter__(self):
        names = list(self.generators.keys())
        values = [list(generator) for generator in self.generators.values()]
        for combo in product(*values):
            self._populate_config_bones(dict(zip(names, combo)))
            yield self.config_bones

    def build(self):
        self._search(self.config)
        self._convert_specs()
        self.built = True

    def _search(self, tree):
        for k, v in tree.items():
            self.path.append(k)
            # if v is a spec
            if self._is_parameter_spec(v):
                self._handle_spec(v)
            # if v is a tree
            elif type(v) == dict:
                self._search(v)
            # if k, v is a name
            self.path.pop()

    def _handle_spec(self, v):
        self.spec_paths.append(self.path.copy())

    def _is_parameter_spec(self, d):
        try:
            self.adapter.validate_python(d)
            return True
        except ValidationError:
            return False

    def _convert_specs(self):
        for spec_path in self.spec_paths:
            self._convert_spec(spec_path)
            self._convert_config_bones(spec_path)

    def _convert_spec(self, spec_path):
        value = self._get_value_from_path(spec_path, self.config)
        self.generators[self._list_to_str(spec_path)] = GeneratorFactory().create(
            self.adapter.validate_python(value)
        )

    def _convert_config_bones(self, spec_path):
        self._set_value_from_path(spec_path, self.config_bones, None)

    def _populate_config_bones(self, params_dict):
        for (
            k,
            v,
        ) in params_dict.items():
            path = self._str_to_list(k)
            self._set_value_from_path(path, self.config_bones, v)

    def _list_to_str(self, path_list: list[str]) -> str:
        return "->".join(path_list)

    def _str_to_list(self, s: str) -> list[str]:
        return s.split("->")

    def _get_value_from_path(self, path, d):
        value = d
        for key in path:
            value = value[key]
        return value

    def _set_value_from_path(self, path, d, new_value):
        value = d
        for key in path[:-1]:
            value = value[key]
        value[path[-1]] = new_value


if __name__ == "__main__":
    data = {"type": "int_range", "min": 3, "max": 21, "step": 2}

    data2 = {"type": "choice", "values": ["path/here", "path/there"]}

    data = IntRangeSpec(**data)
    data2 = ChoiceSpec(**data2)

    g = SearchSpaceFactory().create(data)
    g2 = SearchSpaceFactory().create(data2)

    hp_space = GridSearchSpace({"depth": g, "path": g2})
    for hp in hp_space:
        print(hp)
