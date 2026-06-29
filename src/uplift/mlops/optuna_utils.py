import json

from easydict import EasyDict
from pydantic import TypeAdapter, ValidationError

from uplift.mlops.parameters import *
from uplift.mlops.spec import *


class OptunaHPBuilder:
    def __init__(self, config):
        if type(config) == EasyDict:
            config = json.loads(json.dumps(config))
        self.built = False
        self.config = config
        self.config_bones = config.copy()
        self.path = []
        self.spec_paths = []
        # self.generators = {}
        self.adapter = TypeAdapter(OptunaParameterSpec)
        # self.build()
        self.trial = None

    def get_parameters(self, trial):
        self._search(self.config, trial)
        # self._convert_specs()
        return self.config_bones

    def _search(self, tree, trial):
        for k, v in tree.items():
            self.path.append(k)
            # if v is a spec
            if self._is_parameter_spec(v):
                self._handle_spec(v, trial)
            # if v is a tree
            elif type(v) == dict:
                self._search(v, trial)
            # if k, v is a name
            self.path.pop()

    def _handle_spec(self, v, trial):
        # self.spec_paths.append(self.path.copy())
        spec = self.adapter.validate_python(v)
        d = spec.model_dump()
        method_name = d.pop('type')
        method_name = "suggest_" + method_name
        method = getattr(trial, method_name)
        # d['name'] = ".".join(self.path)
        param = method(**d)
        self._set_value_from_path(self.path, self.config_bones, param)

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

def is_parameter_spec(d):
    adapter = TypeAdapter(OptunaParameterSpec)
    try:
        adapter.validate_python(d)
        return True
    except ValidationError:
        return False

def optuna_to_config(config, params):
    """
    config: is the original config that has a OptunaParameterSpec at nodes
        that Optuna handled
    params: is a flat dictionary produced by Optuna where
        params[param_name] = value
    description: this method translates the optuma params into the original
    structure so that it may be passed to a model/pipeline factory

    """
    if is_parameter_spec(config):
        return params[config['name']]
    if type(config) == EasyDict:
        config = json.loads(json.dumps(config))
        return {k: optuna_to_config(v, params) for k, v in config.items()}
    if type(config) == dict:
        return {k: optuna_to_config(v, params) for k, v in config.items()}
    return config



