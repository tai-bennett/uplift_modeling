from easydict import EasyDict as edict
from pydantic import TypeAdapter

from uplift.config.loaders import get_paths
from uplift.mlops.data_source import *
from uplift.mlops.spec import *
from uplift.mlops.training_data import *
from uplift.mlops.utils import *

from .analysis import AnalysisFactory


class EDA:
    def __init__(self, config):
        self.config = edict(config)
        self.analysis_factory = AnalysisFactory()
        self.figures = []
        self.store = ArtifactStore(get_paths()["eda"])
        self.store.clear_root()

    def run(self):
        print("=========== running eda ============")
        data = self._make_dataset(self.config.data)
        for plt_config in self.config.analyses:
            analysis = self.analysis_factory.create(plt_config.type)(plt_config.config)
            path = self.config.eda.name + "/" + plt_config["name"]
            result = analysis.run(data)
            fig = analysis.create_fig(result)
            self.store.save_direct(path, fig, artifact_codec="plotly_fig")

    def _make_dataset(self, config):
        adapter = TypeAdapter(DataSourceSpec)
        spec = adapter.validate_python(config)
        data_source_class = DataSourceFactory().create(spec)
        config.pop("type")
        data_source = data_source_class(**config)
        return data_source.load()

    def _add_figures(self, current_list, obj):
        if isinstance(obj, list):
            current_list.extend(obj)
        else:
            current_list.append(obj)
