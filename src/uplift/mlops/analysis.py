from abc import ABC, abstractmethod

import plotly.express as px


class AnalysisFactory:
    def create(self, name):
        if name == "feature_distribution":
            return FeatureDistribution
        if name == "class_imbalance":
            return ClassImbalance
        raise ValueError(f"Unknown type of analysis named {name}")


class Analysis(ABC):
    @abstractmethod
    def run(self, data):
        return None

    @abstractmethod
    def create_fig(self, results):
        pass


class FeatureDistribution(Analysis):
    def __init__(self, config):
        self.config = config

    def run(self, data):
        results = []
        self.data = data
        for name in data.metadata["feature_names"]:
            feat_fig = self._histogram(name)
            results.append(feat_fig)
        return results

    def create_fig(self, results):
        fig = results
        return fig

    def _histogram(self, name):
        fig = px.histogram(
            self.data.data,
            x=name,
            color=self.config["color"],
            marginal="box",
            barmode="overlay",
            nbins=50,
            opacity=0.7,
            title=f"distribution of {name}",
        )
        return fig


class ClassImbalance:
    def __init__(self, config):
        self.config = config

    def run(self, data):
        results = []
        self.data = data
        for name in self.config.class_name:
            fig = self._plot(name)
            results.append(fig)
        return results

    def create_fig(self, results):
        return results

    def _plot(self, name):
        counts = self.data.data[name].value_counts()
        fig = px.bar(
            x=counts[name],
            y=counts["count"],
            labels={"x": name, "y": "Counts"},
            title=f"Class distribution for {name}",
        )
        return fig
