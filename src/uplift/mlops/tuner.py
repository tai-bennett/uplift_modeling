from abc import ABC, abstractmethod


class BaseTuner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tune(self):
        pass


class CVTuner(BaseTuner):
    def __init__(self, params):
        super().__init__(self)
        # splitter object: split data into folds or detect folds
        # self.splitter = splitter_factory(params.splitter)

    def tune(self, model, data):
        pass
