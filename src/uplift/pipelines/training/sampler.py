from easydict import EasyDict as edict

from uplift.config.registry import register_sampler


class BaseSampler:
    def __init__(self, data, params):
        self.data = data
        self.params = edict(params)

    def run(self):
        pass


@register_sampler("undersampler")
class UnderSampler(BaseSampler):
    def __init(self, data, params):
        super().__init__(data, params)
        # determine majority class

    def run(self):
        # split data into classes
        # undersample majority class with given ratio
        # combine datasets back
        # return dataset
        pass
