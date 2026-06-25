from easydict import EasyDict as edict


class IntRangeBuilder:
    def __init__(self):
        pass

    def build(self, config):
        config = edict(config)
        config.type == "int_range"
