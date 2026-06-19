MODEL_REGISTRY = {}
SAMPLER_REGISTRY = {}

"""
When defining a model class add the decorator

@register_model("resnet50")
class ResNet(nn.Module)
    def __init__(self, ...)
        blah blah

Then we can get the class models by referencing the dictionary MODEL_REGISTRY['resnet50']
"""

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def register_sampler(name):
    def decorator(cls):
        SAMPLER_REGISTRY[name] = cls
        return cls
    return decorator


