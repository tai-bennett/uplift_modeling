import yaml

def load_yml(path):
    with open(path) as f:
        out = yaml.safe_load(f)
    return out
