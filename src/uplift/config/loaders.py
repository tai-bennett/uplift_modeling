from pathlib import Path

import yaml


def get_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_paths() -> dict:
    root = get_root()
    path = root / "conf" / "base" / "paths.yml"
    with open(path) as f:
        p = yaml.safe_load(f)

    out = {"root": root}
    for k, v in p.items():
        out[k] = root / v

    return out
