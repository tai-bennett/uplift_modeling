from uplift.mlops.serializer import *


def test_serializer_factory():
    factory = SerializerFactory()
    serializer = factory.create("pickle")()
    assert type(serializer) == PickleSerializer

    serializer = factory.create("yaml")()
    assert type(serializer) == YamlSerializer

    serializer = factory.create("numpy")()
    assert type(serializer) == NumpySerializer


def test_serializer_yaml(tmp_path):
    artifact = {
        "a": [1, 2, 3, 4],
        "b": {"un": "hello", "deux": "bye", "trois": "ok"},
        "c": "duh",
    }
    serializer = YamlSerializer()
    path = tmp_path / "test.yml"
    serializer.save(path, artifact)
    loaded_artifact = serializer.load(path)
    assert artifact == loaded_artifact
