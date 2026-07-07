from uplift.mlops.serializer import (
    NumpySerializer,
    PickleSerializer,
    SerializerFactory,
    YamlSerializer,
)


def test_serializer_factory():
    factory = SerializerFactory()
    serializer = factory.create("pickle")()
    assert type(serializer) is PickleSerializer

    serializer = factory.create("yaml")()
    assert type(serializer) is YamlSerializer

    serializer = factory.create("numpy")()
    assert type(serializer) is NumpySerializer


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
