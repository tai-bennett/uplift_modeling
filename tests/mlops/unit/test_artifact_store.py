import pytest

from uplift.mlops.utils import *

def test_artifact_store(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = {'a': {"type": "choice", "values": [1, 2, 3, 4]}}
    super_hash = "abc123"
    metadata = {"notes": "testing the artifact store class"}
    store.save(super_hash, metadata, artifact, artifact_codec='config')
    loaded_artifact = store.get(super_hash, metadata, artifact_codec='config')
    assert artifact == loaded_artifact

