from uplift.pipelines.ingestion.ingestion import get_dataset


def test_ingestion():
    N = 7000
    data = get_dataset({'dataset_name': 'criteo/criteo-uplift', 'test': True})
    assert len(data['train']) == N
