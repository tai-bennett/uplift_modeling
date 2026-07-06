from uplift.pipelines.ingestion.ingestion import get_dataset


def test_ingestion():
    data = get_dataset({'dataset_name': 'criteo/criteo-uplift', 'test': True})
    assert len(data['train']) == 7000
