from kedro.framework.session import KedroSession
from uplift.config.loaders import get_root

def test_ingestion_kedro_run():
    with KedroSession.create(
            project_path=get_root(),
            env="test"
            ) as session:
        context = session.load_context()
