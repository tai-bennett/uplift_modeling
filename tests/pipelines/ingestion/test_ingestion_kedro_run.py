from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

from uplift.config.loaders import get_root


def test_ingestion_kedro_run():
    bootstrap_project(get_root())
    with KedroSession.create(
            project_path=get_root(),
            env="test"
            ) as session:
        context = session.load_context()
        runner = SequentialRunner()
        runner.run(
            pipelines['ingestion'],
            context.catalog
        )
