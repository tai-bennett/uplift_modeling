import subprocess
from pathlib import Path
from uplift.config.loaders import get_paths

def main():
    tracking_db = get_paths()['root'] / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db}"

    subprocess.run(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri,
            "--port",
            "5000"
        ],
        check=True
    )

if __name__ == "__main__":
    main()
