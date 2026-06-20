from pathlib import Path
import os

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parent
ENV_PATH = BACKEND_ROOT / ".env"
DEFAULT_MLFLOW_DIR = BACKEND_ROOT / "model" / "mlruns"


def load_backend_env() -> None:
    load_dotenv(ENV_PATH)


def get_mlflow_tracking_uri() -> str:
    load_backend_env()
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_DIR.as_uri())