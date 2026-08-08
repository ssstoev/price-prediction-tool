import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parent
ENV_PATH = BACKEND_ROOT / ".env"
DEFAULT_MLFLOW_DIR = BACKEND_ROOT / "model" / "mlruns"


def load_backend_env() -> None:
    load_dotenv(ENV_PATH, override=True)


def get_mlflow_tracking_uri() -> str:
    load_backend_env()
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_DIR.as_uri())


def get_mlflow_artifact_root() -> str | None:
    """Return the default artifact root for new experiments (S3/Supabase bucket path)."""
    load_backend_env()
    return os.getenv("MLFLOW_ARTIFACT_ROOT")


def configure_mlflow() -> None:
    """Load .env, forward S3 credentials into the environment, and set the MLflow tracking URI.

    Expects these variables in .env:
      MLFLOW_TRACKING_URI      — postgresql+psycopg://... connection to Neon
      MLFLOW_ARTIFACT_ROOT     — s3://bucket/prefix for new experiment artifact locations
      AWS_ACCESS_KEY_ID        — Supabase Storage S3 access key
      AWS_SECRET_ACCESS_KEY    — Supabase Storage S3 secret key
      MLFLOW_S3_ENDPOINT_URL   — https://<project>.supabase.co/storage/v1/s3
      AWS_DEFAULT_REGION       — region reported by Supabase (usually us-east-1)
    """
    import mlflow

    load_backend_env()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_DIR.as_uri())
    mlflow.set_tracking_uri(tracking_uri)


def launch_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Start a local MLflow tracking server backed by Postgres and Supabase S3.

    Run directly:  python backend/mlflow_settings.py
    """
    load_backend_env()
    subprocess.run(
        [
            "mlflow", "server",
            "--host", host,
            "--port", str(port),
            "--backend-store-uri", get_mlflow_tracking_uri(),
            "--artifacts-destination", get_mlflow_artifact_root(),
            "--serve-artifacts",
        ],
        check=True,
    )


if __name__ == "__main__":
    launch_server()