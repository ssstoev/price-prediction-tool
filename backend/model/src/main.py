import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import mlflow
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from mlflow_settings import get_mlflow_tracking_uri
from config import EXPERIMENTS
from data_load import load_data
from evaluate import pick_best_model
from train import run_experiment
import logging

MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# setup basic log messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    df = load_data()
    # print(df)
    if df is None:
        raise RuntimeError("load_data() returned None — check DB credentials and connection.")
    for exp_name, exp_vals in EXPERIMENTS.items():
        run_experiment(exp_name, exp_vals, df)

    for exp_name, exp_vals in EXPERIMENTS.items():
        registered_name = exp_vals["registered_model_name"]
        version = pick_best_model(exp_name, registered_name)
        print(f"[{exp_name}] Registered '{registered_name}' version: {version}")


if __name__ == "__main__":
    main()