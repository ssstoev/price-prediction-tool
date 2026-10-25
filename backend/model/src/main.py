import os
import sys

from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from mlflow_settings import configure_mlflow
from config import EXPERIMENTS
from data_load import get_data_tags, load_data
from evaluate import pick_best_model
from train import generate_training_batch_id, run_experiment
import logging

configure_mlflow()

# setup basic log messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def main():
    # 1. Load the data
    df = load_data()
    if df is None:
        raise RuntimeError("load_data() returned None — check DB credentials and connection.")

    data_tags = get_data_tags(df)
    training_batch_id = generate_training_batch_id()
    
    # 2. Run the experiments with different settings
    for exp_name, exp_vals in EXPERIMENTS.items():
        run_experiment(training_batch_id, exp_name, exp_vals, df, data_tags)

    for exp_name, exp_vals in EXPERIMENTS.items():
        registered_name = exp_vals["registered_model_name"]
        version = pick_best_model(exp_name, registered_name, training_batch_id)
        log.info("[%s] Registered '%s' version: %s", exp_name, registered_name, version)


if __name__ == "__main__":
    main()