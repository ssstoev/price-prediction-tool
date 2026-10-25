"""Register the best model of an earlier training batch, without retraining.

Usage (from backend/):
  python scripts/log_specific_models.py                       # latest batch of each experiment
  python scripts/log_specific_models.py --batch-id <id>       # a specific batch
  python scripts/log_specific_models.py --experiment real-estate-total-price-v1
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))
sys.path.insert(0, str(_BACKEND_ROOT / "model" / "src"))

from mlflow_settings import configure_mlflow
from config import EXPERIMENTS
from evaluate import pick_best_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("log_specific_models")


def get_latest_batch_id(experiment_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED' and tags.training_batch_id != ''",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No finished runs with a training_batch_id in '{experiment_name}'.")
    return runs.iloc[0]["tags.training_batch_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-id", help="training_batch_id to select from (default: latest per experiment)")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), help="only this experiment (default: all)")
    args = parser.parse_args()

    configure_mlflow()

    experiments = {args.experiment: EXPERIMENTS[args.experiment]} if args.experiment else EXPERIMENTS
    for exp_name, exp_vals in experiments.items():
        batch_id = args.batch_id or get_latest_batch_id(exp_name)
        registered_name = exp_vals["registered_model_name"]
        log.info("[%s] selecting best model from batch %s", exp_name, batch_id)
        version = pick_best_model(exp_name, registered_name, batch_id)
        log.info("[%s] Registered '%s' version: %s", exp_name, registered_name, version)


if __name__ == "__main__":
    main()