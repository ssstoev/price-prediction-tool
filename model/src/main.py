import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import mlflow
# from pathlib import Path
from config import EXPERIMENTS
from data_load import load_data
from evaluate import pick_best_model
from train import run_experiment
import logging
from dotenv import load_dotenv

load_dotenv()

# SUPABASE_HOST     = os.getenv("SUPABASE_HOST")
# SUPABASE_PORT     = int(os.getenv("SUPABASE_PORT", 5432))
# SUPABASE_DB       = os.getenv("SUPABASE_DB")
# SUPABASE_USER     = os.getenv("SUPABASE_USER")
# SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

# MLFLOW_TRACKING_URI = (
#     f"postgresql+psycopg2://{SUPABASE_USER}:{SUPABASE_PASSWORD}"
#     f"@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}"
#     f"?options=-csearch_path%3Dmlflow"
# )

mlflow.set_tracking_uri("http://localhost:5000")

# setup basic log messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    df = load_data()
    for exp_name, exp_vals in EXPERIMENTS.items():
        run_experiment(exp_name, exp_vals, df)

    for exp_name, exp_vals in EXPERIMENTS.items():
        registered_name = exp_vals["registered_model_name"]
        version = pick_best_model(exp_name, registered_name)
        print(f"[{exp_name}] Registered '{registered_name}' version: {version}")


if __name__ == "__main__":
    main()