import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import logging as log

def evaluate_model(X_test, y_test, trained_model):

    y_pred = trained_model.predict(X_test)

    test_mae  = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2   = r2_score(y_test, y_pred)

    eval_dict = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }
    return eval_dict

def pick_best_model(experiment_name: str, registered_model_name: str, metric: str = "cv_rmse") -> str:
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"metrics.{metric} > 0 and attributes.status = 'FINISHED'",
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )
        if runs.empty:
            raise ValueError(
                f"No completed runs with metric '{metric}' found for experiment '{experiment_name}'."
            )

        best_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, name=registered_model_name)
        log.info("Registered '%s' version %s from run %s", registered_model_name, mv.version, best_run_id)
        return mv.version
    
    except mlflow.exceptions.MlflowException as e:
        log.error("MLflow registration failed: %s", e)
        raise