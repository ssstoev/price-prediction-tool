import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import logging as log
import numpy as np

def evaluate_model(X_test, y_test_log, trained_model):
    '''We evaluate the model both in log and normal(EUR) scale'''

    y_pred_log = trained_model.predict(X_test)
    test_mae_log  = mean_absolute_error(y_test_log, y_pred_log)
    test_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    test_r2_log   = r2_score(y_test_log, y_pred_log)

    y_test_eur = np.exp(y_test_log) 
    y_pred_eur = np.exp(y_pred_log)
    test_mae_eur  = mean_absolute_error(y_test_eur, y_pred_eur)
    test_rmse_eur = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))
    test_r2_eur   = r2_score(y_test_eur, y_pred_eur)

    eval_dict = {
        "test_mae_log": test_mae_log,
        "test_rmse": test_rmse_log,
        "test_r2": test_r2_log,
        "test_mae_eur": test_mae_eur,
        "test_rmse_eur": test_rmse_eur,
        "test_r2_eur": test_r2_eur,
    }
    
    return eval_dict

def pick_best_model(experiment_name: str, registered_model_name: str, training_batch_id: str, metric: str = "cv_rmse") -> str:
    uri = mlflow.get_tracking_uri()
    if not uri or uri.startswith("file://"):
        raise RuntimeError(
            f"MLflow is pointing at a local store ({uri!r}). "
            "Call configure_mlflow() before pick_best_model()."
        )
    log.info("pick_best_model: using tracking URI %s", uri.split("@")[-1])
    
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=(
                f"metrics.{metric} > 0 and attributes.status = 'FINISHED' "
                f"and tags.training_batch_id = '{training_batch_id}'"
            ),
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )
        if runs.empty:
            raise ValueError(
                f"No completed runs with metric '{metric}' found for experiment "
                f"'{experiment_name}' and training_batch_id '{training_batch_id}'."
            )

        best_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, name=registered_model_name)
        log.info("Registered '%s' version %s from run %s", registered_model_name, mv.version, best_run_id)
        return mv.version
    
    except mlflow.exceptions.MlflowException as e:
        log.error("MLflow registration failed: %s", e)
        raise