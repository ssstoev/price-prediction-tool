import datetime as dt
import numpy as np
from uuid import uuid4
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import mlflow
import mlflow.sklearn

from config import CV_FOLDS, FEATURE_SETS, MODELS
from evaluate import evaluate_model
from features import build_column_transformer_pipeline, build_target, build_train_test_data
from mlflow_settings import get_mlflow_artifact_root

import logging
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

def generate_training_batch_id():
    '''Generate a unique ID for a specific training run. This is used to differentiate between training runs.
    It captures the entire pipeline run.'''
    # generate a specific training_batch_id to differentiate between pipeline runs
    training_batch_id = (
        f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )

    return training_batch_id

def _rmse_eur(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred))))

def _mae_eur(y_true, y_pred):
    return float(mean_absolute_error(np.exp(y_true), np.exp(y_pred)))

# The target is log-transformed, so EUR metrics need exp() on both y_true and y_pred.
CV_SCORERS = {
    "rmse_log": "neg_root_mean_squared_error",
    "rmse_eur": make_scorer(_rmse_eur, greater_is_better=False),
    "mae_eur": make_scorer(_mae_eur, greater_is_better=False),
}

def train_model(X_train, y_train, model, params, col_transformer, cv_folds=5):
    pipeline = Pipeline([
        ("preprocessor", clone(col_transformer)),
        ("model", clone(model))
    ])

    grid_search_cv = GridSearchCV(
                            estimator=pipeline,
                            param_grid=params,
                            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                            scoring=CV_SCORERS,
                            # best params are still selected on the log-scale RMSE
                            refit="rmse_log"
                            )
    
    grid_search_cv.fit(X_train, y_train)

    # These are the best params (based on lowest avg rmse of all folds for all param combos)
    best_estimator = grid_search_cv.best_estimator_
    # scorers are negated (greater is better), so flip the sign back to get positive errors.

    # Note: for the winning param combo, each value is the AVG across all folds
    idx = grid_search_cv.best_index_
    results = grid_search_cv.cv_results_
    cv_rmse = -grid_search_cv.best_score_
    cv_metrics = {
        "cv_rmse_eur": -results["mean_test_rmse_eur"][idx],
        "cv_mae_eur": -results["mean_test_mae_eur"][idx],
    }
    return best_estimator, cv_rmse, cv_metrics

def run_trial(training_batch_id, model_cfg, f_set_name, col_transformer_pipeline, X_train, y_train, data_tags=None):
    '''Train a specific model with specific feature_sets'''
    try:
        log.info(f"Starting {model_cfg['run_name']} model")
        with mlflow.start_run(run_name=f"{model_cfg['run_name']}_{f_set_name}"):
            best_estimator, cv_rmse, cv_metrics = train_model(X_train, 
                                                  y_train,
                                                  model=model_cfg["estimator"], 
                                                  params=model_cfg["param_grid"], 
                                                  col_transformer=col_transformer_pipeline,
                                                  cv_folds=CV_FOLDS)
            
            grid_param_keys = [k.removeprefix("model__") for k in model_cfg["param_grid"]]
            best_model_params = best_estimator.named_steps["model"].get_params()
            best_grid_params = {k: best_model_params[k] for k in grid_param_keys if k in best_model_params}
            param_suffix = "_".join(f"{k}={v}" for k, v in best_grid_params.items())
            if data_tags:
                mlflow.set_tags(data_tags)
            mlflow.set_tag("training_batch_id", training_batch_id)
            mlflow.set_tag("model_type", model_cfg['run_name'])
            mlflow.set_tag("feature_set", f_set_name)
            mlflow.set_tag("best_params_summary", param_suffix)
            mlflow.log_params(best_grid_params)
            mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metrics(cv_metrics)
            mlflow.sklearn.log_model(best_estimator, name="model")
            log.info("AVG CV RMSE: %s", cv_rmse)
            return cv_rmse
        
    except (Exception, KeyboardInterrupt) as e:
        log.warning(
            "Trial failed: %s | %s — %s",
            model_cfg["run_name"], f_set_name, e, exc_info=True
        )
        return None


def get_best_cv_run(experiment_name: str, training_batch_id: str, metric: str = "cv_rmse"):
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=(
            f"metrics.{metric} > 0 and attributes.status = 'FINISHED' "
            f"and tags.training_batch_id = '{training_batch_id}'"
        ),
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"No completed runs with metric '{metric}' found for experiment "
            f"'{experiment_name}' and training_batch_id '{training_batch_id}'."
        )
    return runs.iloc[0]


# WIP: save the actual test values, predictions and errors (in order to perform analyses without needing to re-predict)
def log_final_test_eval(exp_name, target_transformed, df, training_batch_id):
    best = get_best_cv_run(exp_name, training_batch_id)
    best_run_id = best.run_id
    best_feature_set = best["tags.feature_set"]

    if best_feature_set not in FEATURE_SETS:
        raise KeyError(
            f"Best run feature set '{best_feature_set}' is not found in FEATURE_SETS."
        )

    selected_features = FEATURE_SETS[best_feature_set]
    _, X_test, _, y_test = build_train_test_data(
        df,
        target_col=target_transformed,
        feature_set=selected_features,
    )

    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    test_metrics = evaluate_model(X_test, y_test, best_model)

    with mlflow.start_run(run_name=f"FINAL_TEST_{exp_name}"):
        mlflow.set_tag("stage", "final_test_eval")
        mlflow.set_tag("training_batch_id", training_batch_id)
        mlflow.set_tag("selected_run_id", best_run_id)
        mlflow.set_tag("selected_feature_set", best_feature_set)
        mlflow.set_tag("selected_model_type", best.get("tags.model_type", "unknown"))
        mlflow.log_metric("selected_cv_rmse", float(best["metrics.cv_rmse"]))
        mlflow.log_metrics(test_metrics)

    log.info(
        "Final test evaluation for '%s' (selected run: %s) | feature set: %s | "
        "CV RMSE: %s | Test MAE: %s | Test RMSE: %s | Test R2: %s",
        exp_name, best_run_id, best_feature_set, float(best["metrics.cv_rmse"]),
        test_metrics["test_mae_log"], test_metrics["test_rmse"], test_metrics["test_r2"],
    )

def run_experiment(training_batch_id, exp_name, exp_values, df, data_tags=None):
    '''We have 1 experiment per target feature - total price & price per sqm'''

    # fetch and check if artifact storage path exists
    artifact_root = get_mlflow_artifact_root()
    if artifact_root:
        client = mlflow.MlflowClient()
        # if it doesn't exist, create experiment with custom artifact storage path
        # if we don't include this mlflow will create the storage to its default location (ignoring custom path) 
        if client.get_experiment_by_name(exp_name) is None:
            client.create_experiment(exp_name, artifact_location=f"{artifact_root}/{exp_name}")

    # activate the experiment as context for the subsequent runs
    # it says "log in this experiment" 
    mlflow.set_experiment(experiment_name=exp_name)
    log.info("\n Starting experiment %s (batch %s)", exp_name, training_batch_id)
    target_col = exp_values["target_col"]
    log.info("Target column is %s", target_col)

    df, target_transformed = build_target(df, target_col=target_col)

    for f_set_name, f_set_values in FEATURE_SETS.items():
        log.info("Currently using %s as a feature set", f_set_name)
        X_train, X_test, y_train, y_test = build_train_test_data(df,
                                                                target_col=target_transformed,
                                                                feature_set=f_set_values)
        
        col_transformer_pipeline = build_column_transformer_pipeline(feature_cols=f_set_values)

        # train model with specific feature set & params
        for model_cfg in MODELS:
            run_trial(training_batch_id, model_cfg, f_set_name, col_transformer_pipeline,
                       X_train, y_train, data_tags)

    # One-time holdout evaluation after model selection by CV.
    log_final_test_eval(exp_name, target_transformed, df, training_batch_id)
    
    log.info("Experiment %s finished", exp_name)
