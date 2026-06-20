from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import mlflow
import mlflow.sklearn

from config import CV_FOLDS, FEATURE_SETS, MODELS
from evaluate import evaluate_model
from features import build_column_transformer_pipeline, build_target, build_train_test_data

import logging as log
log.getLogger("mlflow.sklearn").setLevel(log.ERROR)

def train_model(X_train, y_train, model, params, col_transformer, cv_folds=5):
    pipeline = Pipeline([
        ("preprocessor", clone(col_transformer)),
        ("model", clone(model))
    ])

    grid_search_cv = GridSearchCV(estimator=pipeline,
                              param_grid=params,
                              cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                              scoring="neg_root_mean_squared_error",
                              refit=True)
    
    grid_search_cv.fit(X_train, y_train)

    best_estimator = grid_search_cv.best_estimator_
    # scoring is neg_root_mean_squared_error, so convert to positive RMSE.
    cv_rmse = -grid_search_cv.best_score_
    return best_estimator, cv_rmse

def run_trial(model_cfg, f_set_name, col_transformer_pipeline, X_train, y_train):

    try:
        log.info(f"Starting {model_cfg['run_name']} model")
        with mlflow.start_run(run_name=f"{model_cfg['run_name']}_{f_set_name}"):
            best_estimator, cv_rmse = train_model(X_train, 
                                                  y_train,
                                                  model=model_cfg["estimator"], 
                                                  params=model_cfg["param_grid"], 
                                                  col_transformer=col_transformer_pipeline,
                                                  cv_folds=CV_FOLDS)
            
            grid_param_keys = [k.removeprefix("model__") for k in model_cfg["param_grid"]]
            best_model_params = best_estimator.named_steps["model"].get_params()
            best_grid_params = {k: best_model_params[k] for k in grid_param_keys if k in best_model_params}
            param_suffix = "_".join(f"{k}={v}" for k, v in best_grid_params.items())
            mlflow.set_tag("model_type", model_cfg['run_name'])
            mlflow.set_tag("feature_set", f_set_name)
            mlflow.set_tag("best_params_summary", param_suffix)
            mlflow.log_params(best_grid_params)
            mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.sklearn.log_model(best_estimator, name="model")
            print(f"CV RMSE: {cv_rmse}")
            return cv_rmse
        
    except (Exception, KeyboardInterrupt) as e:
        log.warning(
            "Trial failed: %s | %s — %s",
            model_cfg["run_name"], f_set_name, e, exc_info=True
        )
        return None


def get_best_cv_run(experiment_name: str, metric: str = "cv_rmse"):
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"metrics.{metric} > 0 and attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"No completed runs with metric '{metric}' found for experiment '{experiment_name}'."
        )
    return runs.iloc[0]


# WIP: save the actual test values, predictions and errors (in order to perform analyses without needing to re-predict)
def log_final_test_eval(exp_name, target_transformed, df):
    best = get_best_cv_run(exp_name)
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
        mlflow.set_tag("selected_run_id", best_run_id)
        mlflow.set_tag("selected_feature_set", best_feature_set)
        mlflow.set_tag("selected_model_type", best.get("tags.model_type", "unknown"))
        mlflow.log_metric("selected_cv_rmse", float(best["metrics.cv_rmse"]))
        mlflow.log_metrics(test_metrics)

    print("-" * 95)
    print(f"Final test evaluation for '{exp_name}' (selected run: {best_run_id})")
    print(f"Selected feature set: {best_feature_set}")
    print(f"CV RMSE (selection metric): {float(best['metrics.cv_rmse'])}")
    print(f"Test MAE: {test_metrics['test_mae']}")
    print(f"Test RMSE: {test_metrics['test_rmse']}")
    print(f"Test R2: {test_metrics['test_r2']}")
    print("-" * 95)

def run_experiment(exp_name, exp_values, df):
    mlflow.set_experiment(experiment_name=exp_name)
    print("="*95)
    print(f"Starting experiment {exp_name}...")
    print("="*95)
    target_col = exp_values["target_col"]
    print(f"Target column is {target_col}")

    df, target_transformed = build_target(df, target_col=target_col)

    for f_set_name, f_set_values in FEATURE_SETS.items():
        print(f"currently using {f_set_name} as a feature set")
        X_train, X_test, y_train, y_test = build_train_test_data(df,
                                                                target_col=target_transformed,
                                                                feature_set=f_set_values)
        
        col_transformer_pipeline = build_column_transformer_pipeline(feature_cols=f_set_values)

        # train model with specific feature set & params
        for model_cfg in MODELS:
            run_trial(model_cfg, f_set_name, col_transformer_pipeline, X_train, y_train)

    # One-time holdout evaluation after model selection by CV.
    log_final_test_eval(exp_name, target_transformed, df)
    
    print("="*95)
    print(f"Experiment {exp_name} finished!")
    print("="*95)
