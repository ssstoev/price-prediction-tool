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
    return best_estimator

def run_trial(model_cfg, f_set_name, col_transformer_pipeline, X_train, X_test, y_train, y_test):

    try:
        log.info(f"Starting {model_cfg['run_name']} model")
        with mlflow.start_run(run_name=f"{model_cfg['run_name']}_{f_set_name}"):
            best_estimator = train_model(X_train, 
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

            # evaluate & log the metrics
            results = evaluate_model(X_test, y_test, best_estimator)
            mlflow.log_metrics(results)
            mlflow.sklearn.log_model(best_estimator, name="model")

            print(f"Test MAE: {results['test_mae']}")
            print(f"Test RMSE: {results['test_rmse']}")
            print(f"R2: {results['test_r2']} \n")

            return results
        
    except (Exception, KeyboardInterrupt) as e:
        log.warning(
            "Trial failed: %s | %s — %s",
            model_cfg["run_name"], f_set_name, e, exc_info=True
        )
        return None

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
            run_trial(model_cfg, f_set_name, col_transformer_pipeline, X_train, X_test, y_train, y_test)
    
    print("="*95)
    print(f"Experiment {exp_name} finished!")
    print("="*95)
