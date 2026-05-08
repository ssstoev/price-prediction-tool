from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# We experiment for different target variables
EXPERIMENTS = {
    "real-estate-total-price-v2": {
        "target_col": "total_price_eur",
        "target_transform": "log",
        "drop_before_features": [],
        "registered_model_name": "RealEstateTotalPrice",
    },

    "real-estate-price-per-m2-v2": {
        "target_col": "price_m2_eur",
        "target_transform": "log",
        "drop_before_features": [],
        "registered_model_name": "RealEstatePricePerSqm",
    },
}

# we experiment with different feature sets
FEATURE_SETS = {
    "size_only":          ["size_m2"],
    "neighbourhood_only": ["neighbourhood"],
    "size_neighbourhood": ["size_m2", "neighbourhood"],
    "no_size_no_neighbourhood": ["nr_of_rooms", "total_floors", "appartment_floor",
                                  "is_first_floor", "is_last_floor", "includes_parking",
                                  "near_public_transport", "furnished", "new_building", "akt16"],
    "all":                ["size_m2", "nr_of_rooms", "total_floors", "appartment_floor",
                           "neighbourhood", "is_first_floor", "is_last_floor", "includes_parking",
                           "near_public_transport", "furnished", "new_building", "akt16"],
}

# DEV_MODE: small grids and fewer CV folds for fast iteration. Set to False for full search.
DEV_MODE = True
CV_FOLDS = 2 if DEV_MODE else 5

MODELS = [
    {
        "run_name": "XGBoost",
        "estimator": XGBRegressor(objective="reg:squarederror", random_state=42, nthread=1, tree_method="hist"),
        "param_grid": {
            "model__n_estimators": [20] if DEV_MODE else [10, 20, 50],
            "model__max_depth": [5, 10] if DEV_MODE else [5, 10, 15],
            "model__learning_rate": [0.1] if DEV_MODE else [0.01, 0.1, 0.2],
            "model__colsample_bytree": [1] if DEV_MODE else [0.8, 1],
        },
    },
    {
        "run_name": "DecisionTreeRegressor",
        "estimator": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "model__max_depth": [3] if DEV_MODE else [5, 10, 20],
            "model__min_samples_leaf": [10] if DEV_MODE else [5, 10, 20],
        },
    },
    {
        "run_name": "RandomForestRegressor",
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [80] if DEV_MODE else [20, 80, 160],
            "model__max_depth": [5, 10] if DEV_MODE else [5, 10, 20],
            "model__min_samples_leaf": [10] if DEV_MODE else [5, 10, 20],
        },
    },
]

NUMERIC = {"size_m2", "nr_of_rooms", "total_floors", "appartment_floor"}
CATEGORICAL = {"neighbourhood"}
BOOLEAN = {"is_first_floor", "is_last_floor", "includes_parking", "near_public_transport", "furnished", 
                        "new_building", "akt16"}
