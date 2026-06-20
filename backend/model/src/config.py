import os
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

    "no_size_no_neighbourhood": ["nr_of_rooms", "floor", "building_total_floors",
                                  "is_first_floor", "is_last_floor", "is_furnished", "near_public_transport"],
    "all":                ["size_m2", "nr_of_rooms", "floor","building_total_floors",
                           "neighbourhood", "is_first_floor", "is_last_floor","is_furnished", "near_public_transport"],
    # "no_size_no_neighbourhood": ["nr_of_rooms", "total_floors", "appartment_floor",
    #                               "is_first_floor", "is_last_floor", "includes_parking",
    #                               "near_public_transport", "furnished", "new_building", "akt16"],
    # "all":                ["size_m2", "nr_of_rooms", "total_floors", "appartment_floor",
    #                        "neighbourhood", "is_first_floor", "is_last_floor", "includes_parking",
    #                        "near_public_transport", "furnished", "new_building", "akt16"],
}

# DEV_MODE: small grids and fewer CV folds for fast iteration. Set to False for full search.
DEV_MODE = os.environ.get("DEV_MODE", "true").lower() == "true"
CV_FOLDS = 2 if DEV_MODE else 5

MODELS = [
    {
        "run_name": "XGBoost",
        "estimator": XGBRegressor(objective="reg:squarederror", random_state=42, nthread=1, tree_method="hist"),
        "param_grid": {
            "model__n_estimators": [20] if DEV_MODE else [20, 40, 80, 160],
            "model__max_depth": [5, 10] if DEV_MODE else [10, 20, 40, 80],
            "model__learning_rate": [0.1] if DEV_MODE else [0.1, 0.2, 0.4, 0.8],
            "model__colsample_bytree": [1] if DEV_MODE else [1, 1.2, 1.6, 3.2],
        },
    },
    {
        "run_name": "DecisionTreeRegressor",
        "estimator": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "model__max_depth": [3] if DEV_MODE else [10, 20, 40, 80],
            "model__min_samples_leaf": [10] if DEV_MODE else [10, 20, 40, 80],
        },
    },
    {
        "run_name": "RandomForestRegressor",
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [80] if DEV_MODE else [80, 160, 320, 640],
            "model__max_depth": [5, 10] if DEV_MODE else [10, 20, 40, 80],
            "model__min_samples_leaf": [10] if DEV_MODE else [10, 20, 40, 80],
        },
    },
]

NUMERIC = {"size_m2", "nr_of_rooms", "floor", "building_total_floors", "appartment_floor"}
CATEGORICAL = {"neighbourhood"}
BOOLEAN = {"is_first_floor", "is_last_floor", "near_public_transport", "is_furnished"}
