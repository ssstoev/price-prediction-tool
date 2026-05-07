'''Data preparation for ML model training'''

# Define the preprocessing pipeline
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

from config import BOOLEAN, CATEGORICAL, NUMERIC

def build_target(df: pd.DataFrame, target_col: str):
    """
    Compute derived target columns and return df + y.
    Peform a log transformation to the target col.
    Args:
        df (pd.DataFrame): the dataframe
        target_col (str): the name of the target column.

    Returns:
        Tuple of cleaned dataframe and log transformation of target column
    """

    df = df.copy()
    df = df[np.isfinite(df[target_col]) & (df[target_col] > 0)]
    df[f"log_{target_col}"] = np.log(df[target_col])

    target_transformed = f"log_{target_col}"
    return df, target_transformed

def build_train_test_data(df: pd.DataFrame, target_col: str, feature_set: list[str], test_size: float = 0.2):
    '''Perform train test split'''
    X = df[feature_set]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def build_column_transformer_pipeline(feature_cols: list) -> Pipeline:
    '''
    Apply transformation techniques to each feature depending on its dtype.
    Returns a ColumnTransformer Pipeline.
    '''
    num  = [f for f in feature_cols if f in NUMERIC]
    cat  = [f for f in feature_cols if f in CATEGORICAL]
    bool_ = [f for f in feature_cols if f in BOOLEAN]

    transformers = []
    if num:   transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num))
    if cat:   transformers.append(("cat", TargetEncoder(), cat))
    if bool_: transformers.append(("bool", SimpleImputer(strategy="most_frequent"), bool_))

    return ColumnTransformer(transformers)