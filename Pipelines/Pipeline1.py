"""
Pipeline1 will be based on making time series from 0,...T of date features. Consequently this has little correlation
with the target an ends up being removed
"""

# LIbraries

# importing external libraries
from pathlib import Path
import sys
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(transform_output="pandas")

# Adding Modules to system Path

root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)

# Importing function to load data

from Modules.load_data import load_data
from Modules.preprocessing import (
    missing_summary,
    merge_dfs,
    dollar_to_int,
    find_unique_values,
)
from Modules.plotting import Plotter
from Modules.transforming import *

# Creating Pipeline

## Creating pipeline

# Define the fillna transformer
fillna_transformer = FunctionTransformer(lambda X: X.fillna(0))


# Define a function to copy the target column
def copy_target_column(X):
    X = X.copy()  # Ensure no modification to the original DataFrame
    X["is_fraud"] = X["target"]
    return X


target_copy_transformer = FunctionTransformer(copy_target_column, validate=False)

time_series_pipeline = Pipeline(
    steps=[("Convert_to_dt", Date()), ("ts_mapping", TimeSeriesMapper())]
)

# Function to rename columns


def rename_cols(X):
    X = X.copy()
    new_col_names = [col.split("__")[1] for col in list(X.columns)]
    X.columns = new_col_names
    return X


rename_cols_transformer = FunctionTransformer(rename_cols, validate=False)

general_transformation_pipeline = Pipeline(
    steps=[
        ("Return_reduced_df", Target0_Reducer(percentage=0.01)),
        ("Make_target_binary", TargetBinary(type="df")),
        ("Fill_NA", fillna_transformer),  # Add the fillna step
        ("Add_target_copy", target_copy_transformer),
        (
            "Numerical_and_date_transformations",
            ColumnTransformer(
                [
                    ("a", time_series_pipeline, "date"),
                    ("b", time_series_pipeline, "acct_open_date"),
                    ("t", time_series_pipeline, "expires"),
                    ("g", TargetBinary(type="column"), "has_chip"),
                    (
                        "z",
                        TargetBinary(type="column"),
                        "card_on_dark_web",
                    ),  # Added missing comma
                    ("r", DollarToInt(), "amount"),
                    ("s", DollarToInt(), "credit_limit"),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_cols_transformer),
    ]
)


## Target encoder Pipeline

Pipeline_for_exploration = Pipeline(
    steps=[
        # ("general_transformation_pipeline", general_transformation_pipeline),
        (
            "Encoder",
            ColumnTransformer(
                [
                    (
                        "d",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["use_chip"],
                    ),
                    (
                        "e",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["card_brand"],
                    ),
                    (
                        "f",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["card_type"],
                    ),
                    (
                        "h",
                        CustomTargetEncoder(target="target"),
                        ["client_id", "target"],
                    ),
                    ("i", CustomTargetEncoder(target="target"), ["card_id", "target"]),
                    (
                        "j",
                        CustomTargetEncoder(target="target"),
                        ["merchant_id", "target"],
                    ),
                    (
                        "k",
                        CustomTargetEncoder(target="target"),
                        ["merchant_city", "target"],
                    ),
                    (
                        "l",
                        CustomTargetEncoder(target="target"),
                        ["merchant_state", "target"],
                    ),
                    ("m", CustomTargetEncoder(target="target"), ["zip", "target"]),
                    ("n", CustomTargetEncoder(target="target"), ["mcc", "target"]),
                    ("o", CustomTargetEncoder(target="target"), ["errors", "target"]),
                    (
                        "p",
                        CustomTargetEncoder(target="target"),
                        ["card_number", "target"],
                    ),
                    ("q", CustomTargetEncoder(target="target"), ["cvv", "target"]),
                    ("r", MinMaxScaler(), ["amount"]),
                    ("s", MinMaxScaler(), ["credit_limit"]),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_cols_transformer),
    ]
)

# Function to drop is_fraud column for training set

# Pipeline that removes uncorrelated features and can be used in modelling

Pipeline1 = Pipeline(
    steps=[
        ("general_transformation_pipeline", general_transformation_pipeline),
        ("exploration_pipeline", Pipeline_for_exploration),  # Existing pipeline
        ("remove_uncorrelated_features", RemoveUncorrFeatures(p=0.05)),  # New step
    ]
)
