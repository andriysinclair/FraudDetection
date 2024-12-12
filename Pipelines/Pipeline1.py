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

rename_pipeline = Pipeline(
    [
        ("rename_cols", rename_cols_transformer),
    ]
)

binary_pipeline = Pipeline(steps=[("Make_target_binary", TargetBinary(type="column"))])

numerical_pipeline = Pipeline(
    [
        ("dollar_to_int", DollarToInt()),
        # ("min_max_scaler", MinMaxScaler())
    ]
)

numerical_scaling_pipeline = Pipeline(
    [
        # ("dollar_to_int", DollarToInt()),
        ("min_max_scaler", MinMaxScaler())
    ]
)

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
                    ("g", binary_pipeline, "has_chip"),
                    ("z", binary_pipeline, "card_on_dark_web"),  # Added missing comma
                    ("r", numerical_pipeline, "amount"),
                    ("s", numerical_pipeline, "credit_limit"),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_pipeline),
    ]
)


numerical_scaling_pipeline = Pipeline(
    [
        # ("dollar_to_int", DollarToInt()),
        ("min_max_scaler", MinMaxScaler())
    ]
)

one_hot_pipeline = Pipeline(
    [
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

target_encoder_pipeline = Pipeline(
    [("Target_encoder", CustomTargetEncoder(target="target"))]
)


remove_uncorr_features_pipeline = Pipeline(
    [("Remove_uncorr_features", RemoveUncorrFeatures(p=0.05))]
)


## Target encoder Pipeline

Pipeline_for_exploration = Pipeline(
    steps=[
        # ("general_transformation_pipeline", general_transformation_pipeline),
        (
            "Encoder",
            ColumnTransformer(
                [
                    ("d", one_hot_pipeline, ["use_chip"]),
                    ("e", one_hot_pipeline, ["card_brand"]),
                    ("f", one_hot_pipeline, ["card_type"]),
                    ("h", target_encoder_pipeline, ["client_id", "target"]),
                    ("i", target_encoder_pipeline, ["card_id", "target"]),
                    ("j", target_encoder_pipeline, ["merchant_id", "target"]),
                    ("k", target_encoder_pipeline, ["merchant_city", "target"]),
                    ("l", target_encoder_pipeline, ["merchant_state", "target"]),
                    ("m", target_encoder_pipeline, ["zip", "target"]),
                    ("n", target_encoder_pipeline, ["mcc", "target"]),
                    ("o", target_encoder_pipeline, ["errors", "target"]),
                    ("p", target_encoder_pipeline, ["card_number", "target"]),
                    ("q", target_encoder_pipeline, ["cvv", "target"]),
                    ("r", numerical_scaling_pipeline, ["amount"]),
                    ("s", numerical_scaling_pipeline, ["credit_limit"]),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_pipeline),
    ]
)

# Pipeline that removes uncorrelated features and can be used in modelling

Pipeline1 = Pipeline(
    steps=[
        # ("general_transformation_pipeline", general_transformation_pipeline),
        ("exploration_pipeline", Pipeline_for_exploration),  # Existing pipeline
        ("remove_uncorrelated_features", remove_uncorr_features_pipeline),  # New step
    ]
)
