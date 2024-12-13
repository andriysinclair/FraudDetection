"""
Pipeline2 will decompose date features in hour, day, week, month, year..
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

date_pipeline = Pipeline(steps=[("Convert_to_dt", Date())])

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

general_transformation_pipeline2 = Pipeline(
    steps=[
        ("Return_reduced_df", Target0_Reducer(percentage=0.01)),
        ("Make_target_binary", TargetBinary(type="df")),
        ("Fill_NA", fillna_transformer),  # Add the fillna step
        ("Add_target_copy", target_copy_transformer),
        (
            "Numerical_and_date_transformations",
            ColumnTransformer(
                [
                    ("g", binary_pipeline, "has_chip"),
                    ("z", binary_pipeline, "card_on_dark_web"),  # Added missing comma
                    ("r", numerical_pipeline, "amount"),
                    ("s", numerical_pipeline, "credit_limit"),
                    ("t", date_pipeline, "date"),
                    ("u", date_pipeline, "expires"),
                    ("v", date_pipeline, "acct_open_date"),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_pipeline),
        (
            "date_hour",
            DateDecomposer(time_element_to_extract="hour", col_to_decomp="date"),
        ),
        (
            "date_dow",
            DateDecomposer(time_element_to_extract="dow", col_to_decomp="date"),
        ),
        (
            "date_month",
            DateDecomposer(time_element_to_extract="month", col_to_decomp="date"),
        ),
        (
            "date_year",
            DateDecomposer(time_element_to_extract="year", col_to_decomp="date"),
        ),
        (
            "expires_month",
            DateDecomposer(time_element_to_extract="month", col_to_decomp="expires"),
        ),
        (
            "expires_year",
            DateDecomposer(time_element_to_extract="year", col_to_decomp="expires"),
        ),
        (
            "acct_open_date_month",
            DateDecomposer(
                time_element_to_extract="month", col_to_decomp="acct_open_date"
            ),
        ),
        (
            "acct_open_date_year",
            DateDecomposer(
                time_element_to_extract="year", col_to_decomp="acct_open_date"
            ),
        ),
    ],
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

Pipeline_for_exploration2 = Pipeline(
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
                    ("t", target_encoder_pipeline, ["date_hour", "target"]),
                    ("u", target_encoder_pipeline, ["date_dow", "target"]),
                    ("v", target_encoder_pipeline, ["date_month", "target"]),
                    ("w", target_encoder_pipeline, ["date_year", "target"]),
                    ("x", target_encoder_pipeline, ["expires_month", "target"]),
                    ("y", target_encoder_pipeline, ["expires_year", "target"]),
                    ("z", target_encoder_pipeline, ["acct_open_date_month", "target"]),
                    ("za", target_encoder_pipeline, ["acct_open_date_year", "target"]),
                    ("r", numerical_scaling_pipeline, ["amount"]),
                    ("s", numerical_scaling_pipeline, ["credit_limit"]),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_pipeline),
    ]
)

# Function to drop is_fraud column for training set


def drop_target_col(X, target="is_fraud"):
    X = X.drop(columns=target)
    return X


drop_target_transformer = FunctionTransformer(drop_target_col, validate=False)

# Pipeline that removes uncorrelated features and can be used in modelling

Pipeline2 = Pipeline(
    steps=[
        # ("general_transformation_pipeline", general_transformation_pipeline),
        ("exploration_pipeline", Pipeline_for_exploration2),  # Existing pipeline
        ("remove_uncorrelated_features", remove_uncorr_features_pipeline),  # New step
        ("drop_target_transformer", drop_target_transformer),
    ]
)
