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

# Function to rename columns


def rename_cols(X):
    X = X.copy()
    new_col_names = [col.split("__")[1] for col in list(X.columns)]
    X.columns = new_col_names
    return X


rename_cols_transformer = FunctionTransformer(rename_cols, validate=False)

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
                    ("g", TargetBinary(type="column"), "has_chip"),
                    (
                        "z",
                        TargetBinary(type="column"),
                        "card_on_dark_web",
                    ),  # Added missing comma
                    ("r", DollarToInt(), "amount"),
                    ("s", DollarToInt(), "credit_limit"),
                    ("t", Date(), "date"),
                    ("u", Date(), "expires"),
                    ("v", Date(), "acct_open_date"),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_cols_transformer),
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

## Target encoder Pipeline

Pipeline_for_exploration2 = Pipeline(
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
                    (
                        "t",
                        CustomTargetEncoder(target="target"),
                        ["date_hour", "target"],
                    ),
                    ("u", CustomTargetEncoder(target="target"), ["date_dow", "target"]),
                    (
                        "v",
                        CustomTargetEncoder(target="target"),
                        ["date_month", "target"],
                    ),
                    (
                        "w",
                        CustomTargetEncoder(target="target"),
                        ["date_year", "target"],
                    ),
                    (
                        "x",
                        CustomTargetEncoder(target="target"),
                        ["expires_month", "target"],
                    ),
                    (
                        "y",
                        CustomTargetEncoder(target="target"),
                        ["expires_year", "target"],
                    ),
                    (
                        "z",
                        CustomTargetEncoder(target="target"),
                        ["acct_open_date_month", "target"],
                    ),
                    (
                        "za",
                        CustomTargetEncoder(target="target"),
                        ["acct_open_date_year", "target"],
                    ),
                    ("r", MinMaxScaler(), ["amount"]),
                    ("s", MinMaxScaler(), ["credit_limit"]),
                ],
                remainder="passthrough",
            ),
        ),
        ("Rename_columns", rename_cols_transformer),
    ]
)

# Pipeline that removes uncorrelated features and can be used in modelling

Pipeline2 = Pipeline(
    steps=[
        ("general_transformation_pipeline", general_transformation_pipeline2),
        ("exploration_pipeline", Pipeline_for_exploration2),  # Existing pipeline
        ("remove_uncorrelated_features", RemoveUncorrFeatures(p=0.05)),  # New step
    ]
)
