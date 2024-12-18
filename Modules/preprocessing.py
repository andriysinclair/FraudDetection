import pandas as pd
import numpy as np
import json
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline


def missing_summary(df):
    """missing_summary

    Calculates number and relative percentage of missing values in every column

    Args:
        df (pandas.DataFrame): Dataframe for which to calculate missing values

    Returns:
        pandas.DataFrame: DataFrame with missing values and relevant percentages for each column
    """
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_summary = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage missing (%)": missing_percentage}
    )

    return missing_summary


def find_unique_values(df):
    """find_unique_values

    Finds info about unique values in columns

    Args:
        df (pd.DataFrame): DataFrame of interest

    Returns:
        pd.DataFrame: DateFrame with information of unique values
    """

    total_len = len(df)
    columns = list(df.columns)
    unique_values = [df[cols].nunique() for cols in df.columns]
    unique_perc = [np.round(entry / total_len * 100, 2) for entry in unique_values]

    return pd.DataFrame(
        {"columns": columns, "unique_no": unique_values, "unique_%": unique_perc}
    )


def dollar_to_int(df):
    """dollar_to_int

    Turns all columns with entries in dollars into int type

    Args:
        df (pandas.DatFrame): Dataframe which to transform
    """
    for col in df.columns:
        if str(df.loc[0, col]).startswith("$"):
            df[col] = df[col].apply(
                lambda x: (
                    int(float(x[1:])) if isinstance(x, str) and x.startswith("$") else x
                )
            )
