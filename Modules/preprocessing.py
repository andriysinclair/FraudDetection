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


def merge_dfs(transaction_data_df, cards_data_df, data_folder):
    """merge_dfs

    Merges card data with transaction data and export to pickle in the data folder

    Args:
        transaction_data_df (pandas.DataFrame): Dataframe for transaction data
        cards_data_df (pandas.DataFrame): Dataframe for card data
        data_folder (str): Absolute path to data folder
    """

    # Load in the fraud jsons
    with open(data_folder + "/train_fraud_labels.json", "r") as file:
        fraud_data = json.load(file)

    # Function to add a new column to the data set using the JSON

    # Making into a pandas series
    fraud_data = pd.DataFrame(fraud_data)

    # resetting index
    fraud_data.reset_index(inplace=True)

    # Renaming index to id for later merge
    fraud_data.rename(columns={"index": "id"}, inplace=True)

    # Changing id column to int type
    fraud_data["id"] = fraud_data["id"].astype(int)

    # Dropping client column in cards df as it is redunadant
    cards_data_df = cards_data_df.drop(columns=["client_id"])

    # Users data has no identifier to match with the transaction data
    # Merging card data with transaction data
    merged_df = transaction_data_df.merge(
        cards_data_df,
        how="inner",
        left_on="card_id",
        right_on="id",
        suffixes=("_T", "_C"),
    )

    # Adding target column to merged df by merging on id_T and id
    merged_df = merged_df.merge(
        fraud_data, how="inner", left_on="id_T", right_on="id", suffixes=("_T", "_C")
    )

    # Dropping id_C and id_T created from merge and equivalent to id (id of transaction)
    merged_df = merged_df.drop(columns=["id_C", "id_T"])

    # Saving merged_df to pickle
    merged_df.to_pickle(data_folder + "/merged_data.pkl")
