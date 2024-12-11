import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Functions to use in Pipeline


class TargetBinary(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms {"Yes", "No"} in the target column to binary {1, 0}.

        Args:
            X (pd.Series or pd.DataFrame): Input target data.

        Returns:
            pd.Series: Binary transformed target column.
        """
        if self.type == "df":
            X["target"] = X["target"].map({"yes": 1, "no": 0})
            return X

        if self.type == "column":
            X = X.str.lower().map({"yes": 1, "no": 0}).to_frame()
            return X


class Date(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """transform_date

        Transforms date column into datetime and extracts relevant components: hour, day etc.

        Args:
            X (pd.Series): Contains dates

        Returns:
            dict: A dictionary of date column decomposed into components
        """

        return pd.to_datetime(X).to_frame()


class Target0_Reducer(BaseEstimator, TransformerMixin):
    def __init__(self, percentage):
        """
        Args:
            percentage (float): percentage of 0 responses to keep
        """
        if not (0 < percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        self.percentage = percentage

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """transform_date

        Transforms date column into datetime and extracts relevant components: hour, day etc.

        Args:
            X (pd.Series): Contains date

        Returns:
            dict: A dictionary of date column decomposed into components
        """
        if "target" not in X.columns:
            raise KeyError("The input DataFrame must have a 'target' column")

        # Validate the contents of the target column
        unique_values = set(X["target"].unique())
        if not unique_values.issubset({"Yes", "No"}):
            raise ValueError(
                f"The 'target' column contains invalid values: {unique_values - {'Yes', 'No'}}. "
                "It should only contain 'Yes' and 'No'."
            )

        # Find all No responses for fraud
        X_0 = X[X["target"] == "No"]

        # Obtain their indices
        X_0_index = list(X_0.index)

        # Randomly select x% of indices
        num_to_select = int(len(X_0) * self.percentage)
        selected_indices = random.sample(X_0_index, num_to_select)

        X_0_reduced = X_0.loc[selected_indices]

        X_1 = X[X["target"] != "No"]

        return pd.concat([X_0_reduced, X_1], axis=0).reset_index(drop=True)


class CustomTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, target):
        self.target_mapping = None
        self.target = target

    def fit(self, X, y=None):
        # Ensuring X has only width of 2 and has target as 2nd column
        if X.shape[1] != 2 or X.columns[1] != self.target:
            raise ValueError(
                f"Input must have two columns in the order [column_to_target_transform, {self.target}]"
            )

        # Compute and store mapping
        self.target_mapping = X.groupby(X.iloc[:, 0])[self.target].mean().to_dict()
        return self

    def transform(self, X):
        # Check if encoder has been fitted
        if self.target_mapping is None:
            raise ValueError("The encoder has not been fitted yet.")

        # Apply the stored mapping to the column.
        X = X.copy()
        X_transformed = X.iloc[:, 0].map(self.target_mapping).to_frame()
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class DollarToInt(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.apply(
            lambda x: (
                int(float(x[1:])) if isinstance(x, str) and x.startswith("$") else x
            )
        )

        return X.to_frame()


class TimeSeriesMapper:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X must be a date time
        X = pd.Series(X.iloc[:, 0])
        date_values = list(X.sort_values().values)
        date_ts_mapping = {date_value: i for i, date_value in enumerate(date_values)}
        X = X.map(date_ts_mapping)
        return X.to_frame()
