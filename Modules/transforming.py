import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Functions to use in Pipeline


class TargetBinary(BaseEstimator, TransformerMixin):

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
        return X.map({"Yes": 1, "No": 0}).to_frame()


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


class DateComponentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, component):
        self.component = component

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Extracts a specific component (e.g., hour, day of week) from a datetime column.
        """
        # Making X into datetime
        X = pd.to_datetime(X)

        if self.component == "hour":
            return X.dt.hour.to_frame()
        elif self.component == "dow":
            return X.dt.day_name().to_frame()
        elif self.component == "month":
            return X.dt.month.to_frame()
        elif self.component == "year":
            return X.dt.year.to_frame()
        else:
            raise ValueError(f"Unsupported component: {self.component}")


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
        return X.map({"Yes": 1, "No": 0}).to_frame()
