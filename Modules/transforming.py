import pandas as pd
import numpy as np
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


class FeatureUnionToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame(X)
