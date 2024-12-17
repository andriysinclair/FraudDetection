import pandas as pd
import numpy as np
import random
import yaml
from pathlib import Path
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Obtaining Root dir

root = str(Path(__file__).parent.parent)

# Obtaining seed from config.yaml

# Load the config file
with open(root + "/config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["global"]["seed"]

# print(f"seed: {seed}")

# Set global seeds for reproducibility
random.seed(seed)
np.random.seed(seed)

# Use the seed in scikit-learn
random_state = check_random_state(seed)

# Functions to use in Pipeline


class TargetBinary(BaseEstimator, TransformerMixin):
    """TargetBinary

    Converts feature with 'Yes', 'No' into 1,0.

    """

    def __init__(self, type):
        """__init__

        Args:
            type (str): tales input 'df' or 'column' depending on item you desire to trasnform
        """
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
            X = X.copy()
            X["target"] = X["target"].str.lower().map({"yes": 1, "no": 0})
            return X

        elif self.type == "column":
            X = X.copy()
            X = X.str.lower().map({"yes": 1, "no": 0}).to_frame()
            return X

        else:
            TypeError("Only accepts inputs of type 'column' or 'df'")


class Date(BaseEstimator, TransformerMixin):
    """Date

    Converts object to datetime64[ns]

    """

    def __init__(self, format="mixed"):
        """
        Args:
            format (str): Define the format of date elements, default to mixed
        """

        self.format = format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """transform_date

        Transforms date column into datetime and extracts relevant components: hour, day etc.

        Args:
            X (pd.Series): Contains dates

        Returns:
            pd.Series: Converted column of datetime64[ns]
        """

        return pd.to_datetime(X, format=self.format).to_frame()


class DateDecomposer:
    def __init__(self, time_element_to_extract, col_to_decomp):
        self.time_element_to_extract = time_element_to_extract
        self.col_to_decomp = col_to_decomp

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.time_element_to_extract == "hour":
            X[self.col_to_decomp + "_" + self.time_element_to_extract] = X[
                self.col_to_decomp
            ].dt.hour
        elif self.time_element_to_extract == "dow":
            X[self.col_to_decomp + "_" + self.time_element_to_extract] = X[
                self.col_to_decomp
            ].dt.weekday
        elif self.time_element_to_extract == "month":
            X[self.col_to_decomp + "_" + self.time_element_to_extract] = X[
                self.col_to_decomp
            ].dt.month
        elif self.time_element_to_extract == "year":
            X[self.col_to_decomp + "_" + self.time_element_to_extract] = X[
                self.col_to_decomp
            ].dt.year
        else:
            raise KeyError(
                "time_element_to_extract must be one of: 'hour', 'dow', 'month' or 'year'"
            )

        return X


class Target0_Reducer(BaseEstimator, TransformerMixin):
    def __init__(self, percentage=0.01, balanced=False):
        """
        Args:
            percentage (float): percentage of 0 responses to keep
        """
        if not (0 < percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        self.percentage = percentage
        self.balanced = balanced

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

        if not self.balanced:

            # Randomly select x% of indices
            num_to_select = int(len(X_0) * self.percentage)
            selected_indices = random.sample(X_0_index, num_to_select)

            X_0_reduced = X_0.loc[selected_indices]

            X_1 = X[X["target"] != "No"]

            combined = pd.concat([X_0_reduced, X_1], axis=0).reset_index(drop=True)

            # Shuffle the combined DataFrame
            shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

            return shuffled

        # If balanced df required

        else:

            X_1 = X[X["target"] != "No"]

            # Randomly select equal number of nos
            num_to_select = int(len(X_1))
            selected_indices = random.sample(X_0_index, num_to_select)

            X_0_reduced = X_0.loc[selected_indices]

            combined = pd.concat([X_0_reduced, X_1], axis=0).reset_index(drop=True)

            # Shuffle the combined DataFrame
            shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

            return shuffled


class CustomTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, target):
        self.target_mapping = None
        self.target = target
        self.default_value = None

    def fit(self, X, y=None):
        # Ensuring X has only width of 2 and has target as 2nd column
        if X.shape[1] != 2 or X.columns[1] != self.target:
            raise ValueError(
                f"Input must have two columns in the order [column_to_target_transform, {self.target}]"
            )

        # Compute and store mapping
        self.target_mapping = X.groupby(X.iloc[:, 0])[self.target].mean().to_dict()

        # Store default value for future missing categories
        self.default_value = X.iloc[:, 1].mean()
        return self

    def transform(self, X):
        # Check if encoder has been fitted
        if self.target_mapping is None:
            raise ValueError("The encoder has not been fitted yet.")

        # Apply the stored mapping to the column.
        X = X.copy()
        X_transformed = X.iloc[:, 0].map(self.target_mapping).to_frame()

        # For classes not present in the training set assign the global mean
        X_transformed = X_transformed.fillna(self.default_value)
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


class RemoveUncorrFeatures:
    def __init__(self, p, target="is_fraud"):
        self.mapping = None
        self.p = p
        self.target = target

    def fit(self, X, y=None):

        ## Function to remove features with less than X correlation
        suitable_cols = X.corr()[self.target][
            np.abs(X.corr()[self.target].values) > self.p
        ]
        suitable_cols = list(suitable_cols.index)

        self.mapping = suitable_cols

        return self

    def transform(self, X, y=None):
        # Check if encoder has been fitted
        if self.mapping is None:
            raise ValueError("The encoder has not been fitted yet.")

        # Apply the stored mapping to the column.
        X = X.copy()
        X_transformed = X[self.mapping]

        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
