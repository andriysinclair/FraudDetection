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
            type (str): tales input 'df' or 'column' depending on item you desire to transform
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
    """

    Decomposes date-like features into more granular components: hours, day of week, month etc.
    """

    def __init__(self, time_element_to_extract, col_to_decomp):
        """

        Args:
            time_element_to_extract (str): time element to extract, one of: ['hour', 'dow', 'month', 'year']
            col_to_decomp (str): date-like feature to decomp
        """
        self.time_element_to_extract = time_element_to_extract
        self.col_to_decomp = col_to_decomp

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Args:
            X (pd.Series): date-like feature

        Raises:
            KeyError: if time_element_to_extract is not one of: ['hour', 'dow', 'month', 'year']

        Returns:
            pd.Series: Extracted time component
        """
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
    """
    As the original dataset is very unbalanced. Only ~13,000 fraudulent transactions out of ~8,000,000. This transformer
    Randomly selects a percentage of non-fraudulent transactions and appends them to the fraudulent transactions.
    """

    def __init__(self, percentage=0.01, balanced=False, random_state=seed):
        """

        Args:
            percentage (float, optional): percentage of non-fraudulent samples to keep. Defaults to 0.01.
            balanced (bool, optional): Obtain an equal number of non-fraudulent samples as fraudulent samples. Defaults to False.
            random_state (int, optional): random seed. Defaults to seed.

        Raises:
            ValueError: if a non-valid percentage is defined
        """
        if not (0 < percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        self.percentage = percentage
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Args:
            X (pd.DataFrame): Dataframe to reduce

        Raises:
            KeyError: dataframe does not have a target column
            ValueError: target column does not contain unique values: 'yes', 'no'

        Returns:
            pd.DataFrame: Reduced dataframe
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

        random.seed(self.random_state)

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
            shuffled = combined.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            return shuffled


class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """

    Custom target encoding. Provides the target mean of the training set for each category in that column

    """

    def __init__(self, target):
        """
        Args:
            target (str): target column
        """
        self.target_mapping = None
        self.target = target
        self.default_value = None

    def fit(self, X, y=None):
        """


        Args:
            X (pd.Series): column to encode
            y (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: input (when used with ColumnTransformer) must be of the form: [column_to_target_transform, target_column]

        Returns:
            target mapping: target mapping from training set
        """
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
        """
        Args:
            X (pd.Series): column to encode

        Raises:
            ValueError: The encoder has not been fitted

        Returns:
            pd.Series: feature with target encoder applies
        """

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
    """

    A lot of numerical features are in dollars and so are encoded as an object. This class trasnforms into numeric

    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        Args:
            X (pd.Series): Column with units in dolars

        Returns:
            pd.Series: column trasnsformed into int

        """
        X = X.apply(
            lambda x: (
                int(float(x[1:])) if isinstance(x, str) and x.startswith("$") else x
            )
        )

        return X.to_frame()


class TimeSeriesMapper:
    """

    Class that changes date-like column into ordered time series
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Args:
            X (pd.Series): date-like column
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: column ordered by time 1,..,T
        """

        # X must be a date time
        X = pd.Series(X.iloc[:, 0])
        date_values = list(X.sort_values().values)
        date_ts_mapping = {date_value: i for i, date_value in enumerate(date_values)}
        X = X.map(date_ts_mapping)
        return X.to_frame()


class RemoveUncorrFeatures:
    """
    Class to remove features with insignificant correlation with target variable
    """

    def __init__(self, p, target="is_fraud"):
        """
        Args:
            p (int): minimum correlation with target
            target (str, optional): Target column. Defaults to "is_fraud".
        """

        self.mapping = None
        self.p = p
        self.target = target

    def fit(self, X, y=None):
        """
        Args:
            X (pd.Series): Dataframe from which to remove uncorrelated features

        Returns:
            mapping: Mapping of sufficiently correlated features, based on training set
        """

        ## Function to remove features with less than X correlation
        suitable_cols = X.corr()[self.target][
            np.abs(X.corr()[self.target].values) > self.p
        ]
        suitable_cols = list(suitable_cols.index)

        self.mapping = suitable_cols

        return self

    def transform(self, X, y=None):
        """
        Args:
            X (pd.DataFrame): Dataframe from which to remove uncorrelated features

        Raises:
            ValueError: If mapping has not been fitted

        Returns:
            pd.DataFrame: DataFrame with uncorrelated features removed
        """

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
