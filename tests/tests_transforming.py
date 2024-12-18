import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add root to sys.path
root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)
# print("sys.path:", sys.path)

from Modules.transforming import (
    Target0_Reducer,
    CustomTargetEncoder,
)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_Target0_Reducer_percentage():
    print("Testing Target0_Reducer with percentage parameter...")

    # Testing for desired percentage
    percentage_to_test = 0.1
    data = pd.DataFrame(
        {
            "target": [
                "Yes",
                "Yes",
                "Yes",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
            ]
        }
    )

    expected = data["target"].value_counts()
    expected_no_count = int(expected["No"] * percentage_to_test)

    transform_1 = Target0_Reducer(percentage=percentage_to_test)

    results = transform_1.fit_transform(data)
    results_vc = results["target"].value_counts()
    results_no_count = results_vc["No"]

    print(
        f"Expected No count: {expected_no_count}, Actual No count: {results_no_count}\n"
    )
    assert expected_no_count == results_no_count


def test_Target0_Reducer_balanced():
    print("Testing Target0_Reducer with balanced=True parameter...\n")

    # Testing for balanced dataset
    data = pd.DataFrame(
        {
            "target": [
                "Yes",
                "Yes",
                "Yes",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
                "No",
            ]
        }
    )

    expected = data["target"].value_counts()
    expected_no_count = int(expected["Yes"])

    transform_1 = Target0_Reducer(balanced=True)

    results = transform_1.fit_transform(data)
    results_vc = results["target"].value_counts()
    results_no_count = results_vc["No"]

    print(
        f"Expected No count: {expected_no_count}, Actual No count: {results_no_count}\n"
    )
    assert expected_no_count == results_no_count


def test_Target0_Reducer_invalid_target():
    print("Testing Target0_Reducer with invalid target values...\n")

    data = pd.DataFrame({"target": ["Yes", "No", "Invalid"]})

    transform_1 = Target0_Reducer()

    with pytest.raises(ValueError, match="invalid values"):
        transform_1.fit_transform(data)


def test_Target0_Reducer_missing_target_column():
    print("Testing Target0_Reducer with missing target column...\n")

    data = pd.DataFrame({"non_target": ["Yes", "No", "No"]})

    transform_1 = Target0_Reducer()

    with pytest.raises(
        KeyError, match="The input DataFrame must have a 'target' column"
    ):
        transform_1.fit_transform(data)


def test_CustomTargetEncoder_basic():
    logger.info("Testing CustomTargetEncoder with basic configuration...")

    # Training data
    training_data = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B", "A", "A", "B", "A", "B", "A"],
            "target": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        }
    )

    # Testing data
    testing_data = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B", "A", None],
            "target": [1, 1, 1, 1, 0, 1],
        }
    )

    # Initialize and fit the transformer
    transformer = CustomTargetEncoder(target="target")
    transformer.fit(training_data)

    # Transform the test data
    transformed_data = transformer.transform(testing_data)

    # Manually calculate expected values
    target_mean = training_data["target"].mean()
    expected_mapping = training_data.groupby("cat")["target"].mean().to_dict()
    expected_mapping[None] = target_mean

    logger.info(f"Expected mapping: {expected_mapping}")
    logger.info(f"Transformed data:\n{transformed_data}")

    # Validate transformed values
    for i, row in testing_data.iterrows():
        cat = row["cat"]
        expected_value = expected_mapping.get(cat, target_mean)
        assert transformed_data.iloc[i, 0] == expected_value


def test_CustomTargetEncoder_with_unknown_categories():
    logger.info("Testing CustomTargetEncoder with unknown categories...")

    # Training data
    training_data = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B", "A", "A", "B", "A", "B", "A"],
            "target": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        }
    )

    # Testing data with unknown categories
    testing_data = pd.DataFrame(
        {
            "cat": ["C", "D", "E", "B", None],
            "target": [0, 1, 0, 1, 0],
        }
    )

    # Initialize and fit the transformer
    transformer = CustomTargetEncoder(target="target")
    transformer.fit(training_data)

    # Transform the test data
    transformed_data = transformer.transform(testing_data)

    # Manually calculate expected values
    target_mean = training_data["target"].mean()
    expected_mapping = training_data.groupby("cat")["target"].mean().to_dict()
    expected_mapping[None] = target_mean

    logger.info(f"Expected mapping: {expected_mapping}")
    logger.info(f"Transformed data:\n{transformed_data}")

    # Validate transformed values
    for i, row in testing_data.iterrows():
        cat = row["cat"]
        expected_value = expected_mapping.get(cat, target_mean)
        assert transformed_data.iloc[i, 0] == expected_value


def test_CustomTargetEncoder_empty_dataframe():
    logger.info("Testing CustomTargetEncoder with an empty DataFrame...")

    # Training data
    training_data = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B", "A"],
            "target": [1, 0, 1, 1, 0],
        }
    )

    # Empty testing data
    testing_data = pd.DataFrame(columns=["cat", "target"])

    # Initialize and fit the transformer
    transformer = CustomTargetEncoder(target="target")
    transformer.fit(training_data)

    # Transform the test data
    transformed_data = transformer.transform(testing_data)

    logger.info(f"Transformed data for empty DataFrame:\n{transformed_data}")

    # Validate transformed data is also empty
    assert transformed_data.empty


def test_CustomTargetEncoder_only_nan():
    logger.info("Testing CustomTargetEncoder with only NaN values in the test set...")

    # Training data
    training_data = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B", "A"],
            "target": [1, 0, 1, 1, 0],
        }
    )

    # Testing data with only NaN values
    testing_data = pd.DataFrame(
        {
            "cat": [None, None, None],
            "target": [1, 0, 1],
        }
    )

    # Initialize and fit the transformer
    transformer = CustomTargetEncoder(target="target")
    transformer.fit(training_data)

    # Transform the test data
    transformed_data = transformer.transform(testing_data)

    # Manually calculate the expected value
    target_mean = training_data["target"].mean()

    logger.info(f"Target mean: {target_mean}")
    logger.info(f"Transformed data:\n{transformed_data}")

    # Validate transformed values
    for val in transformed_data.iloc[:, 0]:
        assert val == target_mean
