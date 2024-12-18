import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add root to sys.path
root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)
# print("sys.path:", sys.path)

from Modules.transforming import (
    TargetBinary,
    Date,
    DateDecomposer,
    Target0_Reducer,
    CustomTargetEncoder,
    DollarToInt,
    TimeSeriesMapper,
    RemoveUncorrFeatures,
)


def test_TargetBinary():

    # Input data
    data = pd.DataFrame({"target": ["Yes", "no", "No", "Yes"]})
    transformer_df = TargetBinary(type="df")

    # Result
    result_df = transformer_df.fit_transform(data)

    # Expected output
    expected_df = pd.DataFrame({"target": [1, 0, 0, 1]})

    # Assert equality
    pdt.assert_frame_equal(result_df, expected_df)

    # Drop rows which are non yes no???


def test_Date():

    # input data
    data = pd.DataFrame(
        {"date": ["10/03/2022", "13/04/1995", "20/07/2001", "21/09/2013"]}
    )
    transformer = Date()

    # Results
    result = transformer.fit_transform(data["date"])["date"]
    expected = pd.to_datetime(data["date"], format="mixed")

    ## Assert equality
    pdt.assert_series_equal(result, expected)
    # print(result)
    # print(expected)

    # Drop rows that cannot be made into dates


def test_DateDecomposer():

    # Generate hourly intervals for today's date
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    hourly_intervals = [start_time + timedelta(hours=i) for i in range(24)]

    # Create a DataFrame
    data_expected = pd.DataFrame({"date": hourly_intervals})
    data_expected["date_hour"] = data_expected["date"].dt.hour
    data_expected["date_dow"] = data_expected["date"].dt.weekday
    data_expected["date_month"] = data_expected["date"].dt.month
    data_expected["date_year"] = data_expected["date"].dt.year

    data_result = pd.DataFrame({"date": hourly_intervals})

    # Transformer
    t_h = DateDecomposer(time_element_to_extract="hour", col_to_decomp="date")
    t_dow = DateDecomposer(time_element_to_extract="dow", col_to_decomp="date")
    t_month = DateDecomposer(time_element_to_extract="month", col_to_decomp="date")
    t_year = DateDecomposer(time_element_to_extract="year", col_to_decomp="date")

    # Expected
    e_h = t_h.transform(data_result)
    e_dow = t_dow.transform(data_result)
    e_month = t_month.transform(data_result)
    e_year = t_year.transform(data_result)

    ## Assert equality
    pdt.assert_frame_equal(data_result, data_expected)


def test_Target0_Reducer():

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

    assert expected_no_count == results_no_count

    # Testing if balanced df is required

    # Testing for desired percentage

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

    assert expected_no_count == results_no_count
