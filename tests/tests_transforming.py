import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import sys
from pathlib import Path

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


def test_Date():

    # input data
    data = pd.DataFrame(
        {"date": ["10/03/2022", "13/04/1995", "20/07/2001", "21/09/2013"]}
    )
    transformer = Date()

    # Results
    result = transformer.fit_transform(data["date"])
    expected = pd.to_datetime(data["date"], format="mixed")

    # Assert equality
    pdt.assert_series_equal(result, expected)


test_Date()
