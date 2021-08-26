import numpy as np
import pytest
import pandas as pd

from skorecard.preprocessing import WoeEncoder
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, AsIsCategoricalBucketer
from skorecard.pipeline import BucketingProcess

from sklearn.pipeline import make_pipeline

import numpy.testing as npt

# TODO: WoE should treat missing values as a separate bin and thus handled seamlessly.


@pytest.fixture()
def X_y():
    """Set of X,y for testing the transformers."""
    X = pd.DataFrame(
        np.array(
            [[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]],
            np.int32,
        ),
        columns=["col1", "col2"],
    )
    y = pd.Series(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))

    return X, y


@pytest.fixture()
def X_y_2():
    """Set of X,y for testing the transformers.

    In the first column, bucket 3 is not present in class 1.
    """
    X = pd.DataFrame(
        np.array(
            [[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]],
            np.int32,
        ),
        columns=["col1", "col2"],
    )
    y = pd.Series(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))

    return X, y


def test_woe_transformed_dimensions(X_y):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y

    woeb = WoeEncoder(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)
    assert len(new_X["col1"].unique()) == len(X["col1"].unique())
    assert len(new_X["col2"].unique()) == len(X["col2"].unique())


def test_missing_bucket(X_y_2):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y_2

    woeb = WoeEncoder(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)

    assert new_X.shape == X.shape
    assert len(new_X["col1"].unique()) == len(X["col1"].unique())
    assert len(new_X["col2"].unique()) == len(X["col2"].unique())

    # because class 1 will have zero counts, the WOE transformer will divide by the value of epsilon, avoinding infinite
    # numbers
    assert not any(new_X["col1"] == np.inf)
    assert not any(new_X["col2"] == np.inf)

    # If epsilon is set to zero, expect a Division By Zero exception
    with pytest.raises(ZeroDivisionError):
        WoeEncoder(epsilon=0, variables=["col1", "col2"]).fit_transform(X, y)


def test_woe_values(X_y):
    """Tests the value of the WOE."""
    X, y = X_y

    woeb = WoeEncoder(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)
    new_X

    expected = pd.DataFrame(
        {
            "col1": {
                0: -0.22309356256166865,
                1: -0.22304359629388562,
                2: -0.22309356256166865,
                3: -7.824445930877619,
                4: -0.22309356256166865,
                5: -0.22304359629388562,
                6: 8.294299608857235,
                7: 8.294299608857235,
                8: -0.22309356256166865,
            },
            "col2": {
                0: 0.469853677979616,
                1: 0.8752354701118937,
                2: 0.8752354701118937,
                3: -8.517393171418904,
                4: 0.469853677979616,
                5: -8.517393171418904,
                6: 0.8752354701118937,
                7: 0.469853677979616,
                8: 0.8752354701118937,
            },
        }
    )

    pd.testing.assert_frame_equal(new_X, expected)

    assert woeb.transform(X).equals(new_X)


def test_woe_in_pipeline(df):
    """Tests if WoeEncoder works in pipeline context."""
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )

    pipeline = make_pipeline(
        bucketing_process,
        WoeEncoder(),
    )

    out = pipeline.fit_transform(X, y)
    assert all(out.dtypes == float)
    npt.assert_almost_equal(out["LIMIT_BAL"][0], 0.566607, decimal=4)
    npt.assert_almost_equal(out["LIMIT_BAL"][1], -0.165262, decimal=4)
    npt.assert_almost_equal(out["LIMIT_BAL"][3], -0.031407, decimal=4)
    npt.assert_almost_equal(out["BILL_AMT1"][0], 0.124757, decimal=4)
    npt.assert_almost_equal(out["BILL_AMT1"][1], 0.091788, decimal=4)
    npt.assert_almost_equal(out["BILL_AMT1"][3], -0.103650, decimal=4)
