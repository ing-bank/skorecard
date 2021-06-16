import numpy as np
import pandas as pd
import pytest

import skorecard.reporting.report
from skorecard.metrics import metrics
from skorecard.bucketers import DecisionTreeBucketer


@pytest.fixture()
def X_y():
    """Set of X,y for testing the transformers."""
    X = np.array(
        [[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]],
        np.int32,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])

    return X, y


@pytest.fixture()
def X1_X2():
    """Set of dataframes to test psi."""
    X1 = pd.DataFrame(
        [[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], columns=["col1", "col2"]
    )
    X2 = pd.DataFrame(
        [[0, 2], [3, 0], [0, 0], [1, 2], [0, 4], [2, 1], [1, 1], [2, 1], [1, 1]], columns=["col1", "col2"]
    )

    return X1, X2


def test_iv_on_array(X_y):
    """Test the IV calculation for two arrays."""
    X, y = X_y
    X = pd.DataFrame(X, columns=["0", "1"])

    np.testing.assert_almost_equal(metrics._IV_score(y, X["0"]), 5.307, decimal=2)

    np.testing.assert_almost_equal(metrics._IV_score(y, X["1"]), 4.635, decimal=2)


def test_psi_zero(df):
    """Test that PSI on same array is zero."""
    features = ["LIMIT_BAL", "BILL_AMT1"]
    X = df[features]
    y = df["default"]

    X_bins = DecisionTreeBucketer(variables=features).fit_transform(X, y)

    psi_vals = skorecard.reporting.report.psi(X_bins, X_bins)

    assert set(psi_vals.keys()) == set(X_bins.columns)
    assert all([val == 0 for val in psi_vals.values()])


def test_psi_values(X1_X2):
    """Assert PSi values match expectations."""
    X1, X2 = X1_X2
    expected_psi = {"col1": 0.0773, "col2": 0.965}
    psi_vals = skorecard.reporting.report.psi(X1, X2)

    np.testing.assert_array_almost_equal(pd.Series(expected_psi).values, pd.Series(psi_vals).values, decimal=2)


def test_IV_values(X_y):
    """Assert IV values match expectations."""
    X, y = X_y
    X = pd.DataFrame(X, columns=["col1", "col2"])
    expected_iv = {"col1": 5.307, "col2": 4.635}
    iv_vals = skorecard.reporting.report.iv(X, y)

    np.testing.assert_array_almost_equal(pd.Series(expected_iv).values, pd.Series(iv_vals).values, decimal=2)
