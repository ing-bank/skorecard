import numpy as np

from skorecard import datasets
from skorecard.linear_model import LogisticRegression
from skorecard.bucketers import EqualFrequencyBucketer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pytest


@pytest.fixture()
def X_y():
    """Generate dataframe."""
    X, y = datasets.load_uci_credit_card(return_X_y=True)
    return X, y


def test_output_dimensions():
    """Test the dimensions of the new attributes."""
    shape_features = (10, 3)

    X = np.random.uniform(size=shape_features)
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])

    lr = LogisticRegression(fit_intercept=True).fit(X, y, calculate_stats=True)
    assert lr.p_val_coef_.shape[1] == shape_features[1]
    assert lr.z_coef_.shape[1] == shape_features[1]
    assert len(lr.std_err_coef_) == shape_features[1]
    assert not np.isnan(lr.p_val_intercept_)
    assert not np.isnan(lr.z_intercept_)
    assert not np.isnan(lr.std_err_intercept_)

    lr = LogisticRegression(fit_intercept=False).fit(X, y, calculate_stats=True)

    assert lr.p_val_coef_.shape[1] == shape_features[1]
    assert lr.z_coef_.shape[1] == shape_features[1]
    assert len(lr.std_err_coef_) == shape_features[1]
    assert np.isnan(lr.p_val_intercept_)
    assert np.isnan(lr.z_intercept_)
    assert np.isnan(lr.std_err_intercept_)


def test_results(X_y):
    """Test the actual p-values."""
    expected_approx_p_val_coef_ = np.array([[1.0, 1.0, 0.0, 0.8425]])

    lr = LogisticRegression(fit_intercept=True, penalty="none").fit(*X_y, calculate_stats=True)

    np.testing.assert_array_almost_equal(lr.p_val_coef_, expected_approx_p_val_coef_, decimal=3)


def test_linear_model(X_y):
    """Test OHE sparse matrix compatibility."""
    pipeline = Pipeline(
        [
            ("bucketer", EqualFrequencyBucketer(n_bins=10)),
            ("ohe", OneHotEncoder()),
            ("clf", LogisticRegression(calculate_stats=True)),
        ]
    )
    pipeline.fit(*X_y)
    assert pipeline.named_steps["clf"].p_val_coef_.shape[1] > 0


def test_get_stats(X_y):
    """Test that we get the expected pandas dataframe."""
    lr = LogisticRegression(fit_intercept=True).fit(*X_y)

    with pytest.raises(AssertionError):
        lr.get_stats()

    lr = LogisticRegression(fit_intercept=True).fit(*X_y, calculate_stats=True)

    # We add 1 because of the intercept
    assert lr.get_stats().shape[0] == len(X_y[0].columns) + 1
    assert lr.get_stats().index[0] == "const"

    lr = LogisticRegression(fit_intercept=False, calculate_stats=True).fit(*X_y)
    assert lr.get_stats().fillna(-999)["Coef."][0] == 0
    assert lr.get_stats().fillna(-999)["Std.Err"][0] == -999
    assert lr.get_stats().fillna(-999)["z"][0] == -999
    assert lr.get_stats().fillna(-999)["P>|z|"][0] == -999
