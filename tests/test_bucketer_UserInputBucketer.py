import pytest

from sklearn.exceptions import NotFittedError

from skorecard.bucketers import UserInputBucketer
from skorecard.bucket_mapping import BucketMapping


@pytest.fixture()
def example_features_bucket_map():
    """Generate example dict."""
    return {
        "LIMIT_BAL": BucketMapping(
            feature_name="LIMIT_BAL",
            type="numerical",
            map=[105000.0, 265000.0],
            right=True,
        ),
        "BILL_AMT1": BucketMapping(
            feature_name="BILL_AMT1",
            type="numerical",
            map=[34211.5, 173337.5],
            right=True,
        ),
    }


def test_no_transformation(df):
    """Test that we can use the transformer with no input."""
    X = df
    ui_bucketer = UserInputBucketer()

    new_X = ui_bucketer.transform(X)
    assert X.equals(new_X)


def test_manual_transformation(df, example_features_bucket_map):
    """Test that we can use an example dict for ManualBucketTransformer."""
    X = df
    y = df["default"]

    ui_bucketer = UserInputBucketer(example_features_bucket_map)
    new_X = ui_bucketer.fit_transform(X)
    assert len(new_X["LIMIT_BAL"].unique()) == 3
    assert len(new_X["BILL_AMT1"].unique()) == 3

    # specify y
    ui_bucketer = UserInputBucketer(example_features_bucket_map)
    ui_bucketer.fit(X, y)
    ui_bucketer.transform(X)


def test_bucket_table_method(df, example_features_bucket_map):
    """
    Make sure .bucket_table() works properly.
    """
    X = df
    y = df["default"]

    ui_bucketer = UserInputBucketer(example_features_bucket_map)

    with pytest.raises(NotFittedError):
        ui_bucketer.bucket_table("EDUCATION")

    ui_bucketer.fit(X, y)

    with pytest.raises(ValueError):
        # EDUCATION is not in 'example_features_bucket_map'
        ui_bucketer.bucket_table("EDUCATION")

    assert ui_bucketer.bucket_table("BILL_AMT1").shape[0] == 4


def test_summary_method(df, example_features_bucket_map):
    """
    Make sure .summary() works properly.
    """
    X = df
    y = df["default"]

    ui_bucketer = UserInputBucketer(example_features_bucket_map)

    with pytest.raises(NotFittedError):
        ui_bucketer.summary()

    ui_bucketer.fit(X, y)
    assert ui_bucketer.summary().shape[0] == 6
