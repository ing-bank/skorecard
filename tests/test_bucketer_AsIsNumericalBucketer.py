import pytest
import pandas as pd

from skorecard.bucketers import AsIsNumericalBucketer
from skorecard.utils.exceptions import NotPreBucketedError


def test_manual_transformation(df):
    """
    Test that we can use an example dict for ManualBucketTransformer.
    """
    ref = pd.DataFrame({"a": [1, 3, 5]})
    ref2 = pd.DataFrame({"a": [1.001, 2.999, 3, 6]})

    ai = AsIsNumericalBucketer()
    ai.fit(ref)
    assert ai.bucket_table("a")["label"][1] == "(-inf, 1.0]"
    assert list(ai.transform(ref2)["a"].values) == [1, 1, 1, 3]

    ai = AsIsNumericalBucketer(right=False)
    ai.fit(ref)

    assert ai.bucket_table("a")["label"][1] == "[-inf, 1.0)"
    assert list(ai.transform(ref2)["a"].values) == [1, 1, 2, 3]


def test_fit_transform(df):
    """
    Test basic transform.
    """
    features = ["LIMIT_BAL", "EDUCATION", "MARRIAGE"]  # BILL_AMT1 has > 100 unique values
    X = df[features]
    y = df["default"].values

    ai = AsIsNumericalBucketer()
    ai.fit_transform(X, y)

    with pytest.raises(NotPreBucketedError):
        X = df[["BILL_AMT1"]]
        ai = AsIsNumericalBucketer()
        ai.fit_transform(X, y)
