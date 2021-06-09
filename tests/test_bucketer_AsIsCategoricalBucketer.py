import numpy as np
from skorecard.bucketers import AsIsCategoricalBucketer


def test_correct_output(df):
    """Test that correct use of CatBucketTransformer returns expected results."""
    X = df
    y = df["default"].values

    asb = AsIsCategoricalBucketer(variables=["EDUCATION"])
    asb.fit(X, y)
    X_trans = asb.transform(X)
    assert len(X["EDUCATION"].unique()) == len(X_trans["EDUCATION"].unique())


def test_specials(df):
    """Test specials get assigned to the right bin."""
    X = df[["EDUCATION"]]
    y = df["default"]

    asb = AsIsCategoricalBucketer(variables=["EDUCATION"], specials={"EDUCATION": {"ed 0": [1]}})
    asb.fit(X, y)

    X_transform = asb.transform(X)
    # Make sure value 1 is assigned special bucket
    assert np.unique(X_transform[X["EDUCATION"] == 1].values)[0] == -3


def test_missing_default(df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    bucketer = AsIsCategoricalBucketer(variables=["EDUCATION"])
    X["EDUCATION_trans"] = bucketer.fit_transform(X[["EDUCATION"]], y)

    assert len(X["EDUCATION_trans"].unique()) == 8  # 7 unique values + 1 for NAs
