import numpy as np
import pandas as pd
import pytest

from skorecard.bucketers import OrdinalCategoricalBucketer


def test_threshold_min(df) -> None:
    """Test that threshold_min < 1 raises an error."""
    with pytest.raises(ValueError):
        OrdinalCategoricalBucketer(tol=-0.1, variables=["EDUCATION"]).fit(X=None, y=None)
    with pytest.raises(ValueError):
        OrdinalCategoricalBucketer(tol=1.001, variables=["EDUCATION"]).fit(X=None, y=None)


def test_correct_output(df):
    """Test that correct use of CatBucketTransformer returns expected results."""
    X = df
    y = df["default"].values

    cbt = OrdinalCategoricalBucketer(tol=0.44, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 2

    cbt = OrdinalCategoricalBucketer(tol=0.05, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 4

    cbt = OrdinalCategoricalBucketer(tol=0, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == len(X["EDUCATION"].unique())

    # when the threshold is above the maximum value, make sure its only one bucket
    cbt = OrdinalCategoricalBucketer(tol=0.5, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 1


def test_mapping_dict(df):
    """Test that the mapping dict is created correctly."""
    X = df
    y = df["default"].values
    cbt = OrdinalCategoricalBucketer(tol=0, variables=["EDUCATION"])
    cbt.fit(X, y)
    bucket_map = cbt.features_bucket_mapping_.get("EDUCATION")
    assert len(bucket_map.map) == len(np.unique(X["EDUCATION"]))


def test_encoding_method(df):
    """Test the encoding method."""
    X = df[["EDUCATION", "default"]]
    y = df["default"]

    ocb = OrdinalCategoricalBucketer(tol=0.03, variables=["EDUCATION"], encoding_method="frequency")
    ocb.fit(X, y)

    assert ocb.features_bucket_mapping_.get("EDUCATION").map == {1: 1, 2: 0, 3: 2}

    ocb = OrdinalCategoricalBucketer(tol=0.03, variables=["EDUCATION"], encoding_method="ordered")
    ocb.fit(X, y)

    assert ocb.features_bucket_mapping_.get("EDUCATION").map == {1: 0, 2: 2, 3: 1}


def test_specials(df):
    """Test specials get assigned to the highest bin."""
    X = df[["EDUCATION"]]
    y = df["default"]

    ocb = OrdinalCategoricalBucketer(
        tol=0.03, variables=["EDUCATION"], encoding_method="ordered", specials={"EDUCATION": {"ed 0": [1]}}
    )
    ocb.fit(X, y)

    X_transform = ocb.transform(X)
    # Make sure value 1 is assigned bucket -3
    assert np.unique(X_transform[X["EDUCATION"] == 1].values)[0] == -3

    ocb = OrdinalCategoricalBucketer(
        tol=0.03, variables=["EDUCATION"], encoding_method="frequency", specials={"EDUCATION": {"ed 0": [1]}}
    )
    ocb.fit(X, y)

    X_transform = ocb.transform(X)
    # Make sure value 1 is assigned bucket -3
    assert np.unique(X_transform[X["EDUCATION"] == 1].values)[0] == -3


def test_missing_default(df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    bucketer = OrdinalCategoricalBucketer(variables=["EDUCATION"])
    bucketer = OrdinalCategoricalBucketer(max_n_categories=2, variables=["EDUCATION"])
    X["EDUCATION_trans"] = bucketer.fit_transform(X[["EDUCATION"]], y)

    assert len(X["EDUCATION_trans"].unique()) == 3  # 2 + 1 for NAs
    assert X[X["EDUCATION"].isnull()].shape[0] == X[X["EDUCATION_trans"] == -1].shape[0]


def test_missing_manual(df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    bucketer = OrdinalCategoricalBucketer(variables=["EDUCATION"])
    bucketer = OrdinalCategoricalBucketer(
        max_n_categories=2, variables=["EDUCATION"], missing_treatment={"EDUCATION": 0}
    )
    X["EDUCATION_trans"] = bucketer.fit_transform(X[["EDUCATION"]], y)

    assert len(X["EDUCATION_trans"].unique()) == 2
    assert X[X["EDUCATION"].isnull()]["EDUCATION_trans"].sum() == 0


def test_missings():
    """
    Test proper handling of NAs.
    """
    X = pd.DataFrame({"colour": ["blue"] * 10 + ["red"] * 10 + ["grey"] * 10 + [np.nan]})
    y = [1] * 5 + [0] * 5 + [1] * 8 + [0] * 2 + [1] * 1 + [0] * 9 + [1]
    y = np.array(y)
    ocb = OrdinalCategoricalBucketer()
    X_trans = ocb.fit_transform(X, y)
    buckets = X_trans["colour"].unique().tolist()
    buckets.sort()

    assert buckets == [-1, 0, 1, 2]
    assert ocb.bucket_table("colour").shape[0] == 5

    ocb = OrdinalCategoricalBucketer(encoding_method="ordered")
    X_trans = ocb.fit_transform(X, y)
    assert buckets == [-1, 0, 1, 2]
    assert ocb.bucket_table("colour").shape[0] == 5
