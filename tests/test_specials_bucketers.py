from skorecard import datasets
from skorecard.bucketers import DecisionTreeBucketer, EqualFrequencyBucketer, EqualWidthBucketer, OptimalBucketer
import numpy as np

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_specials_tree_bucketer(df):
    """Test that when adding specials,the binner performs as expected.

    Context: special values should be binned in their own bin.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    # Because 2 special buckets are defined, the decision tree
    # will be fitted with max_leaf_nodes=1. This will create a crash in the sklearn implementation.
    # In this case, Skorecard raises an exception with a clear recommendation to the user when the fit method is called.
    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], random_state=1, max_n_bins=3, specials=specials)
    with pytest.raises(ValueError):
        tbt.fit_transform(X, y)

    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], random_state=1, max_n_bins=5, specials=specials)
    X_bins = tbt.fit_transform(X, y)

    assert X_bins["BILL_AMT1"].nunique() == 5
    assert X_bins["LIMIT_BAL"].nunique() == 5

    assert X_bins[X["LIMIT_BAL"] == 50000]["LIMIT_BAL"].unique() == np.array(-3)

    # Test that the labels are properly assigned. Because there are no specials in BILL_AMT1, there should be no extra
    # bins
    assert len(tbt.features_bucket_mapping_.get("BILL_AMT1").labels) == 6
    # check that the last label finishes with inf
    assert tbt.features_bucket_mapping_.get("BILL_AMT1").labels[0].startswith("[-inf")
    assert tbt.features_bucket_mapping_.get("BILL_AMT1").labels[4].endswith("inf)")

    # Test that the labels are properly assigned. Because there are 2 specials in LIMIT_BAL, there should be 2 extra
    # bins
    assert len(tbt.features_bucket_mapping_.get("LIMIT_BAL").labels) == 6
    # check that the labels match the specials dictionary
    assert (
        tbt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-3]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][0])
    )
    assert (
        tbt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-4]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][1])
    )

    # Assert a value error is raised if the specials contains features not defined in the bucketer.
    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}, "Undefinedfeature": {"1": [2]}}

    with pytest.raises(ValueError):
        DecisionTreeBucketer(
            variables=["LIMIT_BAL", "BILL_AMT1"], random_state=1, max_n_bins=3, specials=specials
        ).fit_transform(X, y)


def test_specials_equal_width_bucketer(df):
    """Test that when adding specials,the binner performs as expected.

    Context: special values should be binned in their own bin.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    ebt = EqualWidthBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], n_bins=3, specials=specials)
    X_bins = ebt.fit_transform(X, y)

    assert X_bins["BILL_AMT1"].nunique() == 3
    assert X_bins["LIMIT_BAL"].nunique() == 5  # maximum n_bins +2 coming from the specials

    assert X_bins[X["LIMIT_BAL"] == 50000]["LIMIT_BAL"].unique() == np.array(-3)

    # Test that the labels are properly assigned. Because there are no specials in BILL_AMT1, there should be no extra
    # bins
    assert len(ebt.features_bucket_mapping_.get("BILL_AMT1").labels) == 4
    # check that the last label finishes with inf
    assert ebt.features_bucket_mapping_.get("BILL_AMT1").labels[0].startswith("(-inf")
    assert ebt.features_bucket_mapping_.get("BILL_AMT1").labels[2].endswith("inf]")

    # Test that the labels are properly assigned. Because there are 2 specials in LIMIT_BAL, there should be 2 extra
    # bins
    assert len(ebt.features_bucket_mapping_.get("LIMIT_BAL").labels) == 6
    # check that the labels match the specials dictionary
    assert (
        ebt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-3]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][0])
    )
    assert (
        ebt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-4]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][1])
    )

    # Assert a value error is raised if the specials contains features not defined in the bucketer.
    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}, "Undefinedfeature": {"1": [2]}}

    with pytest.raises(ValueError):
        EqualWidthBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], n_bins=3, specials=specials).fit_transform(X, y)


def test_specials_equal_frequency_bucketer(df):
    """Test that when adding specials,the binner performs as expected.

    Context: special values should be binned in their own bin.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    ebt = EqualFrequencyBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], n_bins=3, specials=specials)
    X_bins = ebt.fit_transform(X, y)

    assert X_bins["BILL_AMT1"].nunique() == 3
    assert X_bins["LIMIT_BAL"].nunique() == 5  # maximum n_bins +2 coming from the specials

    assert X_bins[X["LIMIT_BAL"] == 50000]["LIMIT_BAL"].unique() == np.array(-3)

    # Test that the labels are properly assigned. Because there are no specials in BILL_AMT1, there should be no extra
    # bins
    assert len(ebt.features_bucket_mapping_.get("BILL_AMT1").labels) == 4
    # check that the last label finishes with inf
    assert ebt.features_bucket_mapping_.get("BILL_AMT1").labels[0].startswith("(-inf")
    assert ebt.features_bucket_mapping_.get("BILL_AMT1").labels[2].endswith("inf]")

    # Test that tha labels are properly assigned. Because there are 2 specials in LIMIT_BAL, there should be 2 extra
    # bins
    assert len(ebt.features_bucket_mapping_.get("LIMIT_BAL").labels) == 6
    # check that the labels match the specials dictionary
    assert (
        ebt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-3]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][0])
    )
    assert (
        ebt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-4]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][1])
    )

    # Assert a value error is raised if the specials contains features not defined in the bucketer.
    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}, "Undefinedfeature": {"1": [2]}}

    with pytest.raises(ValueError):
        EqualFrequencyBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], n_bins=3, specials=specials).fit_transform(X, y)


def test_specials_optimal_bucketer(df):
    """Test that when adding specials,the binner performs as expected.

    Context: special values should be binned in their own bin.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    opt = OptimalBucketer(variables=["LIMIT_BAL"], max_n_bins=3, specials=specials)
    X_bins = opt.fit_transform(X, y)

    assert X_bins["LIMIT_BAL"].nunique() == 5

    assert X_bins[X["LIMIT_BAL"] == 50000]["LIMIT_BAL"].unique() == np.array(-3)

    # Test that tha labels are properly assigned. Because there are 2 specials in LIMIT_BAL, there should be 2 extra
    # bins
    assert len(opt.features_bucket_mapping_.get("LIMIT_BAL").labels) == 6
    # check that the labels match the specials dictionary
    assert (
        opt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-3]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][0])
    )
    assert (
        opt.features_bucket_mapping_.get("LIMIT_BAL")
        .labels[-4]
        .endswith([key for key in specials["LIMIT_BAL"].keys()][1])
    )
