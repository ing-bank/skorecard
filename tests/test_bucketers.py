import pytest
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from skorecard.bucketers.bucketers import UserInputBucketer
from skorecard.bucketers import (
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
    OptimalBucketer,
    OrdinalCategoricalBucketer,
    AsIsCategoricalBucketer,
    AsIsNumericalBucketer,
    DecisionTreeBucketer,
)
from skorecard.pipeline import BucketingProcess

BUCKETERS_WITH_SET_BINS = [
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
]

BUCKETERS_WITHOUT_SET_BINS = [
    OptimalBucketer,
    OrdinalCategoricalBucketer,
    AsIsNumericalBucketer,
    AsIsCategoricalBucketer,
    DecisionTreeBucketer,
]

# Except the very special UserInputBucketer of course :)
ALL_BUCKETERS = BUCKETERS_WITH_SET_BINS + BUCKETERS_WITHOUT_SET_BINS
ALL_BUCKETERS_WITH_BUCKETPROCESS = ALL_BUCKETERS + [BucketingProcess]


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_single_bucket(bucketer, df) -> None:
    """Test that using n_bins=1 puts everything into 1 bucket."""
    BUCK = bucketer(n_bins=1, variables=["MARRIAGE"])
    x_t = BUCK.fit_transform(df)
    assert len(x_t["MARRIAGE"].unique()) == 1


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_two_buckets(bucketer, df) -> None:
    """Test that using n_bins=1 puts everything into 2 buckets."""
    X = df
    y = df["default"].values

    BUCK = bucketer(n_bins=2, variables=["MARRIAGE"])
    BUCK.fit(X, y)
    x_t = BUCK.transform(X)
    assert len(x_t["MARRIAGE"].unique()) == 2


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_three_bins(bucketer, df) -> None:
    """Test that we get the number of bins we request."""
    # Test single bin counts
    BUCK = bucketer(n_bins=3, variables=["MARRIAGE"])
    x_t = BUCK.fit_transform(df)
    assert len(x_t["MARRIAGE"].unique()) == 3


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_error_input(bucketer):
    """Test that a non-int leads to problems in bins.

    Note input validation is done on fit, but before data validation.
    """
    with pytest.raises(AssertionError):
        bucketer(n_bins=[2]).fit(X=1, y=1)

    with pytest.raises(AssertionError):
        bucketer(n_bins=4.2, variables=["MARRIAGE"]).fit(X=1, y=1)


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_missings_set(bucketer, df_with_missings) -> None:
    """Test all missing methods work for bucketers with set bins."""
    X = df_with_missings
    y = df_with_missings["default"].values

    BUCK = bucketer(n_bins=2, variables=["MARRIAGE"])
    BUCK.fit(X, y)
    X["MARRIAGE_trans"] = BUCK.transform(X)["MARRIAGE"]
    assert len(X["MARRIAGE_trans"].unique()) == 3
    assert X[np.isnan(X["MARRIAGE"])].shape[0] == X[X["MARRIAGE_trans"] == -1].shape[0]

    X = df_with_missings
    y = df_with_missings["default"].values

    BUCK = bucketer(n_bins=3, variables=["MARRIAGE", "LIMIT_BAL"], missing_treatment={"LIMIT_BAL": 1, "MARRIAGE": 0})
    BUCK.fit(X, y)
    X_trans = BUCK.transform(X)
    assert len(X_trans["MARRIAGE"].unique()) == 3
    assert len(X_trans["LIMIT_BAL"].unique()) == 3

    X["MARRIAGE_TRANS"] = X_trans["MARRIAGE"]
    assert X[np.isnan(X["MARRIAGE"])]["MARRIAGE_TRANS"].sum() == 0  # Sums to 0 as they are all in bucket 0

    assert "| Missing" in [f for f in BUCK.features_bucket_mapping_.get("MARRIAGE").labels.values()][0]
    assert "| Missing" in [f for f in BUCK.features_bucket_mapping_.get("LIMIT_BAL").labels.values()][1]

    BUCK = bucketer(n_bins=3, variables=["MARRIAGE", "LIMIT_BAL"], missing_treatment="most_frequent")
    BUCK.fit(X, y)

    for feature in ["MARRIAGE", "LIMIT_BAL"]:
        assert (
            "Missing"
            in BUCK.bucket_table(feature).sort_values("Count", ascending=False).reset_index(drop=True)["label"][0]
        )

    BUCK_risk = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"], missing_treatment="most_risky")
    BUCK_risk.fit(X, y)

    BUCK_norisk = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"])
    BUCK_norisk.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the riskiest bucket when missings are in a separate bucket
        riskiest_bucket = (
            BUCK_norisk.bucket_table(feature)[BUCK_norisk.bucket_table(feature)["bucket"] >=0]
            .sort_values("Event Rate", ascending=False)
            .reset_index(drop=True)["bucket"][0]
        )
        assert (
            "Missing"
            in BUCK_risk.bucket_table(feature)[
                BUCK_risk.bucket_table(feature)["bucket"] == riskiest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_risk = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"], missing_treatment="least_risky")
    BUCK_risk.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the safest bucket when missings are in a separate bucket
        safest_bucket = (
            BUCK_norisk.bucket_table(feature)[BUCK_norisk.bucket_table(feature)["bucket"] >=0]
            .sort_values("Event Rate", ascending=True)
            .reset_index(drop=True)["bucket"][0]
        )
        assert (
            "Missing"
            in BUCK_risk.bucket_table(feature)[
                BUCK_risk.bucket_table(feature)["bucket"] == safest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_neutral = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"], missing_treatment="neutral")
    BUCK_neutral.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket with WoE closest to 0
        table = BUCK_norisk.bucket_table(feature)
        table["WoE"] = np.abs(table["WoE"])
        closest_bucket = table[table["Count"] > 0].sort_values("WoE").reset_index(drop=True)["bucket"][0]
        assert (
            "Missing"
            in BUCK_neutral.bucket_table(feature)[
                BUCK_neutral.bucket_table(feature)["bucket"] == closest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_similar = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"], missing_treatment="similar")
    BUCK_similar.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket with WoE closest to 0
        table = BUCK_norisk.bucket_table(feature)
        missing_WoE = table[table["label"] == "Missing"]["WoE"].values[0]
        table["New_WoE"] = np.abs(table["WoE"] - missing_WoE)
        closest_bucket = table[table["label"] != "Missing"].sort_values("New_WoE").reset_index(drop=True)["bucket"][0]
        assert (
            "Missing"
            in BUCK_similar.bucket_table(feature)[
                BUCK_similar.bucket_table(feature)["bucket"] == closest_bucket
            ].reset_index()["label"][0]
        )
    
    BUCK_passthrough = bucketer(n_bins=3, variables=["MARRIAGE", "EDUCATION"], missing_treatment="passthrough")
    BUCK_passthrough.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket id with 'Missing' label
        table = BUCK_passthrough.bucket_table(feature)
        impute_missing = "Soup37120"
        table = table.fillna(impute_missing)
        missing_bucket = table[table["label"] == "Missing"]["bucket"].values[0]
        assert missing_bucket == impute_missing
        X_trans = BUCK_passthrough.transform(X)[[feature]]
        original_index_missings = X[X[feature].isnull()][feature].index
        trans_index_missings = X_trans[X_trans[feature].isnull()][feature].index
        assert all(original_index_missings[i] == trans_index_missings[i] for i in range(len(original_index_missings)))


@pytest.mark.parametrize("bucketer", BUCKETERS_WITHOUT_SET_BINS)
def test_missings_without_set(bucketer, df_with_missings) -> None:
    """Test all missing methods work for bucketers without set bins."""
    X = df_with_missings
    y = df_with_missings["default"].values

    BUCK = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="most_frequent")
    BUCK.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        assert (
            "Missing"
            in BUCK.bucket_table(feature).sort_values("Count", ascending=False).reset_index(drop=True)["label"][0]
        )

    BUCK_risk = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="most_risky")
    BUCK_risk.fit(X, y)

    BUCK_norisk = bucketer(variables=["MARRIAGE", "EDUCATION"])
    BUCK_norisk.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the riskiest bucket when missings are in a separate bucket
        riskiest_bucket = (
            BUCK_norisk.bucket_table(feature)[BUCK_norisk.bucket_table(feature)["bucket"] >=0]
            .sort_values("Event Rate", ascending=False)
            .reset_index(drop=True)["bucket"][0]
        )
        assert (
            "Missing"
            in BUCK_risk.bucket_table(feature)[
                BUCK_risk.bucket_table(feature)["bucket"] == riskiest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_risk = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="least_risky")
    BUCK_risk.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the safest bucket when missings are in a separate bucket
        safest_bucket = (
            BUCK_norisk.bucket_table(feature)[BUCK_norisk.bucket_table(feature)["bucket"] >=0]
            .sort_values("Event Rate", ascending=True)
            .reset_index(drop=True)["bucket"][0]
        )
        assert (
            "Missing"
            in BUCK_risk.bucket_table(feature)[
                BUCK_risk.bucket_table(feature)["bucket"] == safest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_neutral = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="neutral")
    BUCK_neutral.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket with WoE closest to 0
        table = BUCK_norisk.bucket_table(feature)
        table["WoE"] = np.abs(table["WoE"])
        closest_bucket = table[table["Count"] > 0].sort_values("WoE").reset_index(drop=True)["bucket"][0]
        assert (
            "Missing"
            in BUCK_neutral.bucket_table(feature)[
                BUCK_neutral.bucket_table(feature)["bucket"] == closest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_similar = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="similar")
    BUCK_similar.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket with WoE closest to 0
        table = BUCK_norisk.bucket_table(feature)
        missing_WoE = table[table["label"] == "Missing"]["WoE"].values[0]
        table["New_WoE"] = np.abs(table["WoE"] - missing_WoE)
        closest_bucket = table[table["label"] != "Missing"].sort_values("New_WoE").reset_index(drop=True)["bucket"][0]
        assert (
            "Missing"
            in BUCK_similar.bucket_table(feature)[
                BUCK_similar.bucket_table(feature)["bucket"] == closest_bucket
            ].reset_index()["label"][0]
        )

    BUCK_passthrough = bucketer(variables=["MARRIAGE", "EDUCATION"], missing_treatment="passthrough")
    BUCK_passthrough.fit(X, y)

    for feature in ["MARRIAGE", "EDUCATION"]:
        # look at the bucket id with 'Missing' label
        table = BUCK_passthrough.bucket_table(feature)
        impute_missing = "Soup37120"
        table = table.fillna(impute_missing)
        missing_bucket = table[table["label"] == "Missing"]["bucket"].values[0]
        assert missing_bucket == impute_missing
        X_trans = BUCK_passthrough.transform(X)[[feature]]
        original_index_missings = X[X[feature].isnull()][feature].index
        trans_index_missings = X_trans[X_trans[feature].isnull()][feature].index
        assert all(original_index_missings[i] == trans_index_missings[i] for i in range(len(original_index_missings)))



@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_type_error_input(bucketer, df):
    """Test that input is always a dataFrame."""
    y = df["default"].values
    X = df.drop(columns=["pet_ownership", "default"])
    pipe = make_pipeline(
        StandardScaler(),
        bucketer(variables=["BILL_AMT1"]),
    )
    with pytest.raises(AssertionError):
        pipe.fit_transform(X, y)


@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_is_not_fitted(bucketer):
    """
    Make sure we didn't make any mistakes when building a bucketer.
    """
    BUCK = bucketer()
    with pytest.raises(NotFittedError):
        check_is_fitted(BUCK)


@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_ui_bucketer(bucketer, df):
    """
    Make sure we didn't make any mistakes when building a bucketer.
    """
    BUCK = bucketer()
    # we drop BILL_AMT1 because that one needs prebucketing for some bucketers.
    df = df.drop(columns=["pet_ownership", "BILL_AMT1"])
    X = df
    y = df["default"].values
    X_trans = BUCK.fit_transform(X, y)

    uib = UserInputBucketer(BUCK.features_bucket_mapping_)
    assert X_trans.equals(uib.transform(X))


@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_zero_indexed(bucketer, df):
    """Test that bins are zero-indexed.

    When no missing values are present, no specials defined,
    bucket transforms should be zero indexed.

    Note that -2 (for 'other') is also allowed,
    f.e. OrdinalCategoricalBucketer puts less frequents cats there.
    """
    BUCK = bucketer()

    y = df["default"].values
    # we drop BILL_AMT1 because that one needs prebucketing for some bucketers.
    x_t = BUCK.fit_transform(df.drop(columns=["pet_ownership", "BILL_AMT1"]), y)
    assert x_t["MARRIAGE"].min() in [0, -2]
    assert x_t["EDUCATION"].min() in [0, -2]
    assert x_t["LIMIT_BAL"].min() in [0, -2]


@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_remainder_argument_no_bins(bucketer, df):
    """Test remainder argument works."""
    BUCK = bucketer(variables=["LIMIT_BAL"], remainder="drop")
    X = df
    y = df["default"].values

    BUCK.fit(X, y)
    X_trans = BUCK.transform(X)
    assert X_trans.columns == "LIMIT_BAL"

    BUCK = bucketer(variables=["LIMIT_BAL"], remainder="passthrough")
    BUCK.fit(X, y)
    X_trans = BUCK.transform(X)
    assert set(X_trans.columns) == set(X.columns)


@pytest.mark.parametrize("bucketer", ALL_BUCKETERS)
def test_summary_no_bins(bucketer, df):
    """Test summary works."""
    BUCK = bucketer(variables=["LIMIT_BAL"], remainder="passthrough")
    X = df
    y = df["default"].values
    BUCK.fit(X, y)
    summary_table = BUCK.summary()
    assert summary_table.shape[0] == 6
    assert set(summary_table.columns) == set(["column", "num_prebuckets", "num_buckets", "IV_score", "dtype"])
