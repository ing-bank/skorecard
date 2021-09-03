from skorecard.bucketers.bucketers import AsIsCategoricalBucketer, UserInputBucketer
from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.preprocessing import WoeEncoder
from skorecard.pipeline import BucketingProcess
from skorecard.pipeline.bucketing_process import _find_remapped_specials
from skorecard.utils import NotPreBucketedError, NotBucketObjectError, NotBucketedError

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import pytest


def test_bucketing_process_order(df):
    """Test that a NotPreBucketedError is raised if the bucketing pipeline is passed before the prebucketing."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]
    X = df[num_cols + cat_cols]
    y = df["default"].values

    # The prebucketing pipeline does not process all numerical columns in bucketing pipeline
    bucketing_process = BucketingProcess(
        specials={"LIMIT_BAL": {"=400000.0": [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=["LIMIT_BAL"], max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    with pytest.raises(NotBucketedError):
        bucketing_process.fit(X, y)

    # The bucketing pipeline does not process all numerical columns in prebucketing pipeline
    bucketing_process = BucketingProcess(
        specials={"LIMIT_BAL": {"=400000.0": [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=["LIMIT_BAL"], max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    with pytest.raises(NotPreBucketedError):
        bucketing_process.fit(X, y)


def test_non_bucketer_in_pipeline(df):
    """Test that putting a non-bucketer in bucket_process raises error."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]
    X = df[num_cols + cat_cols]
    y = df["default"].values

    with pytest.raises(NotBucketObjectError):
        # input validation is only done on fit, as per scikitlearn convention
        BucketingProcess(
            prebucketing_pipeline=make_pipeline(
                DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
            ),
            bucketing_pipeline=make_pipeline(
                OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
                OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
                LogisticRegression(),  # Should break the process
            ),
        ).fit(X, y)

    with pytest.raises(NotBucketObjectError):
        BucketingProcess(
            prebucketing_pipeline=make_pipeline(
                DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
                LogisticRegression(),  # Should break the process
            ),
            bucketing_pipeline=make_pipeline(
                OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
                OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
            ),
        ).fit(X, y)


def test_bucketing_optimization(df):
    """Test that the optimal bucketer returns less or equal number of unique buckets."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

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

    X_bins = bucketing_process.fit_transform(X, y)

    X_prebucketed = bucketing_process.pre_pipeline_.transform(X)
    for col in num_cols + cat_cols:
        assert X_bins[col].nunique() <= X_prebucketed[col].nunique()
        assert X_bins[col].nunique() > 1


def test_bucketing_with_specials(df):
    """Test that specials propogate."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    the_specials = {"LIMIT_BAL": {"=400000.0": [400000.0]}}

    bucketing_process = BucketingProcess(
        specials=the_specials,
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )

    _ = bucketing_process.fit_transform(X, y)

    # Make sure all the prebucketers have the specials assigned
    for step in bucketing_process.pre_pipeline_:
        assert step.specials == the_specials

    # Test the specials in the prebucket table
    prebuckets = bucketing_process.prebucket_table("LIMIT_BAL")
    assert prebuckets["Count"][0] == 45.0
    assert prebuckets["label"][0] == "Special: =400000.0"

    # Test the specials in the bucket table
    buckets = bucketing_process.bucket_table("LIMIT_BAL")
    assert buckets["Count"][0] == 45.0
    assert buckets["label"][0] == "Special: =400000.0"


def test_bucketing_process_in_pipeline(df):
    """Test that it works fine withing a sklearn pipeline."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

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

    pipeline = make_pipeline(bucketing_process, WoeEncoder(), LogisticRegression())

    pipeline.fit(X, y)
    preds = pipeline.predict_proba(X)

    assert preds.shape[0] == X.shape[0]


def test_bucketing_process_with_numerical_specials(df):
    """
    Test we get expected results for numerical specials.
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={"LIMIT_BAL": {"=400000.0": [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    bucketing_process.fit(X, y)

    table = bucketing_process.prebucket_table("LIMIT_BAL")
    assert len(table["bucket"].unique()) == 11
    assert table[["label"]].values[0] == "Special: =400000.0"

    table = bucketing_process.prebucket_table("MARRIAGE")
    assert table.shape[0] == 6


def test_bucketing_process_with_categorical_specials(df):
    """
    Test we get expected results for numerical specials.
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={"MARRIAGE": {"=0": [0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            DecisionTreeBucketer(variables=cat_cols, max_n_bins=100, min_bin_size=0.05),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    bucketing_process.fit(X, y)

    table = bucketing_process.prebucket_table("MARRIAGE")
    assert table.shape[0] == 4
    assert table["label"][0] == "Special: =0"


def test_uib_process(df):
    """
    Test merging works properly.
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={"MARRIAGE": {"=0": [0]}, "LIMIT_BAL": {"=400000.0": [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )

    bucketing_process.fit(X, y)

    uib = UserInputBucketer(bucketing_process.features_bucket_mapping_)

    X_trans = bucketing_process.transform(X)
    assert X_trans.equals(uib.transform(X))


def test_bucketing_process_summary(df):
    """
    Test bucketing process.

    Test we get expected results for .summary()
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={"MARRIAGE": {"=0": [0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            DecisionTreeBucketer(
                variables=cat_cols, max_n_bins=100, min_bin_size=0.05
            ),  # note this is wrong.. it's now treated as a numerical feature
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )

    bucketing_process.fit(X, y)
    table = bucketing_process.summary()
    assert set(table.columns) == set(["column", "num_prebuckets", "num_buckets", "IV_score", "dtype"])
    assert table[table["column"] == "pet_ownership"]["num_prebuckets"].values[0] == "not_prebucketed"
    assert table[table["column"] == "pet_ownership"]["num_buckets"].values[0] == "not_bucketed"
    assert len(table["dtype"].unique()) == 3
    assert all(table["IV_score"] >= 0)


def test_bucketing_process_remainder(df):
    """
    Test bucketing process.

    Test we get expected results for .summary()
    """
    y = df["default"]
    X = df.drop(columns=["default", "pet_ownership"])

    num_cols = ["LIMIT_BAL"]  # left out "BILL_AMT1"
    cat_cols = ["EDUCATION"]  # left out "MARRIAGE"

    # Don't drop variables.
    # If we specify bucketingprocess.variables
    # other variables should not be altered by prebucketing or bucketing pipeline
    bucketing_process = BucketingProcess(
        variables=num_cols + cat_cols,
        remainder="passthrough",
        specials={"EDUCATION": {"=0": [0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(max_n_bins=100, min_bin_size=0.05),  # note we didnt specify variables here!
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(max_n_bins=10, min_bin_size=0.05),  # note we didnt specify variables here!
        ),
    )
    new_X = bucketing_process.fit_transform(X, y)
    assert new_X["MARRIAGE"].equals(X["MARRIAGE"])
    assert len(new_X.columns) == 4

    # now with remainder drop
    bucketing_process = BucketingProcess(
        variables=num_cols + cat_cols,
        remainder="drop",
        specials={"EDUCATION": {"=0": [0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            DecisionTreeBucketer(
                variables=cat_cols, max_n_bins=100, min_bin_size=0.05
            ),  # note it's now treated as a numerical feature
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    new_X = bucketing_process.fit_transform(X, y)
    assert len(new_X.columns) == 2


def test_remapping_specials():
    """
    Test remapping works.
    """
    bucket_labels = {
        0: "(-inf, 25000.0)",
        1: "[25000.0, 45000.0)",
        2: "[45000.0, 55000.0)",
        3: "[55000.0, 75000.0)",
        4: "[75000.0, 85000.0)",
        5: "[85000.0, 105000.0)",
        6: "[105000.0, 145000.0)",
        7: "[145000.0, 175000.0)",
        8: "[175000.0, 225000.0)",
        9: "[225000.0, 275000.0)",
        10: "[275000.0, 325000.0)",
        11: "[325000.0, 385000.0)",
        12: "[385000.0, inf)",
        13: "Missing",
        14: "Special: =400000.0",
    }

    var_specials = {"=400000.0": [400000.0]}

    assert _find_remapped_specials(bucket_labels, var_specials) == {"=400000.0": [14]}

    assert _find_remapped_specials(bucket_labels, None) == {}
    assert _find_remapped_specials(None, None) == {}

    bucket_labels = {13: "Special: =12345 or 123456", 14: "Special: =400000.0"}

    var_specials = {"=400000.0": [400000.0], "=12345 or 123456": [12345, 123456]}
    assert _find_remapped_specials(bucket_labels, var_specials) == {"=400000.0": [14], "=12345 or 123456": [13]}
