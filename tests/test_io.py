import pytest
import os
import yaml

from skorecard.bucketers import (
    DecisionTreeBucketer,
    UserInputBucketer,
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
    OptimalBucketer,
    OrdinalCategoricalBucketer,
    AsIsNumericalBucketer,
    AsIsCategoricalBucketer,
)
from skorecard.pipeline import BucketingProcess, to_skorecard_pipeline

from sklearn.pipeline import make_pipeline
from contextlib import contextmanager


BUCKETERS_WITH_SET_BINS = [EqualWidthBucketer, AgglomerativeClusteringBucketer, EqualFrequencyBucketer]

AS_IS_BUCKETERS = [AsIsNumericalBucketer, AsIsCategoricalBucketer]


@contextmanager
def working_directory(path):
    """
    Temporary working directories.

    A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    Usage:

    ```python
    # Do something in original directory
    with working_directory('/my/new/path'):
        # Do something in new directory
    # Back to old directory
    ````

    Credits: https://gist.github.com/nottrobin/3d675653244f8814838a
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.fixture
def fpath(tmpdir):
    """
    Temporary path.
    """
    return tmpdir.mkdir("sub")


def test_ordinal_bucketer_to_file(df, tmpdir):
    """Test that saving the bucketer to file and loading it back in the user input bucketer works."""
    features = ["EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"]

    ocb = OrdinalCategoricalBucketer(variables=["EDUCATION", "MARRIAGE"], tol=0.1)
    X_trans = ocb.fit_transform(X, y)[features]

    # Test save to yaml in bucketer
    ocb.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)

    # Test save to yaml with str path
    with working_directory(tmpdir):
        ocb.save_yml("buckets.yml")
        buckets_yaml = yaml.safe_load(open("buckets.yml", "r"))
        X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
        assert X_trans.equals(X_trans_yaml)

        # test alternative flow
        ocb.save_yml("buckets.yml")
        X_trans_yaml = UserInputBucketer("buckets.yml").transform(X)
        assert X_trans.equals(X_trans_yaml)


def test_decision_tree_bucketer_to_file(df, tmpdir):
    """Test that saving the bucketer to file and loading it back in the user input bucketer works."""
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"]

    tbt = DecisionTreeBucketer(
        variables=features, max_n_bins=5, dt_kwargs={"criterion": "entropy", "min_impurity_decrease": 0.001}
    )
    X_trans = tbt.fit_transform(X, y)

    # Test save to yaml in bucketer
    tbt.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_bucketers_with_n_bins_to_file(bucketer, df, tmpdir) -> None:
    """Test that saving the bucketer to file and loading it back in the user input bucketer works."""
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"].values

    BUCK = bucketer(n_bins=3, variables=features)
    BUCK.fit(X)
    X_trans = BUCK.fit_transform(X, y)

    # Test save to yaml in bucketer
    BUCK.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)


def test_bucketers_with_sklearn_pipeline(df, tmpdir):
    """
    Test bucketers saved in a sklearn pipeline.
    """
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"]

    bucketing = make_pipeline(
        DecisionTreeBucketer(
            variables=[features[0]], max_n_bins=5, dt_kwargs={"criterion": "entropy", "min_impurity_decrease": 0.001}
        ),
        DecisionTreeBucketer(variables=[features[1]], max_n_bins=3),
    )

    X_trans = bucketing.fit_transform(X, y)
    bucketing = to_skorecard_pipeline(bucketing)

    # Test save to yaml
    bucketing.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)


def test_bucketing_process_to_file(df, tmpdir):
    """Test that saving the bucketing_process to file and loading it back in the user input bucketer works."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            OrdinalCategoricalBucketer(variables=cat_cols, tol=0.01),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.1),
        ),
    )

    X_trans = bucketing_process.fit_transform(X, y)

    # Test save to yaml in bucketer
    bucketing_process.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    assert buckets_yaml == bucketing_process.features_bucket_mapping_.as_dict()
    # Test transforms work the same
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)


@pytest.mark.parametrize("bucketer", AS_IS_BUCKETERS)
def test_as_is_bucketers_to_file(bucketer, df, tmpdir) -> None:
    """Test that saving the bucketer to file and loading it back in the user input bucketer works."""
    features = ["LIMIT_BAL", "EDUCATION", "MARRIAGE"]  # "BILL_AMT1" has >100 unique values, so cannot be AsIs Bucketed
    X = df[features]
    y = df["default"].values

    BUCK = bucketer()
    BUCK.fit(X)
    X_trans = BUCK.fit_transform(X, y)

    # Test save to yaml in bucketer
    BUCK.save_yml(open(os.path.join(tmpdir, "buckets.yml"), "w"))
    buckets_yaml = yaml.safe_load(open(os.path.join(tmpdir, "buckets.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml).transform(X)
    assert X_trans.equals(X_trans_yaml)

    # Test inside a skorecard pipeline
    pipe = to_skorecard_pipeline(make_pipeline(BUCK))
    pipe.fit(X, y)
    assert X_trans.equals(pipe.transform(X))
    pipe.save_yml(open(os.path.join(tmpdir, "buckets3.yml"), "w"))
    buckets_yaml3 = yaml.safe_load(open(os.path.join(tmpdir, "buckets3.yml"), "r"))
    X_trans_yaml = UserInputBucketer(buckets_yaml3).transform(X)
    assert X_trans.equals(X_trans_yaml)
