from skorecard.features_bucket_mapping import FeaturesBucketMapping
from skorecard import Skorecard
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, OrdinalCategoricalBucketer
from skorecard.pipeline import BucketingProcess
from skorecard.utils import BucketerTypeError
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_is_fitted
import pytest


def run_checks(X, y, bucketer, features, expected_probas):
    """
    Run some standard asserts on Skorecard instance.
    """
    skorecard_model = Skorecard(bucketing=bucketer, variables=features, calculate_stats=True)
    skorecard_model.fit(X, y)

    # make sure sklearn recognizes this as fitted
    check_is_fitted(skorecard_model)
    check_is_fitted(skorecard_model.pipeline_.steps[-1][1])
    if isinstance(skorecard_model.bucketing_, Pipeline):
        check_is_fitted(skorecard_model.bucketing_.steps[0][1])
    else:
        check_is_fitted(skorecard_model.bucketing_)

    # Make sure proper attribures are there
    assert isinstance(skorecard_model.bucketing_.features_bucket_mapping_, FeaturesBucketMapping)
    assert isinstance(skorecard_model.bucketing_.bucket_tables_, dict)

    top_probas = skorecard_model.predict_proba(X)[:2]
    np.testing.assert_array_almost_equal(top_probas, expected_probas, decimal=2)

    # assure naming convention is fixed and ordered
    assert [names for names in skorecard_model.pipeline_.named_steps] == [
        "bucketer",
        "encoder",
        "column_selector",
        "model",
    ]

    if len(features) == 0:
        features = X.columns.tolist()

    # test bucketers did not change
    if bucketer is not None:
        assert (
            skorecard_model.bucket_transform(X)[features].head(3).equals(bucketer.fit_transform(X, y)[features].head(3))
        )

    # check that columns are properly selected
    assert skorecard_model.bucket_transform(X).shape == X.shape
    assert skorecard_model.woe_transform(X).shape == X.shape
    # Last transformer selects the features
    assert skorecard_model.pipeline_[:-1].transform(X).shape == (X.shape[0], len(features))

    # test that the stats showcase only the selected features
    assert skorecard_model.get_stats().index.tolist() == ["const"] + features

    # Test bucket table works
    assert (
        skorecard_model.bucket_table("LIMIT_BAL").shape[0]
        == skorecard_model.pipeline_[:1].transform(X)["LIMIT_BAL"].nunique() + 1
    )

    # prebucketing methods
    if not isinstance(skorecard_model.bucketing_, BucketingProcess):
        # test a BucketerTypeError is raised if the plot_prebucket('LIMIT_BAL') function is called (not defined
        # #for standard bucketers)
        with pytest.raises(BucketerTypeError):
            skorecard_model.plot_prebucket("LIMIT_BAL")

        # test a BucketerTypeError is raised if the prebucket_table function is called
        #  (not defined for standard bucketers)
        with pytest.raises(BucketerTypeError):
            skorecard_model.prebucket_table("LIMIT_BAL")
    else:
        prebucket_table = skorecard_model.prebucket_table("LIMIT_BAL")
        bucket_table = skorecard_model.bucket_table("LIMIT_BAL")
        assert prebucket_table.shape[0] >= bucket_table.shape[0]


def test_skorecard_with_num_bucketers(df):
    """Test a workflow, with numerical bucketer and numerical features alone."""
    X = df.drop("default", axis=1)
    y = df["default"]

    features = ["LIMIT_BAL", "BILL_AMT1"]
    bucketer = DecisionTreeBucketer(variables=features, max_n_bins=5)
    expected_probas = np.array([[0.841, 0.159], [0.738, 0.262]])

    run_checks(X, y, bucketer, features, expected_probas)

    # and with features not defined
    features = []
    expected_probas = np.array([[0.88, 0.12], [0.73, 0.27]])
    run_checks(X, y, bucketer, features, expected_probas)


def test_passing_kwargs(df):
    """Test passing keyword args to LR."""
    X = df.drop("default", axis=1)
    y = df["default"]

    features = ["LIMIT_BAL", "BILL_AMT1"]
    bucketer = DecisionTreeBucketer(variables=features, max_n_bins=5)

    skorecard_model = Skorecard(
        bucketing=bucketer,
        variables=features,
        lr_kwargs={"penalty": "none", "C": 1, "multi_class": "ovr", "n_jobs": 1, "max_iter": int(1e3)},
    )
    skorecard_model.fit(X, y)

    assert skorecard_model.bucket_transform(X).shape == X.shape
    assert skorecard_model.woe_transform(X).shape == X.shape


def test_skorecard_with_bucketing_process(df):
    """Test a workflow, with Bucketin Process."""
    X = df.drop("default", axis=1)
    y = df["default"]

    features = ["LIMIT_BAL", "BILL_AMT1"]
    cat_features = [col for col in X.columns if col not in features]

    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=features, max_n_bins=100),
        OrdinalCategoricalBucketer(variables=cat_features, tol=0.01),
    )
    bucketing_pipeline = make_pipeline(
        OptimalBucketer(variables=features, max_n_bins=5, min_bin_size=0.08),
        OptimalBucketer(variables=cat_features, variables_type="categorical", max_n_bins=5, min_bin_size=0.08),
    )
    bucketer = BucketingProcess(prebucketing_pipeline=prebucketing_pipeline, bucketing_pipeline=bucketing_pipeline)

    expected_probas = np.array([[0.851, 0.149], [0.747, 0.252]])

    run_checks(X, y, bucketer, features, expected_probas)

    # and with features not defined
    features = []
    expected_probas = np.array([[0.88, 0.12], [0.76, 0.24]])
    run_checks(X, y, bucketer, features, expected_probas)


def test_skorecard_with_pipeline_of_buckets(df):
    """Test a workflow, with numerical pipeline of bucketers."""
    X = df.drop("default", axis=1)
    y = df["default"]
    features = ["LIMIT_BAL", "BILL_AMT1"]
    cat_features = [col for col in X.columns if col not in features]

    bucketer = make_pipeline(
        DecisionTreeBucketer(variables=features, max_n_bins=5),
        OrdinalCategoricalBucketer(variables=cat_features, tol=0.05),
    )

    expected_probas = np.array([[0.841, 0.159], [0.738, 0.262]])
    run_checks(X, y, bucketer, features, expected_probas)

    features = []
    expected_probas = np.array([[0.877, 0.122], [0.734, 0.266]])
    run_checks(X, y, bucketer, features, expected_probas)


def test_default_skorecard_class(df):
    """Test a workflow, when no bucketer is defined."""
    X = df.drop("default", axis=1)
    y = df["default"]
    features = ["LIMIT_BAL", "BILL_AMT1"]

    skorecard_model = Skorecard(verbose=0, variables=features)
    skorecard_model.fit(X, y)
    assert isinstance(skorecard_model.bucketing_, BucketingProcess)

    bucketer = None
    expected_probas = np.array([[0.862, 0.138], [0.748, 0.252]])
    run_checks(X, y, bucketer, features, expected_probas)

    bucketer = None
    features = []
    expected_probas = np.array([[0.895, 0.105], [0.752, 0.248]])
    run_checks(X, y, bucketer, features, expected_probas)


def test_random_state(df):
    """Test that all of the random_state shenanigans work as expected."""
    X = df.drop("default", axis=1)
    y = df["default"]
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    # Test assigning a random state
    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100),
        OrdinalCategoricalBucketer(variables=cat_cols, tol=0.01)
    )
    bucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=5, min_bin_size=0.08)
    )

    bucketer = BucketingProcess(prebucketing_pipeline=prebucketing_pipeline, bucketing_pipeline=bucketing_pipeline)
    skorecard_model = Skorecard(bucketing=bucketer, variables=num_cols + cat_cols, random_state=411)
    skorecard_model.fit(X, y)
    assert bucketer.pre_pipeline_.named_steps['decisiontreebucketer'].random_state == 411
    assert bucketer.pipeline_.named_steps['decisiontreebucketer'].random_state == 411
    assert skorecard_model.pipeline_.named_steps['model'].random_state == 411

    # Test overwriting a random state
    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, random_state=123),
        OrdinalCategoricalBucketer(variables=cat_cols, tol=0.01)
    )
    bucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, random_state=123),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=5, min_bin_size=0.08)
    )

    bucketer = BucketingProcess(prebucketing_pipeline=prebucketing_pipeline, bucketing_pipeline=bucketing_pipeline)
    skorecard_model = Skorecard(bucketing=bucketer, variables=num_cols + cat_cols, random_state=411)
    skorecard_model.fit(X, y)
    assert bucketer.pre_pipeline_.named_steps['decisiontreebucketer'].random_state == 411
    assert bucketer.pipeline_.named_steps['decisiontreebucketer'].random_state == 411
    assert skorecard_model.pipeline_.named_steps['model'].random_state == 411

    # Test keeping a random state
    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, random_state=123),
        OrdinalCategoricalBucketer(variables=cat_cols, tol=0.01)
    )
    bucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, random_state=123),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=5, min_bin_size=0.08)
    )

    bucketer = BucketingProcess(prebucketing_pipeline=prebucketing_pipeline, bucketing_pipeline=bucketing_pipeline)
    skorecard_model = Skorecard(bucketing=bucketer, variables=num_cols + cat_cols, lr_kwargs={"random_state": 411})
    skorecard_model.fit(X, y)
    assert bucketer.pre_pipeline_.named_steps['decisiontreebucketer'].random_state == 123
    assert bucketer.pipeline_.named_steps['decisiontreebucketer'].random_state == 123
    assert skorecard_model.pipeline_.named_steps['model'].random_state == 411

    # Test with Pipeline

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucket_pipe = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=5),
        OrdinalCategoricalBucketer(variables=cat_cols, tol=0.05)
    )

    skorecard_model = Skorecard(bucketing=bucket_pipe, variables=num_cols + cat_cols, random_state=411)
    skorecard_model.fit(X, y)
    assert skorecard_model.pipeline_.named_steps['bucketer']['decisiontreebucketer'].random_state == 411
    assert skorecard_model.pipeline_.named_steps['model'].random_state == 411
