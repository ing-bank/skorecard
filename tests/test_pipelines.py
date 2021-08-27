import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from skorecard.bucketers import (
    EqualWidthBucketer,
    EqualFrequencyBucketer,
    OrdinalCategoricalBucketer,
    DecisionTreeBucketer,
    OptimalBucketer,
)
from skorecard.pipeline import (
    get_features_bucket_mapping,
    KeepPandas,
    BucketingProcess,
    find_bucketing_step,
)
from skorecard.pipeline.pipeline import to_skorecard_pipeline, SkorecardPipeline
from skorecard.bucket_mapping import BucketMapping
from skorecard.utils import BucketingPipelineError


@pytest.mark.filterwarnings("ignore:sklearn.")
def test_keep_pandas(df, caplog):
    """Tests the KeepPandas() class."""
    y = df["default"].values
    X = df.drop(columns=["default", "pet_ownership"])

    bucket_pipeline = make_pipeline(
        StandardScaler(),
        EqualWidthBucketer(n_bins=5, variables=["LIMIT_BAL", "BILL_AMT1"]),
    )
    # Doesn't work, input should be a pandas dataframe.
    with pytest.raises(AssertionError):
        bucket_pipeline.fit(X, y)

    bucket_pipeline = make_pipeline(
        KeepPandas(StandardScaler()),
        EqualWidthBucketer(n_bins=5, variables=["LIMIT_BAL", "BILL_AMT1"]),
    )

    with pytest.raises(NotFittedError):
        bucket_pipeline.transform(X)

    bucket_pipeline.fit(X, y)
    assert type(bucket_pipeline.transform(X)) == pd.DataFrame

    bucket_pipeline = ColumnTransformer(
        [
            ("categorical_preprocessing", OrdinalCategoricalBucketer(), ["EDUCATION", "MARRIAGE"]),
            ("numerical_preprocessing", EqualWidthBucketer(n_bins=5), ["LIMIT_BAL", "BILL_AMT1"]),
        ],
        remainder="passthrough",
    )

    # Make sure warning is raised
    caplog.clear()
    KeepPandas(make_pipeline(bucket_pipeline))
    assert "sklearn.compose.ColumnTransformer can change" in caplog.text

    # Make sure warning is raised
    caplog.clear()
    KeepPandas(bucket_pipeline)
    assert "sklearn.compose.ColumnTransformer can change" in caplog.text

    assert type(KeepPandas(bucket_pipeline).fit_transform(X, y)) == pd.DataFrame


def test_bucketing_pipeline(df):
    """Test the class."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    prebucket_pipeline = make_pipeline(DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05))

    bucket_pipeline = make_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )

    pipe = make_pipeline(prebucket_pipeline, bucket_pipeline)
    pipe.fit(X, y)
    # Make sure we can fit it twice
    pipe.fit(X, y)

    # make sure transforms work.
    pipe.transform(X)
    pipe.fit_transform(X, y)


def test_find_coarse_classing_step(df):
    """Tests coarse classing step."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )
    pipeline = make_pipeline(bucketing_process)
    assert find_bucketing_step(pipeline, identifier="bucketingprocess") == 0


def test_get_features_bucket_mapping(df):
    """Test retrieving info from sklearn pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    nested_pipeline = make_pipeline(
        make_pipeline(EqualWidthBucketer(n_bins=5, variables=["LIMIT_BAL", "BILL_AMT1"])),
        OrdinalCategoricalBucketer(variables=["EDUCATION", "MARRIAGE"]),
    )

    with pytest.raises(NotFittedError):
        get_features_bucket_mapping(nested_pipeline)

    nested_pipeline.fit(X, y)
    bm = get_features_bucket_mapping(nested_pipeline)
    assert (
        bm.get("EDUCATION").map
        == BucketMapping(feature_name="EDUCATION", type="categorical", map={2: 0, 1: 1, 3: 2}, right=True).map
    )


# TODO: write tests with different kinds of sklearn pipelines
# - ColumnTransformer and ColumnSelector usage


def test_make_pipeline(df):
    """Make sure bucketers work inside a pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    pipe = make_pipeline(
        EqualWidthBucketer(n_bins=4, variables=["LIMIT_BAL"]),
        EqualFrequencyBucketer(n_bins=7, variables=["BILL_AMT1"]),
    )
    new_X = pipe.fit_transform(X, y)
    assert isinstance(new_X, pd.DataFrame)


def test_pipeline_errors(df):
    """Make sure incorrect input also throws correct errors in pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    bu = EqualWidthBucketer(n_bins=4, variables=["LIMIT_BAL", "BILL_AMT1"])
    with pytest.raises(NotFittedError):
        bu.transform(X)  # not fitted yet
    with pytest.raises(ValueError):
        bu.fit_transform(np.array([1, 2, 3]), y)


def test_pipeline_has_no_duplicated_features(df):
    """Test that the columns in bucketers in a scikit pipeline are unique, otherwise an error is raised."""
    y = df["default"].values
    X = df.drop(columns=["default"])
    features_1 = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION"]
    features_2 = ["EDUCATION", "MARRIAGE", "BILL_AMT1"]

    bucketer = make_pipeline(
        DecisionTreeBucketer(variables=features_1, max_n_bins=5),
        OrdinalCategoricalBucketer(variables=features_2, tol=0.05),
    )

    # during fit
    with pytest.raises(BucketingPipelineError):
        to_skorecard_pipeline(bucketer).fit(X, y)

    # after fit
    bucketer.fit(X, y)
    with pytest.raises(BucketingPipelineError):
        to_skorecard_pipeline(bucketer)

    with pytest.raises(BucketingPipelineError):
        SkorecardPipeline(
            [
                ("dtb", DecisionTreeBucketer(variables=features_1, max_n_bins=5)),
                ("ocb", OrdinalCategoricalBucketer(variables=features_2, tol=0.05)),
            ]
        ).fit(X, y)

    # What if one of the pipelines applies to all
    # And the other to some?
    # That should also be a conflict
    bucketer = make_pipeline(
        DecisionTreeBucketer(max_n_bins=5),
        OrdinalCategoricalBucketer(variables=features_2, tol=0.05),
    )
    with pytest.raises(BucketingPipelineError):
        p = to_skorecard_pipeline(bucketer)
        feat = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
        p.fit(X[feat], y)


def test_skorecard_pipeline(df):
    """Test that the skorecard Pipeline returns the same results."""
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"].values

    features_1 = ["LIMIT_BAL", "BILL_AMT1"]
    features_2 = ["EDUCATION", "MARRIAGE"]

    bucketer = make_pipeline(
        DecisionTreeBucketer(variables=features_1, max_n_bins=5),
        OrdinalCategoricalBucketer(variables=features_2, tol=0.05),
    )

    bucketer.fit(X, y)

    sk_pipe = to_skorecard_pipeline(bucketer)

    assert bucketer.transform(X).equals(sk_pipe.transform(X))
    assert isinstance(sk_pipe.summary(), pd.DataFrame)
    assert isinstance(sk_pipe.bucket_table("LIMIT_BAL"), pd.DataFrame)
    assert isinstance(sk_pipe.bucket_table("EDUCATION"), pd.DataFrame)

    # Now with specials
    bucketer = make_pipeline(
        DecisionTreeBucketer(
            variables=features_1, max_n_bins=5, specials={"BILL_AMT1": {"Some specials": [201800, 76445, 79244]}}
        ),
        OrdinalCategoricalBucketer(variables=features_2, tol=0.05, specials={"EDUCATION": {"Some specials": [1, 2]}}),
    )
    bucketer.fit(X, y)
    sk_pipe = to_skorecard_pipeline(bucketer)


# def test_bucket_transformer_bin_count_list(df):
#     """Test the exception is raised in scikit-learn pipeline."""
#     with pytest.raises(AttributeError):
#         transformer = ColumnTransformer(
#             transformers=[
#                 ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#                 ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#                 ("quantile", QuantileBucketTransformer(bin_count=[10]), [3]),
#             ],
#             remainder="passthrough",
#         )
#         transformer.fit_transform(df.values)

#     return None


# def test_bucket_transformer_exception(df):
#     """Test the exception is raised in scikit-learn pipeline."""
#     with pytest.raises(DimensionalityError):
#         transformer = ColumnTransformer(
#             transformers=[
#                 ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#                 ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#                 ("quantile", QuantileBucketTransformer(bin_count=10), [2, 3]),
#             ],
#             remainder="passthrough",
#         )
#         transformer.fit_transform(df.values)

#     return None


# def test_bucket_transformer(df):
#     """Test that we can utilise the main bucket transformers in a scikit-learn pipeline."""
#     transformer = ColumnTransformer(
#         transformers=[
#             ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#             ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#             ("quantile_0", QuantileBucketTransformer(bin_count=10), [2]),
#             ("quantile_1", QuantileBucketTransformer(bin_count=6), [3]),
#         ],
#         remainder="passthrough",
#     )

#     X = transformer.fit_transform(df.values)

#     # Test only non-categorical variables
#     assert len(np.unique(X[:, 2])) == 10
#     assert len(np.unique(X[:, 3])) == 6
#     assert np.all(X[:, 4] == df["default"].values)

#     return None
