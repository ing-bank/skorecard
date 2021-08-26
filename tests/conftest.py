import random
import numpy as np
import pytest

from skorecard import datasets

import skorecard.linear_model
import skorecard.bucketers
import skorecard.pipeline
import skorecard.preprocessing
import skorecard.metrics
import skorecard.bucket_mapping
import skorecard.utils
import skorecard.rescale
import skorecard.datasets


BUCKETERS = [
    skorecard.bucketers.OptimalBucketer,
    skorecard.bucketers.EqualWidthBucketer,
    skorecard.bucketers.AgglomerativeClusteringBucketer,
    skorecard.bucketers.EqualFrequencyBucketer,
    skorecard.bucketers.DecisionTreeBucketer,
    skorecard.bucketers.OrdinalCategoricalBucketer,
    skorecard.bucketers.UserInputBucketer,
    skorecard.bucketers.AsIsCategoricalBucketer,
    skorecard.bucketers.AsIsNumericalBucketer,
    skorecard.bucketers.UserInputBucketer,
]

TRANSFORMERS = BUCKETERS + [
    skorecard.pipeline.BucketingProcess,
    skorecard.preprocessing.ColumnSelector,
    skorecard.preprocessing.WoeEncoder,
]

CLASSIFIERS = [
    skorecard.Skorecard,
    skorecard.linear_model.LogisticRegression,
]

# List of all classes and functions we want tested for the docstrings
CLASSES_TO_TEST = (
    TRANSFORMERS
    + CLASSIFIERS
    + [
        skorecard.pipeline.KeepPandas,
        skorecard.pipeline.SkorecardPipeline,
        skorecard.rescale.ScoreCardPoints,
        skorecard.features_bucket_mapping.FeaturesBucketMapping,
        skorecard.bucket_mapping.BucketMapping,
        skorecard.utils.DimensionalityError,
        skorecard.pipeline.SkorecardPipeline,
    ]
)
FUNCTIONS_TO_TEST = [
    skorecard.utils.reshape_1d_to_2d,
    skorecard.pipeline.get_features_bucket_mapping,
    skorecard.reporting.build_bucket_table,
    skorecard.reporting.iv,
    skorecard.reporting.psi,
    skorecard.pipeline.to_skorecard_pipeline,
    skorecard.datasets.load_uci_credit_card,
]


@pytest.fixture()
def df():
    """Generate dataframe."""
    df = datasets.load_uci_credit_card(as_frame=True)
    # Add a fake categorical
    pets = ["no pets"] * 3000 + ["cat lover"] * 1500 + ["dog lover"] * 1000 + ["rabbit"] * 498 + ["gold fish"] * 2
    random.Random(42).shuffle(pets)
    df["pet_ownership"] = pets

    return df


@pytest.fixture()
def df_with_missings(df):
    """
    Add missing values to above df.
    """
    df_with_missings = df.copy()

    for col in ["EDUCATION", "MARRIAGE", "BILL_AMT1", "LIMIT_BAL", "pet_ownership"]:
        df_with_missings.loc[df_with_missings.sample(frac=0.2, random_state=42).index, col] = np.nan

    # Make sure there are 8 unique values (7 unique plus some NA)
    assert len(df_with_missings["EDUCATION"].unique()) == 8
    # Make sure there are NAs
    assert any([np.isnan(x) for x in df_with_missings["EDUCATION"].unique().tolist()])

    return df_with_missings
