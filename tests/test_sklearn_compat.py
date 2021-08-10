from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from skorecard.bucketers import (
    DecisionTreeBucketer,
    OptimalBucketer,
)
from skorecard.pipeline import (
    BucketingProcess,
)


def test_cross_val(df):
    """
    Test using CV.

    When defining specials combined with using CV, we would get a

    ValueError: Specials should be defined on the BucketingProcess level,
    remove the specials from DecisionTreeBucketer(specials={'EDUCATION': {'Some specials': [1, 2]}})

    This unit test ensures specials with CV keep working.
    """
    y = df["default"].values
    X = df.drop(columns=["default", "pet_ownership"])

    specials = {"EDUCATION": {"Some specials": [1, 2]}}

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(max_n_bins=100, min_bin_size=0.05),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(max_n_bins=10, min_bin_size=0.05),
        ),
        specials=specials,
    )

    pipe = make_pipeline(bucketing_process, StandardScaler(), LogisticRegression(solver="liblinear", random_state=0))

    cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
