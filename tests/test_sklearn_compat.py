import collections
import pytest
import itertools

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import estimator_checks

from skorecard.bucketers import (
    DecisionTreeBucketer,
    OptimalBucketer,
)
from skorecard.pipeline import (
    BucketingProcess,
)

from category_encoders.woe import WOEEncoder

from tests.conftest import CLASSIFIERS, TRANSFORMERS

# checks lists shamelessly copied from
# https://github.com/koaning/human-learn/blob/master/tests/conftest.py
classifier_checks = (
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_non_transformer_estimators_n_iter,
    estimator_checks.check_decision_proba_consistency,
)

transformer_checks = (
    estimator_checks.check_transformer_data_not_an_array,
    estimator_checks.check_transformer_general,
    estimator_checks.check_transformers_unfitted,
)

general_checks = (
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters,
)

nonmeta_checks = (
    estimator_checks.check_estimators_pickle,
    estimator_checks.check_estimators_dtypes,
    estimator_checks.check_fit_score_takes_y,
    estimator_checks.check_dtype_object,
    estimator_checks.check_estimators_fit_returns_self,
    estimator_checks.check_complex_data,
    estimator_checks.check_estimators_empty_data_messages,
    estimator_checks.check_pipeline_consistency,
    estimator_checks.check_estimators_nan_inf,
    estimator_checks.check_estimators_overwrite_params,
    estimator_checks.check_estimator_sparse_data,
)


def select_tests(include, exclude=[]):
    """Return an iterable of include with all tests whose name is not in exclude.

    Credits: https://github.com/koaning/human-learn/blob/master/tests/conftest.py
    """
    for test in include:
        if test.__name__ not in exclude:
            yield test


def flatten(nested_iterable):
    """
    Returns an iterator of flattened values from an arbitrarily nested iterable.

    Usage:

    ```python
    from hulearn.common import flatten
    res1 = list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    res2 = list(flatten(['test1', ['test2']]))
    assert res1 == ['test1', 'test2', 'a', 'b', 'c', 'd']
    assert res2 == ['test1', 'test2']
    ```

    Credits: https://github.com/koaning/human-learn/blob/master/hulearn/common.py
    """  # noqa
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


@pytest.mark.parametrize(
    "transformer,test_fn",
    list(
        itertools.product(
            TRANSFORMERS,
            select_tests(
                include=flatten([general_checks, transformer_checks, nonmeta_checks]),
                exclude=[
                    "check_fit2d_1sample",
                    "check_methods_subset_invariance",
                    "check_estimators_nan_inf",
                    "check_estimators_empty_data_messages",
                    "check_transformer_data_not_an_array",
                    "check_dtype_object",
                    "check_complex_data",
                    "check_fit1d",
                    "check_transformers_unfitted",
                ],
            ),
        )
    ),
)
def test_transformer_checks(transformer, test_fn):
    """
    Runs a scikitlearn check on a skorecard transformer.
    """
    t = transformer()
    test_fn(t.__class__.__name__, t)


@pytest.mark.parametrize(
    "classifier,test_fn",
    list(
        itertools.product(
            CLASSIFIERS,
            select_tests(
                include=flatten([general_checks, classifier_checks, nonmeta_checks]),
                exclude=[
                    "check_methods_subset_invariance",
                    "check_fit2d_1sample",
                    "check_fit2d_1feature",
                    "check_classifier_data_not_an_array",
                    "check_classifiers_one_label",
                    "check_classifiers_classes",
                    "check_classifiers_train",
                    "check_supervised_y_2d",
                    "check_estimators_pickle",
                    "check_pipeline_consistency",
                    "check_fit2d_predict1d",
                    "check_fit1d",
                    "check_dtype_object",
                    "check_complex_data",
                    "check_estimators_empty_data_messages",
                    "check_estimators_nan_inf",
                    "check_estimator_sparse_data",
                    "check_supervised_y_no_nan",
                    "check_estimators_partial_fit_n_features",
                ],
            ),
        )
    ),
)
def test_classifier_checks(classifier, test_fn):
    """
    Runs a scikitlearn check on a skorecard transformer.
    """
    clf = classifier()
    test_fn(clf.__class__.__name__, clf)


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


def test_cv_pipeline(df):
    """
    Another CV.
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

    pipe = make_pipeline(
        bucketing_process, WOEEncoder(cols=X.columns), LogisticRegression(solver="liblinear", random_state=0)
    )

    with pytest.warns(None) as _:
        cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")

    # also make sure no warnings were raised
    # assert len(record) == 0
