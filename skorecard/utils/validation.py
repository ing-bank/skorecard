import pandas as pd
from sklearn.utils import check_array
from typing import Dict
import warnings


def is_fitted(estimator) -> bool:
    """
    Checks if an estimator is fitted.

    Loosely taken from
    https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/utils/validation.py#L1034
    """  # noqa

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    return len(attrs) > 0


def ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure X is a pandas DataFrame.
    """
    # checks if the input is a dataframe.
    if not isinstance(X, pd.DataFrame):
        # Convert X to pd.DataFrame. Not recommended,
        # as you'll lose column name info.
        # but bucketer will still work on numpy matrix
        # also required for full scikitlearn compatibility
        X = X.copy()
        X = check_array(X, force_all_finite=False, accept_sparse=False, dtype=None)
        X = pd.DataFrame(X)
        X.columns = list(X.columns)  # sometimes columns can be a RangeIndex..
    else:
        # Create a copy
        # important not to transform the original dataset.
        X = X.copy()

    if X.shape[0] == 0:
        raise ValueError("Dataset has no rows!")
    if X.shape[1] == 0:
        raise ValueError(f"0 feature(s) (shape=({X.shape[0]}, 0)) while a minimum of 1 is required.")

    return X


def check_args(args: Dict, obj):
    """
    Checks if keys from args dictionary are valid args to an object.

    Note: this assumes 'obj' is scikit-learn compatible and thus has .get_params() implemented.
    """
    valid_args = obj().get_params()
    for arg in args.keys():
        if arg not in valid_args:
            msg = f"Argument '{arg}' is not a valid argument for object '{obj}'"
            warnings.warn(msg)
