import pandas as pd


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
        return pd.DataFrame(X)
    else:
        # Create a copy
        # important not to transform the original dataset.
        return X.copy()
