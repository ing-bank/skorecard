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
