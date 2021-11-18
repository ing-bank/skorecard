import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer


def woe_1d(X, y, epsilon=0.00001):
    """Compute the weight of evidence on a 1-dimensional array.

    Args:
        X (np.array): bucketed dataframe
        y (np.array): target
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.

    Returns: (dictionary)
        - bins: indices of unique values of X
        - woe_values: calculated weight of evidence for every unique bin
        - counts_0: count of entries per bin where y==0
        - counts_1: count of entries per bin where y==1
    """
    X = X.copy().reset_index(drop=True)
    if not isinstance(y, pd.Series):
        if y.shape[0] == X.shape[0]:
            y = pd.Series(y).reset_index(drop=True)
        else:
            raise ValueError(f"y has {y.shape[0]}, but expected {X.shape[0]}")

    # Ensure classes in y start at zero
    y = y - min(y)

    df = pd.concat([X, y], axis=1, ignore_index=True)
    df.columns = ["feat", "target"]

    total_pos = df["target"].sum()
    total_neg = df.shape[0] - total_pos
    df["non_target"] = np.where(df["target"] == 1, 0, 1)

    pos = (df.groupby(["feat"])["target"].sum() / total_pos) + epsilon
    neg = (df.groupby(["feat"])["non_target"].sum() / total_neg) + epsilon

    # Make sure to give informative error when dividing by zero error occurs
    msg = """
    One of the unique values in X has no occurances of the %s class.
    Set epsilon to a very small value, or use a more coarse binning.
    """
    if any(neg == 0):
        raise ZeroDivisionError(msg % "negative")
    if any(pos == 0):
        raise ZeroDivisionError(msg % "positive")

    t = pd.concat([pos, neg], axis=1)
    t["woe"] = np.log(t["non_target"] / t["target"])

    if t["woe"].isnull().any():
        msg = "Woe Calculation produced NaNs! "
        msg += "Perhaps check your target distribution contains more than 1 class."
        raise ValueError(msg)

    return t


def _IV_score(y_test, y_pred, epsilon=0.0001, digits=None):
    """Using the unique values in y_pred, calculates the information value for the specific np.array.

    Args:
        y_test: (np.array), binary features, target
        y_pred: (np.array), predictions, indices of the buckets where the IV should be computed
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
        digits: (int): number of significant decimal digits in the IV calculation

    Returns:
        iv (float): information value

    """
    df = woe_1d(y_pred, y_test, epsilon=epsilon)

    iv = ((df["non_target"] - df["target"]) * df["woe"]).sum()

    if digits:
        assert isinstance(digits, int), f"Digits must be an integer! Passed a {type(int)} variable"

        iv = np.round(iv, digits)
    return iv


@make_scorer
def IV_scorer(y_test, y_pred):
    """Decorated version, makes the IV score usable for sklearn grid search pipelines.

    Using the unique values in y_pred, calculates the information value for the specific np.array.

    Args:
        y_test: (np.array), binary features, target
        y_pred: (np.array), predictions, indices of the buckets where the IV should be computed

    Returns:
        iv (float): information value

    """
    return _IV_score(y_test, y_pred, digits=3)
