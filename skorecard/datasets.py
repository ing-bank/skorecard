import pkgutil
import io
import pandas as pd
from sklearn.datasets import fetch_openml


def load_uci_credit_card(return_X_y=False, as_frame=False):
    """Loads the UCI Credit Card Dataset.

    This dataset contains a sample of [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).

    Example:

    ```python
    from skorecard import datasets
    df = datasets.load_uci_credit_card(as_frame=True)
    ```

    Args:
        return_X_y (bool): If True, returns `(data, target)` instead of a dict object.
        as_frame (bool): give the pandas dataframe instead of X, y matrices (default=False).

    Returns: (pd.DataFrame, dict or tuple) features and target, with as follows:
        - if as_frame is True: returns pd.DataFrame with y as a target
        - return_X_y is True: returns a tuple: (X,y)
        - is both are false (default setting): returns a dictionary where the key `data` contains the features,
        and the key `target` is the target

    """  # noqa
    file = pkgutil.get_data("skorecard", "data/UCI_Credit_Card.zip")
    df = pd.read_csv(io.BytesIO(file), compression="zip")
    df = df.rename(columns={"default.payment.next.month": "default"})
    if as_frame:
        return df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT1", "default"]]
    X, y = (
        df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT1"]],
        df["default"].values,
    )
    if return_X_y:
        return X, y

    return {"data": X, "target": y}


def load_credit_card(return_X_y=False, as_frame=False):
    """
    Loads the complete UCI Credit Card Dataset, by fetching it from open_ml.

    Args:
        return_X_y:  (bool) If True, returns ``(data, target)`` instead of a dict object.
        as_frame: (bool) give the pandas dataframe instead of X, y matrices (default=False).

    Returns: (pd.DataFrame, dict or tuple) features and target, with as follows:
        - if as_frame is True: returns pd.DataFrame with y as a target
        - return_X_y is True: returns a tuple: (X,y)
        - is both are false (default setting): returns a dictionary where the key `data` contains the features,
        and the key `target` is the target

    """
    try:
        data = fetch_openml(
            name="default-of-credit-card-clients",
            data_home=None,
            cache=True,
            as_frame=as_frame,
            return_X_y=return_X_y,
        )
    except Exception as e:
        # update the error message with a more helpful message.
        error_msg = (
            "Cannot retrieve the dataset from repository. Make sure there is no firewall blocking "
            "the connection.\nAlternatively, download it manually from https://www.openml.org/d/42477"
        )
        raise type(e)(f"{e.args[0]}\n{error_msg}")

    # The target is by default encoded as a string.
    # Ensure it is returned as a integer.
    if as_frame:
        data = data["frame"]
        data["y"] = data["y"].astype(int)
    if return_X_y:
        X = data[0]
        y = data[1]
        y = y.astype(int)
        return X, y

    return data
