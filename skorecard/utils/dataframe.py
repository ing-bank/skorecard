import pandas as pd

from typing import List, Tuple
from skorecard.utils import NotInstalledError

try:
    import dabl as db
except ModuleNotFoundError:
    db = NotInstalledError("dabl")


def detect_types(X: pd.DataFrame) -> Tuple[List, List]:
    """Detects numerical and categorical columns.

    Wrapper around `dabl.detect_types`.

    ```python
    from skorecard import datasets
    from skorecard.utils import detect_types

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    num_cols, cat_cols = detect_types(X)
    print(f"Categorical column(s) = {cat_cols}")
    print(f"Numerical column(s) = {num_cols}")
    ```
    Args:
        X (pd.DataFrame): Input dataframe
    """
    assert isinstance(X, pd.DataFrame)
    detected_types = db.detect_types(X)

    cat_columns = X.columns[(detected_types["categorical"] is True) | (detected_types["low_card_int"] is True)]
    num_columns = X.columns[(detected_types["continuous"] is True) | (detected_types["dirty_float"] is True)]

    return list(num_columns), list(cat_columns)
