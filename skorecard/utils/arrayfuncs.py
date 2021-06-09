import numpy as np
import pandas as pd
import scipy
from skorecard.utils.exceptions import DimensionalityError


def reshape_1d_to_2d(x):
    """Converts/reshapes the input x to a numpy array o (n,1).

    Args:
        x: list, numpy array, pandas dataframe (one column only), pandas series

    Returns: numpy array of shape (n,1)
    """
    X_array = None
    if isinstance(x, list):
        X_array = np.array(x)
    if isinstance(x, np.ndarray):
        if x.ndim > 1 and x.shape[1] > 1:
            raise DimensionalityError("Expected one column only. Can't reshape")
        X_array = x
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1:
            X_array = x.values.flatten()
        else:
            raise DimensionalityError("Expected one column only. Can't reshape")
    if isinstance(x, pd.core.series.Series):
        X_array = x.values

    return X_array.reshape(-1, 1)


def convert_sparse_matrix(x):
    """
    Converts a sparse matrix to a numpy array.

    This can prevent problems arising from, e.g. OneHotEncoder.

    Args:
        x: numpy array, sparse matrix

    Returns:
        numpy array of x
    """
    if scipy.sparse.issparse(x):
        return x.toarray()
    else:
        return x
