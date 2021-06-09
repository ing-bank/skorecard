import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that performs selection of variables from a pandas dataframe.

    Useful in pipelines, where we require a step that selects feautures.

    Example:

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import ColumnSelector

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    cs = ColumnSelector(variables=['EDUCATION'])
    assert cs.fit_transform(X, y).columns == ['EDUCATION']
    ```
    """

    def __init__(self, variables=None):
        """Transformer constructor.

        Args:
            variables: list of columns to select. Default value is set to None - in this case, there is no selection of
                columns.
        """
        self.variables = variables

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Here to be compliant with the sklearn API, does not fit anything.
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"
        return self

    def transform(self, X):
        """
        Selects the columns.

        Args:
            X (pd.DataFrame): Dataset
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"

        if self.variables is not None:
            return X[self.variables]
        else:
            return X
