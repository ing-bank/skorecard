from typing import List
from skorecard.utils.validation import ensure_dataframe
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

    def __init__(self, variables: List = []):
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
        # scikit-learn requires checking that X has same shape on transform
        # this is because scikit-learn is still positional based (no column names used)
        self.n_train_features_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Selects the columns.

        Args:
            X (pd.DataFrame): Dataset
        """
        X = ensure_dataframe(X)
        if hasattr(self, "n_train_features_"):
            if X.shape[1] != self.n_train_features_:
                msg = f"Number of features in X ({X.shape[1]}) is different "
                msg += f"from the number of features in X during fit ({self.n_train_features_})"
                raise ValueError(msg)

        if len(self.variables) > 0:
            return X[self.variables]
        else:
            return X

    def _more_tags(self):
        """
        Estimator tags are annotations of estimators that allow programmatic inspection of their capabilities.

        See https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """  # noqa
        return {"requires_fit": False}
