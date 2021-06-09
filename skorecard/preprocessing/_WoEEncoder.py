import numpy as np

from skorecard.bucketers.base_bucketer import BaseBucketer
from skorecard.metrics.metrics import woe_1d

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class WoeEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes unique values in features to their Weight of Evidence estimation.

    Only works for binary classification.

    The weight of evidence is given by: `np.log( p(1) / p(0) )`
    The target probability ratio is given by: `p(1) / p(0)`

    For example in the variable colour, if the mean of the target = 1 for blue is 0.8 and
    the mean of the target = 0 is 0.2, blue will be replaced by: np.log(0.8/0.2) = 1.386
    if log_ratio is selected. Alternatively, blue will be replaced by 0.8 / 0.2 = 4 if ratio is selected.

    More details:

    - [blogpost on weight of evidence](https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/)

    Example:

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import WoeEncoder

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    we = WoeEncoder(variables=['EDUCATION'])
    we.fit_transform(X, y)
    we.fit_transform(X, y)['EDUCATION'].value_counts()
    ```

    Credits: Some inspiration taken from [feature_engine.categorical_encoders](https://feature-engine.readthedocs.io/en/latest/encoding/index.html).
    """  # noqa

    def __init__(self, epsilon=0.0001, variables=[]):
        """
        Constructor for WoEBucketer.

        Args:
            epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
                calculation.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert epsilon >= 0

        self.epsilon = epsilon
        self.variables = variables

    def fit(self, X, y):
        """Calculate the WOE for every column.

        Args:
            X (np.array): (binned) features
            y (np.array): target
        """
        assert y is not None, "WoEBucketer needs a target y"
        assert len(np.unique(y)) == 2, "WoEBucketer is only suited for binary classification"

        self.variables = BaseBucketer._check_variables(X, self.variables)

        X = BaseBucketer._is_dataframe(X)
        # TODO: WoE should treat missing values as a separate bin and thus handled seamlessly.
        BaseBucketer._check_contains_na(X, self.variables)

        self.woe_mapping_ = {}

        for var in self.variables:
            t = woe_1d(X[var], y, epsilon=self.epsilon)

            self.woe_mapping_[var] = t["woe"].to_dict()

        return self

    def transform(self, X):
        """Transform X to weight of evidence encoding.

        Args:
            X (pd.DataFrame): dataset
        """
        check_is_fitted(self)
        X = BaseBucketer._is_dataframe(X)

        for feature in self.variables:
            woe_dict = self.woe_mapping_.get(feature)
            X[feature] = X[feature].map(woe_dict)

        return X
