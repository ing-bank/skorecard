import numpy as np
from collections import defaultdict

from skorecard.bucketers.base_bucketer import BaseBucketer
from skorecard.metrics.metrics import woe_1d
from skorecard.utils.validation import ensure_dataframe

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import warnings


class WoeEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes unique values in features to their Weight of Evidence estimation.

    **This class has been deprecated in favor of category_encoders.woe.WOEEncoder**

    Only works for binary classification (target y has 0 and 1 values).

    The weight of evidence is given by: `np.log( p(1) / p(0) )`
    The target probability ratio is given by: `p(1) / p(0)`

    For example in the variable colour, if the mean of the target = 1 for blue is 0.8 and
    the mean of the target = 0 is 0.2, blue will be replaced by: np.log(0.8/0.2) = 1.386
    if log_ratio is selected. Alternatively, blue will be replaced by 0.8 / 0.2 = 4 if ratio is selected.

    More formally:

    - for each unique value 𝑥,  consider the corresponding rows in the training set
    - compute what percentage of positives is in these rows, compared to the whole set
    - compute what percentage of negatives is in these rows, compared to the whole set
    - take the ratio of these percentages
    - take the natural logarithm of that ratio to get the weight of evidence corresponding to  𝑥,  so that  𝑊𝑂𝐸(𝑥)  is either positive or negative according to whether  𝑥  is more representative of positives or negatives

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

    def __init__(self, epsilon=0.0001, variables=[], handle_unknown="value"):
        """
        Constructor for WoEEncoder.

        Args:
            epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
                calculation.
            variables (list): The features to bucket. Uses all features if not defined.
            handle_unknown (str): How to handle any new values encountered in X on transform().
                options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
        """
        self.epsilon = epsilon
        self.variables = variables
        self.handle_unknown = handle_unknown

        warnings.warn(
            "This encoder will be deprecated. Please use category_encoders.woe.WOEEncoder instead.", DeprecationWarning
        )

    def fit(self, X, y):
        """Calculate the WOE for every column.

        Args:
            X (np.array): (binned) features
            y (np.array): target
        """
        assert self.epsilon >= 0
        # Check data
        X = ensure_dataframe(X)
        assert y is not None, "WoEBucketer needs a target y"
        y = BaseBucketer._check_y(y)

        y = y.astype(float)
        if len(np.unique(y)) > 2:
            raise AssertionError("WoEBucketer is only suited for binary classification")
        self.variables_ = BaseBucketer._check_variables(X, self.variables)

        # WoE currently does not support NAs
        # This is also flagged in self._more_tags()
        # We could treat missing values as a separate bin (-1) and thus handle seamlessly.
        BaseBucketer._check_contains_na(X, self.variables_)

        # scikit-learn requires checking that X has same shape on transform
        # this is because scikit-learn is still positional based (no column names used)
        self.n_train_features_ = X.shape[1]

        self.woe_mapping_ = {}
        for var in self.variables_:
            t = woe_1d(X[var], y, epsilon=self.epsilon)

            woe_dict = t["woe"].to_dict()
            # If new categories encountered, returns WoE = 0
            if self.handle_unknown == "value":
                woe_dict = defaultdict(int, woe_dict)

            self.woe_mapping_[var] = woe_dict

        return self

    def transform(self, X):
        """Transform X to weight of evidence encoding.

        Args:
            X (pd.DataFrame): dataset
        """
        assert self.handle_unknown in ["value", "error", "return_nan"]
        check_is_fitted(self)
        X = ensure_dataframe(X)

        if X.shape[1] != self.n_train_features_:
            msg = f"Number of features in X ({X.shape[1]}) is different "
            msg += f"from the number of features in X during fit ({self.n_train_features_})"
            raise ValueError(msg)

        for feature in self.variables_:
            woe_dict = self.woe_mapping_.get(feature)
            if self.handle_unknown == "error":
                new_cats = [x for x in list(X[feature].unique()) if x not in list(woe_dict.keys())]
                if len(new_cats) > 0:
                    msg = "WoEEncoder encountered unknown new categories "
                    msg += f"in column {feature} on .transform(): {new_cats}"
                    raise AssertionError(msg)

            X[feature] = X[feature].map(woe_dict)

        return X

    def _more_tags(self):
        """
        Estimator tags are annotations of estimators that allow programmatic inspection of their capabilities.

        See https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """  # noqa
        return {"binary_only": True, "allow_nan": False}
