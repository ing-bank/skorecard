from sklearn import linear_model as lm
import scipy
import numpy as np
import pandas as pd
from skorecard.utils import convert_sparse_matrix
from sklearn.utils.validation import check_is_fitted


class LogisticRegression(lm.LogisticRegression):
    """Extended Logistic Regression.

    Extends [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

    This class provides the following extra statistics, calculated on `.fit()` and accessible via `.get_stats()`:

    - `cov_matrix_`: covariance matrix for the estimated parameters.
    - `std_err_intercept_`: estimated uncertainty for the intercept
    - `std_err_coef_`: estimated uncertainty for the coefficients
    - `z_intercept_`: estimated z-statistic for the intercept
    - `z_coef_`: estimated z-statistic for the coefficients
    - `p_value_intercept_`: estimated p-value for the intercept
    - `p_value_coef_`: estimated p-value for the coefficients

    Example:

    ```python
    from skorecard.datasets import load_uci_credit_card
    from skorecard.bucketers import EqualFrequencyBucketer
    from skorecard.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    X, y = load_uci_credit_card(return_X_y=True)

    pipeline = Pipeline([
        ('bucketer', EqualFrequencyBucketer(n_bins=10)),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X, y)
    assert pipeline.named_steps['clf'].p_val_coef_[0][0] > 0

    pipeline.named_steps['clf'].get_stats()
    ```

    An example output of `.get_stats()`:

    Index     | Coef.     | Std.Err  |   z       | Pz
    --------- | ----------| ---------| ----------| ------------
    const     | -0.537571 | 0.096108 | -5.593394 | 2.226735e-08
    EDUCATION | 0.010091  | 0.044874 | 0.224876  | 8.220757e-01

    """  # noqa

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Fit the model.

        Overwrites [sklearn.linear_model.LogisticRegression.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

        In addition to the standard fit by sklearn, this function will compute the covariance of the coefficients.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y : array-like of shape (n_samples,)
                Target vector relative to X.
            sample_weight : array-like of shape (n_samples,) default=None
                Array of weights that are assigned to individual samples.
                If not provided, then each sample is given unit weight.

        Returns:
            self (LogisticRegression): Fitted estimator.
        """  # noqa
        X = convert_sparse_matrix(X)
        if isinstance(X, pd.DataFrame):
            self.names = ["const"] + [f for f in X.columns]
        else:
            self.names = ["const"] + [f"x{i}" for i in range(X.shape[1])]

        lr = super().fit(X, y, sample_weight=sample_weight, **kwargs)

        predProbs = self.predict_proba(X)

        # Design matrix -- add column of 1's at the beginning of your X matrix
        if lr.fit_intercept:
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_design = X

        p = np.product(predProbs, axis=1)
        self.cov_matrix_ = np.linalg.inv((X_design * p[..., np.newaxis]).T @ X_design)
        std_err = np.sqrt(np.diag(self.cov_matrix_)).reshape(1, -1)

        # In case fit_intercept is set to True, then in the std_error array
        # Index 0 corresponds to the intercept, from index 1 onwards it relates to the coefficients
        # If fit intercept is False, then all the values are related to the coefficients
        if lr.fit_intercept:

            self.std_err_intercept_ = std_err[:, 0]
            self.std_err_coef_ = std_err[:, 1:][0]

            self.z_intercept_ = self.intercept_ / self.std_err_intercept_

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_)) * 2

        else:
            self.std_err_intercept_ = np.array([np.nan])
            self.std_err_coef_ = std_err[0]

            self.z_intercept_ = np.array([np.nan])

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = np.array([np.nan])

        self.z_coef_ = self.coef_ / self.std_err_coef_
        self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2

        return self

    def get_stats(self) -> pd.DataFrame:
        """
        Puts the summary statistics of the fit() function into a pandas DataFrame.

        Returns:
            data (pandas DataFrame): The statistics dataframe, indexed by
                the column name
        """
        check_is_fitted(self)

        data = {
            "Coef.": (self.intercept_.tolist() + self.coef_.tolist()[0]),
            "Std.Err": (self.std_err_intercept_.tolist() + self.std_err_coef_.tolist()),
            "z": (self.z_intercept_.tolist() + self.z_coef_.tolist()[0]),
            "P>|z|": (self.p_val_intercept_.tolist() + self.p_val_coef_.tolist()[0]),
        }

        return pd.DataFrame(data, index=self.names)
