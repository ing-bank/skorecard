from typing import Optional, Dict, List, TypeVar
import pandas as pd
import numpy as np
import itertools
import pathlib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

from skorecard.reporting.plotting import PlotBucketMethod
from skorecard.reporting.report import BucketTableMethod, SummaryMethod
from skorecard.features_bucket_mapping import FeaturesBucketMapping
from skorecard.bucket_mapping import BucketMapping
from skorecard.reporting import build_bucket_table
from skorecard.utils.exceptions import NotInstalledError
from skorecard.utils.validation import is_fitted

# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")

try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")


from skorecard.apps.app_layout import add_basic_layout
from skorecard.apps.app_callbacks import add_bucketing_callbacks
from skorecard.utils.validation import ensure_dataframe

PathLike = TypeVar("PathLike", str, pathlib.Path)


class BaseBucketer(BaseEstimator, TransformerMixin, PlotBucketMethod, BucketTableMethod, SummaryMethod):
    """Base class for bucket transformers."""

    @staticmethod
    def _check_y(y):
        # checks that y is an appropriate type and shape
        if y is None:
            return y

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise AttributeError("If passing y as a DataFrame, it must be 1 column.")
            y = y.values.reshape(-1)

        if isinstance(y, pd.core.series.Series):
            y = y.values

        if isinstance(y, np.ndarray):
            if y.ndim > 2:
                raise AttributeError("If passing y as a Numpy array, y must be a 1-dimensional")
            elif y.ndim == 2:
                if y.shape[1] != 1:
                    raise AttributeError("If passing y as a Numpy array, y must be a 1-dimensional")
                else:
                    y = y.reshape(-1, 1)

        y = check_array(y, ensure_2d=False)

        return y

    @staticmethod
    def _is_allowed_missing_treatment(missing_treatment):
        # checks if the argument for missing_values is valid
        allowed_str_missing = [
            "separate",
            "most_frequent",
            "most_risky",
            "least_risky",
            "neutral",
            "similar",
            "passthrough",
        ]

        if type(missing_treatment) == str:
            if missing_treatment not in allowed_str_missing:
                raise ValueError(f"missing_treatment must be in {allowed_str_missing} or a dict")

        elif type(missing_treatment) == dict:
            for _, v in enumerate(missing_treatment):
                if missing_treatment[v] < 0:
                    raise ValueError("As an integer, missing_treatment must be greater than 0")
                elif type(missing_treatment[v]) != int:
                    raise ValueError("Values of the missing_treatment dict must be integers")

        else:
            raise ValueError(f"missing_treatment must be in {allowed_str_missing} or a dict")

    @staticmethod
    def _check_contains_na(X, variables: Optional[List]):

        has_missings = X[variables].isnull().any()
        vars_missing = has_missings[has_missings].index.tolist()

        if vars_missing:
            raise ValueError(f"The variables {vars_missing} contain missing values. Consider using an imputer first.")

    @staticmethod
    def _check_variables(X, variables: Optional[List]):
        assert isinstance(variables, list)
        if len(variables) == 0:
            variables = list(X.columns)
        else:
            for var in variables:
                assert var in list(X.columns), f"Column {var} not present in X"
        assert variables is not None and len(variables) > 0
        return variables

    @staticmethod
    def _filter_specials_for_fit(X, y, specials: Dict):
        """
        We need to filter out the specials from a vector.

        Because we don't want to use those values to determine bin boundaries.
        """
        flt_vals = list(itertools.chain(*specials.values()))
        flt = X.isin(flt_vals)
        X_out = X[~flt]

        if y is not None:
            y_out = y[~flt]
        else:
            y_out = y
        return X_out, y_out

    def _find_missing_bucket(self, feature):
        """
        Used for when missing_treatment is in:
        ["most_frequent", "most_risky", "least_risky", "neutral", "similar", "passthrough"]

        Calculates the new bucket for us to put the missing values in.
        """
        if self.missing_treatment == "most_frequent":
            most_frequent_row = (
                self.bucket_tables_[feature].sort_values("Count", ascending=False).reset_index(drop=True).iloc[0]
            )
            if most_frequent_row["label"] != "Missing":
                missing_bucket = int(most_frequent_row["bucket_id"])
            else:
                # missings are already the most common bucket, pick the next one
                missing_bucket = int(
                    self.bucket_tables_[feature]
                    .sort_values("Count", ascending=False)
                    .reset_index(drop=True)["bucket_id"][1]
                )
        elif self.missing_treatment in ["most_risky", "least_risky"]:
            if self.missing_treatment == "least_risky":
                ascending = True
            else:
                ascending = False
            # if fitted with .fit(X) and not .fit(X, y)
            if "Event" not in self.bucket_tables_[feature].columns:
                raise AttributeError("bucketer must be fit with y to determine the risk rates")

            missing_bucket = int(
                self.bucket_tables_[feature][self.bucket_tables_[feature]["bucket_id"] >=0]
                .sort_values("Event Rate", ascending=ascending)
                .reset_index(drop=True)
                .iloc[0]["bucket_id"]
            )

        elif self.missing_treatment in ["neutral"]:
            table = self.bucket_tables_[feature]
            table["WoE"] = np.abs(table["WoE"])
            missing_bucket = int(
                table[table["Count"] > 0].sort_values("WoE").reset_index(drop=True).iloc[0]["bucket_id"]
            )

        elif self.missing_treatment in ["similar"]:
            table = self.bucket_tables_[feature]
            missing_WoE = table[table["label"] == "Missing"]["WoE"].values[0]
            table["New_WoE"] = np.abs(table["WoE"] - missing_WoE)
            missing_bucket = int(
                table[table["label"] != "Missing"].sort_values("New_WoE").reset_index(drop=True).iloc[0]["bucket_id"]
            )

        elif self.missing_treatment in ["passthrough"]:
            missing_bucket = np.nan

        else:
            raise AssertionError(f"Invalid missing treatment '{self.missing_treatment}' specified")


        return missing_bucket

    def _filter_na_for_fit(self, X: pd.DataFrame, y):
        """
        We need to filter out the missing values from a vector.

        Because we don't want to use those values to determine bin boundaries.

        Note pd.DataFrame.isna and pd.DataFrame.isnull are identical
        """
        # let's also treat infinite values as NA
        # scikit-learn's check_estimator might throw those at us
        with pd.option_context("mode.use_inf_as_na", True):
            flt = pd.isna(X).values
        X_out = X[~flt]
        if y is not None and len(y) > 0:
            y_out = y[~flt]
        else:
            y_out = y
        return X_out, y_out

    @staticmethod
    def _verify_specials_variables(specials: Dict, variables: List) -> None:
        """
        Make sure all specials columns are also in the data.
        """
        diff = set(specials.keys()).difference(set(variables))
        if len(diff) > 0:
            raise ValueError(f"Features {diff} are defined in the specials dictionary, but not in the variables.")

    def fit(self, X, y=None):
        """Fit X, y."""
        # Do __init__ validation.
        # See https://scikit-learn.org/stable/developers/develop.html#parameters-and-init
        assert self.remainder in ["passthrough", "drop"]
        if hasattr(self, "missing_treatment"):
            self._is_allowed_missing_treatment(self.missing_treatment)
        if hasattr(self, "n_bins"):
            assert isinstance(self.n_bins, int)
            assert self.n_bins >= 1
        if hasattr(self, "encoding_method"):
            assert self.encoding_method in ["frequency", "ordered"]
        if hasattr(self, "tol"):
            if self.tol < 0 or self.tol > 1:
                raise ValueError("tol takes values between 0 and 1")
        if hasattr(self, "max_n_categories"):
            if self.max_n_categories is not None:
                if self.max_n_categories < 0 or not isinstance(self.max_n_categories, int):
                    raise ValueError("max_n_categories takes only positive integer numbers")

        # Do data validation
        X = ensure_dataframe(X)
        y = self._check_y(y)
        if y is not None:
            assert len(y) == X.shape[0], "y and X not same length"
            # Store the classes seen during fit
            self.classes_ = unique_labels(y)

        # scikit-learn requires checking that X has same shape on transform
        # this is because scikit-learn is still positional based (no column names used)
        self.n_train_features_ = X.shape[1]

        self.variables_ = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)
        self.features_bucket_mapping_ = FeaturesBucketMapping()
        self.bucket_tables_ = {}

        for feature in self.variables_:
            # filter specials for the fit
            if feature in self.specials.keys():
                special = self.specials[feature]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt, y_flt = X[feature], y
                special = {}

            # filter missing values for the fit
            X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)

            # Find the splits
            # This method is implemented by each bucketer
            assert isinstance(X_flt, pd.Series)
            splits, right = self._get_feature_splits(feature, X=X_flt, y=y_flt, X_unfiltered=X)

            self._update_column_fit(X, y, feature, special, splits, right)

        if self.get_statistics: 
            self._generate_summary(X, y)

        return self

    def _update_column_fit(self, X, y, feature, special, splits, right, generate_summary=False):
        """
        Extract out part of the fit for a column.

        Useful when we want to interactively update the fit.
        """
        # Deal with missing values
        missing_bucket = None
        if isinstance(self.missing_treatment, dict):
            missing_bucket = self.missing_treatment.get(feature)

        self.features_bucket_mapping_[feature] = BucketMapping(
            feature_name=feature,
            type=self.variables_type,
            missing_bucket=missing_bucket,
            map=splits,
            right=right,
            specials=special,

        )

        # Calculate the bucket table
        if self.get_statistics: 
            self.bucket_tables_[feature] = build_bucket_table(
                X,
                y,
                column=feature,
                bucket_mapping=self.features_bucket_mapping_.get(feature),
            )

            if self.missing_treatment in [
                "most_frequent",
                "most_risky",
                "least_risky",
                "neutral",
                "similar",
                "passthrough",
            ]:
                missing_bucket = self._find_missing_bucket(feature=feature)
                # Repeat above procedure now we know the bucket distribution
                self.features_bucket_mapping_[feature] = BucketMapping(
                    feature_name=feature,
                    type=self.variables_type,
                    missing_bucket=missing_bucket,
                    map=splits,
                    right=right,
                    specials=special,

                )

                # Recalculate the bucket table with the new bucket for missings
                self.bucket_tables_[feature] = build_bucket_table(
                    X,
                    y,
                    column=feature,
                    bucket_mapping=self.features_bucket_mapping_.get(feature),
            )

        if generate_summary:
            self._generate_summary(X, y)

    def fit_interactive(self, X, y=None, mode="external", **server_kwargs):
        """
        Fit a bucketer and then interactive edit the fit using a dash app.

        Note we are using a [jupyterdash](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e) app,
        which supports 3 different modes:

        - 'external' (default): Start dash server and print URL
        - 'inline': Start dash app inside an Iframe in the jupyter notebook
        - 'jupyterlab': Start dash app as a new tab inside jupyterlab

        """
        # We need to make sure we only fit if not already fitted
        # This prevents a user loosing manually defined boundaries
        # when re-running .fit_interactive()
        if not is_fitted(self):
            self.fit(X, y)

        self.app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        add_basic_layout(self)
        add_bucketing_callbacks(self, X, y)
        self.app.run_server(mode=mode, **server_kwargs)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transforms an array into the corresponding buckets fitted by the Transformer.

        Args:
            X (pd.DataFrame): dataframe which will be transformed into the corresponding buckets
            y (array): target

        Returns:
            df (pd.DataFrame): dataset with transformed features
        """
        check_is_fitted(self)
        X = ensure_dataframe(X)
        y = self._check_y(y)
        if y is not None:
            assert len(y) == X.shape[0], "y and X not same length"
        # If bucketer was fitted, make sure X has same columns as train
        if hasattr(self, "n_train_features_"):
            if X.shape[1] != self.n_train_features_:
                raise ValueError("number of features in transform is different from the number of features in fit")

        # Some bucketers do not have a .fit() method
        # and if user did not specify any variables
        # use all the variables defined in the features_bucket_mapping
        if not hasattr(self, "variables_"):
            if self.variables == []:
                self.variables_ = list(self.features_bucket_mapping_.maps.keys())
            else:
                self.variables_ = self.variables

        for feature in self.variables_:
            bucket_mapping = self.features_bucket_mapping_.get(feature)
            X[feature] = bucket_mapping.transform(X[feature])

        if self.remainder == "drop":
            return X[self.variables_]
        else:
            return X

    def predict(self, X: pd.DataFrame):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (pd.DataFrame): The numerical data which will be transformed into the corresponding buckets

        Returns:
            y (np.array): Transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X)

    def predict_proba(self, X: pd.DataFrame):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (pd.DataFrame): The numerical data which will be transformed into the corresponding buckets

        Returns:
            yhat (np.array): transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X)

    def save_yml(self, fout: PathLike) -> None:
        """
        Save the features bucket to a yaml file.

        Args:
            fout: file output
        """
        check_is_fitted(self)
        if isinstance(self.features_bucket_mapping_, dict):
            FeaturesBucketMapping(self.features_bucket_mapping_).save_yml(fout)
        else:
            self.features_bucket_mapping_.save_yml(fout)

    def _more_tags(self):
        """
        Estimator tags are annotations of estimators that allow programmatic inspection of their capabilities.

        See https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """  # noqa
        return {"binary_only": True, "allow_nan": True}
