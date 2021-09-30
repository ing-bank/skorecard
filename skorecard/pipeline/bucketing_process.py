import pathlib
import pandas as pd
import numpy as np
import warnings

from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline

from skorecard.utils import NotPreBucketedError, NotBucketedError
from skorecard.pipeline import to_skorecard_pipeline
from skorecard.pipeline.pipeline import _get_all_steps
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
from skorecard.reporting import build_bucket_table
from skorecard.reporting.report import BucketTableMethod, SummaryMethod
from skorecard.reporting.plotting import PlotBucketMethod, PlotPreBucketMethod
from skorecard.features_bucket_mapping import FeaturesBucketMapping, merge_features_bucket_mapping
from skorecard.utils.validation import is_fitted, ensure_dataframe
from skorecard.utils.exceptions import NotInstalledError

from typing import Dict, TypeVar, List


# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")

try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")


from skorecard.apps.app_layout import add_bucketing_process_layout
from skorecard.apps.app_callbacks import add_bucketing_process_callbacks


PathLike = TypeVar("PathLike", str, pathlib.Path)


class BucketingProcess(
    BaseEstimator,
    TransformerMixin,
    BucketTableMethod,
    PlotBucketMethod,
    PlotPreBucketMethod,
    SummaryMethod,
):
    """
    A two-step bucketing pipeline allowing for pre-bucketing before bucketing.

    Often you want to pre-bucket features (f.e. to 100 buckets) before bucketing to a smaller set.
    This brings some additional challenges around propagating specials and defining a bucketer that is able to go from raw data to final bucket.
    This class facilicates the process and also provides all regular methods and attributes:

    - `.summary()`: See which columns are bucketed
    - `.plot_bucket()`: Plot buckets of a column
    - `.bucket_table()`: Table with buckets of a column
    - `.save_to_yaml()`: Save information necessary for bucketing to a YAML file
    - `.features_bucket_mapping_`: Access bucketing information

    Example:

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, AsIsCategoricalBucketer
    from skorecard.pipeline import BucketingProcess
    from sklearn.pipeline import make_pipeline

    df = datasets.load_uci_credit_card(as_frame=True)
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            AsIsCategoricalBucketer(variables=cat_cols),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05),
        )
    )

    bucketing_process.fit(X, y)

    # Details
    bucketing_process.summary() # all vars, and # buckets
    bucketing_process.bucket_table("LIMIT_BAL")
    bucketing_process.plot_bucket("LIMIT_BAL")
    bucketing_process.prebucket_table("LIMIT_BAL")
    bucketing_process.plot_prebucket("LIMIT_BAL")
    ```
    """  # noqa

    def __init__(
        self,
        prebucketing_pipeline=make_pipeline(DecisionTreeBucketer(max_n_bins=50, min_bin_size=0.02)),
        bucketing_pipeline=make_pipeline(OptimalBucketer(max_n_bins=6, min_bin_size=0.05)),
        variables: List = [],
        specials: Dict = {},
        random_state: int = None,
        remainder="passthrough",
    ):
        """
        Define a BucketingProcess to first prebucket and then bucket multiple columns in one go.

        Args:
            prebucketing_pipeline (Pipeline): The scikit-learn pipeline that does pre-bucketing.
                Defaults to an all-numeric DecisionTreeBucketer pipeline.
            bucketing_pipeline (Pipeline): The scikit-learn pipeline that does bucketing.
                Defaults to an all-numeric OptimalBucketer pipeline.
                Must transform same features as the prebucketing pipeline.
            variables (list): The features to bucket. Uses all features if not defined.
            specials: (nested) dictionary of special values that require their own binning.
                Will merge when specials are also defined in any bucketers in a (pre)bucketing pipeline, and overwrite in case there are shared keys.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            remainder (str): How we want the non-specified columns to be transformed. It must be in ["passthrough", "drop"].
                passthrough (Default): all columns that were not specified in "variables" will be passed through.
                drop: all remaining columns that were not specified in "variables" will be dropped.
        """  # noqa
        # Save original input params
        # We overwrite the input later, so we need to save
        # original so we can clone instances
        # https://scikit-learn.org/dev/developers/develop.html#cloning
        # https://scikit-learn.org/dev/developers/develop.html#get-params-and-set-params
        # Assigning the variable in the init to the attribute with the same name is a requirement of
        # sklearn.base.BaseEstimator. See the notes in
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator
        self.prebucketing_pipeline = prebucketing_pipeline
        self.bucketing_pipeline = bucketing_pipeline
        self.remainder = remainder
        self.variables = variables
        self.specials = specials
        self.random_state = random_state

    @property
    def name(self):
        """
        To be able to identity the bucketingprocess in a pipeline.
        """
        return "bucketingprocess"

    def fit(self, X, y=None):
        """
        Fit the prebucketing and bucketing pipeline with `X`, `y`.

        Args:
            X (pd.DataFrame): Data to fit on.
            y (np.array, optional): target. Defaults to None.
        """
        X = ensure_dataframe(X)

        # input validation
        assert self.remainder in ["passthrough", "drop"]

        # Convert to skorecard pipelines
        # This does some checks on the pipelines
        # and adds some convenience methods to the pipeline.
        self.pre_pipeline_ = to_skorecard_pipeline(deepcopy(self.prebucketing_pipeline))
        self.pipeline_ = to_skorecard_pipeline(deepcopy(self.bucketing_pipeline))

        # Add/Overwrite specials to all pre-bucketers
        for step in _get_all_steps(self.pre_pipeline_):
            if hasattr(step, "specials") and len(step.specials) != 0 and len(self.specials) != 0:
                # note, specials defined BucketingProcess level
                # will overwrite any specials on bucketer level.
                warnings.warn(f"Overwriting specials of {step} with specials of bucketingprocess", UserWarning)
                step.specials = {**step.specials, **self.specials}
            else:
                step.specials = self.specials

            if len(self.variables) != 0:
                if len(step.variables) != 0:
                    warnings.warn(f"Overwriting variables of {step} with variables of bucketingprocess", UserWarning)
                step.variables = self.variables

            # Overwrite random_state to bucketers
            if hasattr(step, "random_state") and self.random_state is not None:
                if step.random_state is not None:
                    warnings.warn(f"Overwriting random_state of {step} with random_state of bucketingprocess",
                                  UserWarning)
                step.random_state = self.random_state

        # Overwrite variables to all bucketers
        if len(self.variables) != 0:
            for step in _get_all_steps(self.pipeline_):
                if len(step.variables) != 0:
                    warnings.warn(f"Overwriting variables of {step} with variables of bucketingprocess", UserWarning)
                step.variables = self.variables

        # Overwrite random_state to bucketers
        for step in _get_all_steps(self.pipeline_):
            if hasattr(step, "random_state") and self.random_state is not None:
                if step.random_state is not None:
                    warnings.warn(f"Overwriting random_state of {step} with random_state of bucketingprocess",
                                  UserWarning)
                step.random_state = self.random_state

        self._prebucketing_specials = self.specials
        self._bucketing_specials = dict()  # will be determined later.

        # Fit the prebucketing pipeline
        X_prebucketed_ = self.pre_pipeline_.fit_transform(X, y)
        assert isinstance(X_prebucketed_, pd.DataFrame)

        # Calculate the prebucket tables.
        self.prebucket_tables_ = dict()
        for column in X.columns:
            if column in self.pre_pipeline_.features_bucket_mapping_.maps.keys():
                self.prebucket_tables_[column] = build_bucket_table(
                    X, y, column=column, bucket_mapping=self.pre_pipeline_.features_bucket_mapping_.get(column)
                )

        # Find the new bucket numbers of the specials after prebucketing,
        for var, var_specials in self._prebucketing_specials.items():
            bucket_labels = self.pre_pipeline_.features_bucket_mapping_.get(var).labels
            new_specials = _find_remapped_specials(bucket_labels, var_specials)
            if len(new_specials):
                self._bucketing_specials[var] = new_specials

        # Then assign the new specials to all bucketers in the bucketing pipeline
        for step in self.pipeline_.steps:
            if type(step) != tuple:
                step.specials = self._bucketing_specials
            else:
                step[1].specials = self._bucketing_specials

        # Fit the bucketing pipeline
        # And save the bucket mapping
        self.pipeline_.fit(X_prebucketed_, y)

        # Make sure all columns that are bucketed have also been pre-bucketed.
        not_prebucketed = []
        for col in self.pipeline_.features_bucket_mapping_.columns:
            if self.pipeline_.features_bucket_mapping_.get(col).type == "numerical":
                if col not in self.pre_pipeline_.features_bucket_mapping_.columns:
                    not_prebucketed.append(col)
        if len(not_prebucketed):
            msg = "These numerical columns are bucketed but have not been pre-bucketed: "
            msg += f"{', '.join(not_prebucketed)}.\n"
            msg += "Consider adding a numerical bucketer to the prebucketing pipeline,"
            msg += "for example AsIsNumericalBucketer or DecisionTreeBucketer."
            raise NotPreBucketedError(msg)

        # Make sure all columns that have been pre-bucketed also have been bucketed
        not_bucketed = []
        for col in self.pre_pipeline_.features_bucket_mapping_.columns:
            if self.pre_pipeline_.features_bucket_mapping_.get(col).type == "numerical":
                if col not in self.pipeline_.features_bucket_mapping_.columns:
                    not_bucketed.append(col)
        if len(not_bucketed):
            msg = "These numerical columns are prebucketed but have not been bucketed: "
            msg += f"{', '.join(not_bucketed)}.\n"
            msg += "Consider updating the bucketing pipeline."
            raise NotBucketedError(msg)

        # calculate the bucket tables.
        self.bucket_tables_ = dict()
        for column in X.columns:
            if column in self.pipeline_.features_bucket_mapping_.maps.keys():
                self.bucket_tables_[column] = build_bucket_table(
                    X_prebucketed_,
                    y,
                    column=column,
                    bucket_mapping=self.pipeline_.features_bucket_mapping_.get(column),
                )

        # Calculate the summary
        self._generate_summary(X, y)

        return self

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
        # This prevents a user losing manually defined boundaries
        # when re-running .fit_interactive()
        if not is_fitted(self):
            self.fit(X, y)

        self.app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        add_bucketing_process_layout(self)
        add_bucketing_process_callbacks(self, X, y)
        self.app.run_server(mode=mode, **server_kwargs)

    def transform(self, X):
        """
        Transform `X` through the prebucketing and bucketing pipelines.
        """
        check_is_fitted(self)
        X_prebucketed = self.pre_pipeline_.transform(X)

        new_X = self.pipeline_.transform(X_prebucketed)

        if self.remainder == "drop":
            return new_X[self.variables]
        else:
            return new_X

    def save_yml(self, fout: PathLike) -> None:
        """
        Save the features bucket to a yaml file.

        Args:
            fout: path for output file
        """
        check_is_fitted(self)
        fbm = self.features_bucket_mapping_
        if isinstance(fbm, dict):
            FeaturesBucketMapping(fbm).save_yml(fout)
        else:
            fbm.save_yml(fout)

    @property
    def features_bucket_mapping_(self):
        """
        Returns a `FeaturesBucketMapping` instance.

        In normal bucketers, you can access `.features_bucket_mapping_`
        to retrieve a `FeaturesBucketMapping` instance. This contains
        all the info you need to transform values into their buckets.

        In this class, we basically have a two step bucketing process:
        first prebucketing, and then we bucket the prebuckets.

        In order to still be able to use BucketingProcess as if it were a normal bucketer,
        we'll need to merge both into one.
        """
        check_is_fitted(self)

        return merge_features_bucket_mapping(
            self.pre_pipeline_.features_bucket_mapping_, self.pipeline_.features_bucket_mapping_
        )

    def prebucket_table(self, column: str) -> pd.DataFrame:
        """
        Generates the statistics for the buckets of a particular column.

        An example is seen below:

        pre-bucket | label      | Count | Count (%) | Non-event | Event | Event Rate | WoE   | IV   | bucket
        -----------|------------|-------|-----------|-----------|-------|------------|-------|------|------
        0          | (-inf, 1.0)| 479   | 7.98      | 300       | 179   |  37.37     |  0.73 | 0.05 | 0
        1          | [1.0, 2.0) | 370   | 6.17      | 233       | 137   |  37.03     |  0.71 | 0.04 | 0

        Args:
            column (str): The column we wish to analyse

        Returns:
            df (pd.DataFrame): A pandas dataframe of the format above
        """  # noqa
        check_is_fitted(self)
        if column not in self.prebucket_tables_.keys():
            raise ValueError(f"column '{column}' was not part of the pre-bucketing process")

        table = self.prebucket_tables_.get(column)
        table = table.rename(columns={"bucket_id": "pre-bucket"})

        # Find bucket for each pre-bucket
        bucket_mapping = self.pipeline_.features_bucket_mapping_.get(column)
        table["bucket"] = bucket_mapping.transform(table["pre-bucket"])

        # Find out missing bucket
        if -1 in table["pre-bucket"].values:
            table.loc[table["pre-bucket"] == -1, "bucket"] = bucket_mapping.transform([np.nan])[0]

        # Find out the 'other' bucket
        if bucket_mapping.type == "categorical" and -2 in table["pre-bucket"].values:
            something_random = "84a088e251d2fa058f37145222e536dc"
            table.loc[table["pre-bucket"] == -2, "bucket"] = bucket_mapping.transform([something_random])[0]

        return table

    def _more_tags(self):
        """
        Estimator tags are annotations of estimators that allow programmatic inspection of their capabilities.

        See https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """  # noqa
        return {"binary_only": True}


def _find_remapped_specials(bucket_labels: Dict, var_specials: Dict) -> Dict:
    """
    Remaps the specials after the prebucketing process.

    Basically, every bucketer in the bucketing pipeline will now need to
    use the prebucketing bucket as a different special value,
    because prebucketing put the specials into a bucket.

    Args:
        bucket_labels (dict): The label for each unique bucket of a variable
        var_specials (dict): The specials for a variable, if any.
    """
    if bucket_labels is None or var_specials is None:
        return {}

    new_specials = {}
    for label in var_specials.keys():
        for bucket, bucket_label in bucket_labels.items():
            if bucket_label == f"Special: {label}":
                new_specials[label] = [bucket]

    return new_specials
