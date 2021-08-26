import logging

import pandas as pd
import numpy as np
from typing import Dict, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from skorecard.features_bucket_mapping import FeaturesBucketMapping
from skorecard.reporting.plotting import PlotBucketMethod
from skorecard.reporting.report import BucketTableMethod, SummaryMethod
from skorecard.utils import BucketingPipelineError, NotBucketObjectError, NotInstalledError
from skorecard.utils.validation import is_fitted

# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")


from skorecard.apps.app_layout import add_basic_layout
from skorecard.apps.app_callbacks import add_bucketing_callbacks


class KeepPandas(BaseEstimator, TransformerMixin):
    """
    Wrapper to keep column names of pandas dataframes in a `scikit-learn` transformer.

    Any scikit-learn transformer wrapped in KeepPandas will return a `pd.DataFrame` on `.transform()`.

    !!! warning
        You should only use `KeepPandas()` when you know for sure `scikit-learn`
        did not change the order of your columns.

    Example:

    ```python
    from skorecard.pipeline import KeepPandas
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        KeepPandas(StandardScaler()),
        EqualWidthBucketer(n_bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
    )
    bucket_pipeline.fit_transform(X, y)
    ```
    """

    def __init__(self, transformer):
        """Initialize."""
        self.transformer = transformer

        # Warn if there is a chance order of columns are changed
        if isinstance(transformer, Pipeline):
            for step in _get_all_steps(transformer):
                self._check_for_column_transformer(step)
        else:
            self._check_for_column_transformer(transformer)

    def __repr__(self):
        """String representation."""
        return self.transformer.__repr__()

    @staticmethod
    def _check_for_column_transformer(obj):
        msg = "sklearn.compose.ColumnTransformer can change the order of columns"
        msg += ", be very careful when using with KeepPandas()"
        if type(obj).__name__ == "ColumnTransformer":
            logging.warning(msg)

    def fit(self, X, y=None, *args, **kwargs):
        """Fit estimator."""
        assert isinstance(X, pd.DataFrame)
        self.columns_ = list(X.columns)
        self.transformer.fit(X, y, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        """Transform X."""
        check_is_fitted(self)
        new_X = self.transformer.transform(X, *args, **kwargs)
        return pd.DataFrame(new_X, columns=self.columns_)

    def get_feature_names(self):
        """Return estimator feature names."""
        check_is_fitted(self)
        return self.columns_


def find_bucketing_step(pipeline: Pipeline, identifier: str = "bucketingprocess"):
    """
    Finds a specific step in a sklearn Pipeline that has a 'name' attribute equalling 'identifier'.

    This is usefull to extract certain steps from a pipeline, f.e. a BucketingProcess.

    Args:
        pipeline (sklearn.pipeline.Pipeline): sklearn pipeline
        identifier (str): the attribute used to find the pipeline step

    Returns:
        index (int): position of bucketing step in pipeline.steps
    """
    # Find the bucketing pipeline step
    bucket_pipes = [s for s in pipeline.steps if getattr(s[1], "name", "") == identifier]

    # Raise error if missing
    if len(bucket_pipes) == 0:
        msg = """
        Did not find a bucketing pipeline step. Identity the bucketing pipeline step
        using skorecard.pipeline.make_bucketing_pipeline or skorecard.pipeline.make_prebucketing_pipeline.

        Note that the pipeline should always have a skorecard.pipeline.make_prebucketing_pipeline defined.
        If you do not need prebucketing simply leave it empty.

        Example:
        
        ```python
        from sklearn.pipeline import make_pipeline
        from skorecard.pipeline import make_bucketing_pipeline, make_prebucketing_pipeline

        pipeline = make_pipeline(
            make_prebucketing_pipeline(),
            make_bucketing_pipeline(
                    OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
                    OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
            )
        )
        ```
        """
        raise AssertionError(msg)

    if len(bucket_pipes) > 1:
        msg = """
        You need to identity only the bucketing step,
        using skorecard.pipeline.make_bucketing_pipeline and skorecard.pipeline.make_prebucketing_pipeline only once.
        
        Example:
        
        ```python
        from skorecard.pipeline import make_bucketing_pipeline
        bucket_pipeline = make_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        )
        ```
        """
        raise AssertionError(msg)

    index_bucket_pipeline = pipeline.steps.index(bucket_pipes[0])
    return index_bucket_pipeline


def get_features_bucket_mapping(pipe: Pipeline) -> FeaturesBucketMapping:
    """Get feature bucket mapping from a sklearn pipeline object.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer
    from skorecard.pipeline import get_features_bucket_mapping

    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        EqualWidthBucketer(n_bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
        OrdinalCategoricalBucketer(variables=['EDUCATION', 'MARRIAGE'])
    )

    pipe = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])

    pipe.fit(X, y)
    features_bucket_mapping = get_features_bucket_mapping(pipe)
    ```

    Args:
        pipe (Pipeline): fitted scikitlearn pipeline with bucketing transformers

    Returns:
        FeaturesBucketMapping: skorecard class with the bucket info
    """
    assert isinstance(pipe, BaseEstimator)

    features_bucket_mapping = {}
    for step in _get_all_steps(pipe):
        check_is_fitted(step)
        if hasattr(step, "features_bucket_mapping_"):
            features_bucket_mapping.update(step.features_bucket_mapping_.as_dict())

    assert (
        len(features_bucket_mapping) > 0
    ), "pipeline does not have any fitted skorecard bucketer. Update the pipeline or fit(X,y) first"
    return FeaturesBucketMapping(features_bucket_mapping)


def _get_all_steps(pipeline: Pipeline) -> List:
    """
    Returns a list of steps in a sklearn pipeline.

    Args:
        pipeline (Pipeline): A scikitlearn pipeline.
    """
    steps = []
    for step in pipeline.steps:
        if type(step) == tuple:
            step = step[1]
        if hasattr(step, "steps"):
            steps += _get_all_steps(step)
        else:
            steps.append(step)
    return steps


class SkorecardPipeline(Pipeline, PlotBucketMethod, BucketTableMethod, SummaryMethod):
    """
    A sklearn Pipeline with several attribute and methods added.

    This Pipeline of bucketers behaves more like a bucketer and adds:

    - `.summary()`: See which columns are bucketed
    - `.plot_bucket()`: Plot buckets of a column
    - `.bucket_table()`: Table with buckets of a column
    - `.save_to_yaml()`: Save information necessary for bucketing to a YAML file
    - `.features_bucket_mapping_`: Access bucketing information
    - `.fit_interactive()`: Edit fitted buckets interactively in a dash app

    ```python
    from skorecard.pipeline.pipeline import SkorecardPipeline
    from skorecard.bucketers import DecisionTreeBucketer, OrdinalCategoricalBucketer
    from skorecard import datasets

    pipe = SkorecardPipeline([
        ('decisiontreebucketer', DecisionTreeBucketer(variables = ["LIMIT_BAL", "BILL_AMT1"],max_n_bins=5)),
        ('ordinalcategoricalbucketer', OrdinalCategoricalBucketer(variables = ["EDUCATION", "MARRIAGE"], tol =0.05)),
    ])

    df = datasets.load_uci_credit_card(as_frame=True)
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"].values

    pipe.fit(X, y)
    pipe.bucket_table('LIMIT_BAL')
    ```
    """

    def __init__(self, steps, *, memory=None, verbose=False):
        """
        Wraps sklearn Pipeline.
        """
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        self._check_pipeline_all_bucketers(self)
        self._check_pipeline_duplicated_columns(self)

    @property
    def features_bucket_mapping_(self):
        """
        Retrieve features bucket mapping.
        """
        check_is_fitted(self.steps[-1][1])
        return get_features_bucket_mapping(Pipeline(self.steps))

    @property
    def bucket_tables_(self):
        """
        Retrieve bucket tables.

        Used by .bucket_table()
        """
        check_is_fitted(self.steps[-1][1])
        bucket_tables = dict()
        for step in self.steps:
            bucket_tables.update(step[1].bucket_tables_)
        return bucket_tables

    @property
    def summary_dict_(self) -> Dict:
        """
        Retrieve summary_dicts and combine.

        Used by .summary()
        """
        summary_dict = {}
        for step in self.steps:
            summary_dict.update(step[1].summary_dict_)
        return summary_dict

    def save_yml(self, fout):
        """
        Save the features bucket to a yaml file.

        Args:
            fout: file output
        """
        check_is_fitted(self.steps[-1][1])
        self.features_bucket_mapping_.save_yml(fout)

    @staticmethod
    def _check_pipeline_duplicated_columns(pipeline: Pipeline) -> None:
        """
        Check that the pipeline has no duplicated columns.

        This check only works on fitted pipelines!
        """
        assert isinstance(pipeline, Pipeline)

        bucketers_vars = []
        for step in _get_all_steps(pipeline):
            if hasattr(step, "variables"):
                bucketers_vars += step.variables

        if any([x is None for x in bucketers_vars]):
            if not all([x is None for x in bucketers_vars]):
                raise BucketingPipelineError(
                    "One of the bucketers applies to all variables, which means a feature will be bucketed twice."
                )

        if len(set(bucketers_vars)) != len(bucketers_vars):
            values, counts = np.unique(bucketers_vars, return_counts=True)
            duplicates = set(values[counts > 1])

            raise BucketingPipelineError(f"The features {duplicates} appear in multiple bucketers.")

    @staticmethod
    def _check_pipeline_all_bucketers(pipeline: Pipeline) -> None:
        """
        Ensure all specified bucketing steps are skorecard bucketers.

        Args:
            pipeline: scikit-learn pipeline.
        """
        assert isinstance(pipeline, Pipeline)

        for step in _get_all_steps(pipeline):
            if all(x not in str(type(step)) for x in ["bucketing_process", "skorecard.bucketers"]):
                msg = "All bucketing steps must be skorecard bucketers."
                msg += f"Remove {step} from the pipeline."
                raise NotBucketObjectError(msg)

    @property
    def variables(self):
        """
        Helper function to show which features are in scope of this pipeline.
        """
        return self.features_bucket_mapping_.columns

    def fit_interactive(self, X, y=None, mode="external"):
        """
        Fit a bucketer and then interactively edit the fit using a dash app.

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

        import dash_bootstrap_components as dbc

        self.app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        add_basic_layout(self)
        add_bucketing_callbacks(self, X, y)
        self.app.run_server(mode=mode)

    def _update_column_fit(self, X, y, feature, special, splits, right, generate_summary=False):
        """
        Extract out part of the fit for a column.

        Useful when we want to interactively update the fit.
        """
        for step in self.steps:
            if feature in step[1].variables:
                step[1]._update_column_fit(
                    X=X,
                    y=y,
                    feature=feature,
                    special=special,
                    splits=splits,
                    right=right,
                    generate_summary=generate_summary,
                )


def to_skorecard_pipeline(pipeline: Pipeline) -> SkorecardPipeline:
    """
    Transform a scikit-learn Pipeline to a SkorecardPipeline.

    A SkorecardPipeline is a normal scikit-learn pipeline with some extra methods and attributes.

    Example:

    ```python
    from skorecard.pipeline.pipeline import SkorecardPipeline, to_skorecard_pipeline
    from skorecard.bucketers import DecisionTreeBucketer, OrdinalCategoricalBucketer
    from skorecard import datasets

    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        DecisionTreeBucketer(variables = ["LIMIT_BAL", "BILL_AMT1"],max_n_bins=5),
        OrdinalCategoricalBucketer(variables = ["EDUCATION", "MARRIAGE"], tol =0.05)
    )
    sk_pipe = to_skorecard_pipeline(pipe)

    df = datasets.load_uci_credit_card(as_frame=True)

    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]
    X = df[features]
    y = df["default"].values
    ```

    Args:
        pipeline (Pipeline): `scikit-learn` pipeline instance.

    Returns:
        pipeline (skorecard.pipeline.SkorecardPipeline): modified pipeline instance.
    """
    assert isinstance(pipeline, Pipeline)
    if isinstance(pipeline, SkorecardPipeline):
        return pipeline
    else:
        return SkorecardPipeline(steps=pipeline.steps, memory=pipeline.memory, verbose=pipeline.verbose)
