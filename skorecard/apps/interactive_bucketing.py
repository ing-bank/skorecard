import copy
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from skorecard.utils.exceptions import NotInstalledError
from skorecard.pipeline import find_bucketing_step, get_features_bucket_mapping
from skorecard.apps.app_layout import add_layout
from skorecard.apps.app_callbacks import add_callbacks


# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")


class BucketTweakerApp(object):
    """Tweak bucketing in a sklearn pipeline manually using a Dash web app.

    Example:

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import BucketingProcess
    from skorecard.apps import BucketTweakerApp
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    df = datasets.load_uci_credit_card(as_frame=True)
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        )
    )

    pipeline = make_pipeline(
        bucketing_process,
        OneHotEncoder(),
        LogisticRegression()
    )

    pipeline.fit(X, y)
    tweaker = BucketTweakerApp(pipeline, X, y)
    # tweaker.run_server()
    # tweaker.stop_server()
    tweaker.pipeline # or tweaker.get_pipeline()
    ```
    """

    def __init__(self, pipeline, X, y):
        """Setup for being able to run the dash app.

        Args:
            pipeline (Pipeline): fitted sklearn pipeline object
            X (pd.DataFrame): input dataframe
            y (np.array): target array
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"
        check_is_fitted(pipeline.steps[-1][1])

        # Save X so we can calculate the AUC and other metrics
        # Save y so we can use it for calculating stats like WoE in tables.
        self.X = X
        self.y = y
        # Make sure we don't change instance of input pipeline
        self.pipeline = copy.deepcopy(pipeline)

        index_bucketing_process = find_bucketing_step(self.pipeline, identifier="bucketingprocess")
        self.bucketingprocess = self.pipeline.steps[index_bucketing_process][1]

        # Extract the features bucket mapping information
        self.original_prebucket_feature_mapping = copy.deepcopy(
            get_features_bucket_mapping(self.bucketingprocess.prebucketing_pipeline)
        )
        self.original_bucket_feature_mapping = copy.deepcopy(
            get_features_bucket_mapping(self.bucketingprocess.bucketing_pipeline)
        )

        # Save prebucketed features
        self.X_prebucketed = self.bucketingprocess.prebucketing_pipeline.transform(X)
        # # Prebucketed features should have at most 100 unique values.
        # # otherwise app prebinning table is too big.
        for feature in self.X_prebucketed.columns:
            if len(self.X_prebucketed[feature].unique()) > 100:
                raise AssertionError(f"{feature} has >100 values. Did you apply pre-bucketing?")

        # # Initialize the Dash app, with layout and callbacks
        self.app = JupyterDash(__name__)
        add_layout(self)
        add_callbacks(self)

    def run_server(self, *args, **kwargs):
        """Start a dash server.

        Passes arguments to app.run_server().

        Note we are using a [jupyterdash](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e) app,
        which supports 3 different modes:

        - 'external' (default): Start dash server and print URL
        - 'inline': Start dash app inside an Iframe in the jupyter notebook
        - 'jupyterlab': Start dash app as a new tab inside jupyterlab

        Use like `run_server(mode='inline')`
        """
        return self.app.run_server(*args, **kwargs)

    def stop_server(self):
        """Stop a running app server.

        This is handy when you want to stop a server running in a notebook.

        [More info](https://community.plotly.com/t/how-to-shutdown-a-jupyterdash-app-in-external-mode/41292/3)
        """
        self.app._terminate_server_for_port("localhost", 8050)

    def get_pipeline(self):
        """Returns pipeline object."""
        return self.pipeline


# This section is here to help debug the Dash app
# This custom code start the underlying flask server from dash directly
# allowing better debugging in IDE's f.e. using breakpoint()
# Example:
# python -m ipdb -c continue manual_bucketer_app.py
if __name__ == "__main__":

    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import BucketingProcess
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    df = datasets.load_uci_credit_card(as_frame=True)
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(
        specials={"LIMIT_BAL": {"=400000.0": [400000.0]}},
        prebucketing_pipeline=make_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        ),
        bucketing_pipeline=make_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        ),
    )

    pipeline = make_pipeline(bucketing_process, OneHotEncoder(), LogisticRegression())

    pipeline.fit(X, y)
    tweaker = BucketTweakerApp(pipeline, X, y)
    # tweaker.run_server()

    application = tweaker.app.server
    application.run(debug=True)
