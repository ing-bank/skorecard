import ast
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from skorecard.apps.app_utils import determine_boundaries, get_bucket_colors
from skorecard.utils.exceptions import NotInstalledError

# Dash + dependencies
try:
    from dash.dependencies import Input, Output, State
    from dash import no_update
    import dash_table
except ModuleNotFoundError:
    Input = NotInstalledError("dash", "dashboard")
    Output = NotInstalledError("dash", "dashboard")
    State = NotInstalledError("dash", "dashboard")
    dash_table = NotInstalledError("dash_table", "dashboard")

# Dash Bootstrap
try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")

# plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")

import json


def is_monotonic_increasing(x):
    """
    Helper function to determine if a list is monotonically increasing.
    """
    dx = np.diff(x)
    return np.all(dx >= 0)


def is_increasing(x):
    """
    Helper function to determine if a list is increasing.
    """
    dx = np.diff(x)
    return np.all(dx > 0)


def is_sequential(x):
    """
    Helper function to determine if a list is monotonically increasing with step size 1.
    """
    dx = np.diff(x)
    return np.all(np.isin(dx, [0, 1]))


def add_bucketing_callbacks(self, X, y):
    """
    Adds callbacks to the interactive bucketing app.

    Meant for normal bucketers, not two step BucketingProcess.
    """
    app = self.app

    @app.callback(
        [Output("column_title", "children")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_column_title(title):
        """Update the content title."""
        return [f"Feature '{title}'"]

    @app.callback(
        [
            Output("bucket_table", "data"),
            Output("graph-bucket", "figure"),
            Output("input_map", "invalid"),
            Output("input_map_feedback", "children"),
        ],
        [Input("input_map", "value")],
        [State("input_column", "value")],
    )
    def get_prebucket_table(input_map, col):
        """Loads the table and the figure, when the input_map changes."""
        col_type = self.features_bucket_mapping_.get(col).type

        # Load the object from text input into python object
        if col_type == "numerical":
            try:
                input_map = json.loads(input_map)
                assert len(input_map) > 0
            except Exception:
                msg = "Make sure the input is properly formatted as a list"
                return no_update, no_update, True, [msg]
            # validate input
            if not is_increasing(input_map):
                return no_update, no_update, True, ["Make sure the list values are in increasing order"]
        else:
            try:
                # note using ast.literal_eval is not safe
                # for use when you don't trust the user input
                # in this case, it's a local user using his/her own kernel
                input_map = ast.literal_eval(input_map)
            except Exception:
                msg = "Make sure the input is properly formatted as a dictionary"
                return no_update, no_update, True, [msg]
            # validate input
            if not min(input_map.values()) == 0:
                msg = "Dictionary values (buckets) must start at 0"
                return no_update, no_update, True, [msg]
            if not is_sequential(list(input_map.values())):
                msg = "Dictionary values (buckets) must be sequentially increasing with steps of 1"
                return no_update, no_update, True, [msg]

        # Update the fit for this specific column
        special = self.features_bucket_mapping_.get(col).specials
        right = self.features_bucket_mapping_.get(col).right
        # Note we passed X, y to add_bucketing_callbacks()
        self._update_column_fit(X, y, col, special, input_map, right)
        # make sure to re-generate the summary table
        self._generate_summary(X, y)

        # Retrieve the new bucket tables and plots
        table = self.bucket_table(col)
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        fig = self.plot_bucket(col)
        # remove title from plot
        fig.update_layout(title="")
        return table.to_dict("records"), fig, False, no_update

    @app.callback(
        [Output("input_map", "value")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_input_map(col):
        """Update bucketer map."""
        input_map = self.features_bucket_mapping_.get(col).map
        col_type = self.features_bucket_mapping_.get(col).type

        if col_type == "categorical":
            str_repr = json.dumps(input_map, indent=4)
        else:
            str_repr = str(input_map)
        return [str_repr]

    @app.callback(
        [Output("code_export", "content")],
        [Input("input_map", "value")],
    )
    def update_code_export(input_map):
        return [f"UserInputBucketer({self.features_bucket_mapping_.as_dict()})"]


def add_callbacks(self):
    """
    Single place where all callbacks for the dash app are defined.
    """
    app = self.app

    @app.callback(
        Output("collapse-menu-boundaries", "is_open"),
        [Input("menu-boundaries", "n_clicks")],
        [State("collapse-menu-boundaries", "is_open")],
    )
    def toggle_collapse(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse-menu-save-versions", "is_open"),
        [Input("menu-save-versions", "n_clicks")],
        [State("collapse-menu-save-versions", "is_open")],
    )
    def toggle_collapse2(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse-menu-model-performance", "is_open"),
        [Input("menu-model-performance", "n_clicks")],
        [State("collapse-menu-model-performance", "is_open")],
    )
    def toggle_collapse3(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("is_not_monotonic_badge", "is_open"),
        [Input("bucket_table", "data")],
    )
    def badge_is_monotonic(bucket_table):
        event_rates = [x.get("Event Rate") for x in bucket_table]
        dx = np.diff(event_rates)
        monotonic = np.all(dx <= 0) or np.all(dx >= 0)
        return not monotonic

    @app.callback(
        Output("has_5perc_badge", "is_open"),
        [Input("bucket_table", "data")],
    )
    def badge_is_has_5perc(bucket_table):
        event_perc = [x.get("Count (%)") for x in bucket_table]
        return not all([float(x) >= 5 for x in event_perc])

    @app.callback(
        Output("original_boundaries", "children"),
        [Input("input_column", "value")],
    )
    def update_original_boundaries(col):
        return str(self.original_bucket_feature_mapping.get(col).map)

    @app.callback(
        Output("updated_boundaries", "children"), [Input("bucket_table", "data")], State("input_column", "value")
    )
    def update_updated_boundaries(bucket_table, col):
        return str(self.bucketingprocess._features_bucket_mapping.get(col).map)

    @app.callback(
        Output("input_column", "value"),
        [Input("reset-boundaries-button", "n_clicks")],
        State("input_column", "value"),
    )
    def reset_boundaries(n_clicks, col):
        original_map = self.original_bucket_feature_mapping.get(col).map
        self.bucketingprocess._features_bucket_mapping.get(col).map = original_map
        # update same column to input_colum
        # this will trigger other components to update
        return col

    @app.callback(
        [Output("prebucket_table", "data"), Output("graph-prebucket", "figure")],
        [Input("input_column", "value")],
    )
    def get_prebucket_table(col):
        table = self.bucketingprocess.prebucket_table(col)
        fig = self.bucketingprocess.plot_prebucket(col)
        return table.to_dict("records"), fig

    @app.callback(
        [Output("bucket_table", "data"), Output("pre-bucket-error", "children")],
        [Input("prebucket_table", "data")],
        State("input_column", "value"),
    )
    def get_bucket_table(prebucket_table, col):
        # Determine the boundaries from the buckets set in the prebucket table
        new_buckets = pd.DataFrame()
        new_buckets["pre_buckets"] = [row.get("pre-bucket") for row in prebucket_table]
        new_buckets["buckets"] = [int(row.get("bucket")) for row in prebucket_table]

        features_bucket_mapping = self.bucketingprocess._features_bucket_mapping
        boundaries = determine_boundaries(new_buckets, features_bucket_mapping.get(col))

        # Update the feature_bucket_mapping in the bucketingprocess
        features_bucket_mapping.get(col).map = boundaries
        self.bucketingprocess._set_bucket_mapping(features_bucket_mapping, self.X_prebucketed, self.y)

        # Get the new bucketing table
        table = self.bucketingprocess.bucket_table(col)

        # Explicit error handling
        # if all(new_buckets["buckets"].sort_values().values == new_buckets["buckets"].values):
        #     error = []
        # else:
        #     error = dbc.Alert("The buckets most be in ascending order!", color="danger")
        #     return None, error
        error = []

        return table.to_dict("records"), error

    @app.callback(
        Output("menu-model-performance", "children"),
        [Input("bucket_table", "data")],
    )
    def update_auc(bucket_table):
        yhat = [x[1] for x in self.pipeline.predict_proba(self.X)]
        auc = roc_auc_score(self.y, yhat)
        return f"AUC: {auc:.3f}"

    @app.callback(
        Output("graph-prebucket", "figure"),
        [Input("prebucket_table", "data")],
    )
    def plot_prebucket_bins(col, prebucket_table):

        return self.bucketingprocess.plot_prebucket(col)

    @app.callback(
        Output("graph-bucket", "figure"),
        [Input("bucket_table", "data")],
    )
    def plot_bucket_bins(data):

        bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
        buckets = [int(x.get("bucket")) for x in data]
        bar_colors = [bucket_colors[i] for i in buckets]

        plotdf = pd.DataFrame(
            {
                "bucket": [int(row.get("bucket")) for row in data],
                "counts": [int(row.get("Count")) for row in data],
                "counts %": [float(row.get("Count (%)")) for row in data],
                "Event Rate": [row.get("Event Rate") for row in data],
            }
        )

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Bar(x=plotdf["bucket"], y=plotdf["counts %"], name="counts (%)"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=plotdf["bucket"], y=plotdf["Event Rate"], name="Event Rate", line=dict(color="#454c57")),
            secondary_y=True,
        )
        fig.update_layout(transition_duration=50)
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title="Bucket")
        # Set y-axes titles
        fig.update_yaxes(title_text="counts (%)", secondary_y=False)
        fig.update_yaxes(title_text="event rate (%)", secondary_y=True)
        fig.update_layout(title="Bucketed")
        fig.update_xaxes(type="category")
        fig.update_traces(
            marker=dict(color=bar_colors),
            selector=dict(type="bar"),
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
        )
        return fig
