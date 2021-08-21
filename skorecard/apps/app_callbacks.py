import ast
import json
import pandas as pd

from skorecard.reporting import build_bucket_table
from skorecard.apps.app_utils import determine_boundaries, is_increasing, is_sequential
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


def add_bucketing_callbacks(self, X, y):
    """
    Adds callbacks to the interactive bucketing app.

    Meant for normal bucketers, not two step BucketingProcess.
    """
    app = self.app
    add_common_callbacks(self)

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
            # We also allow for treating numerical as categoricals
            # if key is a string, we'll need to quote them
            if isinstance(list(input_map.keys())[0], str):
                str_repr = ",\n\t".join([f"'{k}': {v}" for k, v in input_map.items()])
            else:
                str_repr = ",\n\t".join([f"{k}: {v}" for k, v in input_map.items()])
            str_repr = f"{{\n\t{str_repr}\n}}"
        else:
            str_repr = str(input_map)
        return [str_repr]

    @app.callback(
        [Output("input_map_helptext", "children")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_input_map_feedback(col):
        col_type = self.features_bucket_mapping_.get(col).type
        right = self.features_bucket_mapping_.get(col).right
        if col_type == "categorical":
            msg = "Edit the prebucket mapping dictionary, e.g. {'value' : 'pre-bucket'}"
        if col_type == "numerical" and right:
            msg = "Edit the prebucket mapping boundaries. "
            msg += "Values up to and including the boundary are put into a bucket (right=True)"
        if col_type == "numerical" and not right:
            msg = "Edit the prebucket mapping boundaries. "
            msg += "Values up to but not including the boundary are put into a bucket (right=False)"
        return [msg]

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
    def get_bucket_table(input_map, col):
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
                # re-sort on value, key
                input_map = dict(sorted(input_map.items(), key=lambda x: (x[1], x[0])))
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
        # Note we passed X, y to add_bucketing_callbacks() so they are available here.
        # make sure to re-generate the summary table
        self._update_column_fit(
            X=X, y=y, feature=col, special=special, splits=input_map, right=right, generate_summary=True
        )

        # Retrieve the new bucket tables and plots
        table = self.bucket_table(col)
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        fig = self.plot_bucket(col)
        # remove title from plot
        fig.update_layout(title="")
        return table.to_dict("records"), fig, False, no_update


def add_bucketing_process_callbacks(self, X, y):
    """
    Adds callbacks to the interactive bucketing app.

    Meant two step BucketingProcess.
    """
    app = self.app
    add_common_callbacks(self)

    @app.callback(
        [Output("bucketingprocess-helptext", "style")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_sidebar_helptext(col):
        return [{"display": "block"}]

    @app.callback(
        [Output("input_map", "value")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_input_map(col):
        """Update bucketer map."""
        input_map = self.pre_pipeline.features_bucket_mapping_.get(col).map
        col_type = self.pre_pipeline.features_bucket_mapping_.get(col).type

        if col_type == "categorical":
            # We also allow for treating numerical as categoricals
            # if key is a string, we'll need to quote them
            if isinstance(list(input_map.keys())[0], str):
                str_repr = ",\n\t".join([f"'{k}': {v}" for k, v in input_map.items()])
            else:
                str_repr = ",\n\t".join([f"{k}: {v}" for k, v in input_map.items()])
            str_repr = f"{{\n\t{str_repr}\n}}"
        else:
            str_repr = str(input_map)
        return [str_repr]

    @app.callback(
        [Output("input_map_helptext", "children")],
        [
            Input("input_column", "value"),
        ],
    )
    def update_input_map_feedback(col):
        col_type = self.pre_pipeline.features_bucket_mapping_.get(col).type
        right = self.pre_pipeline.features_bucket_mapping_.get(col).right
        if col_type == "categorical":
            msg = "Edit the prebucket mapping dictionary, e.g. {'value' : 'pre-bucket'}"
        if col_type == "numerical" and right:
            msg = "Edit the prebucket mapping boundaries. "
            msg += "Values up to and including the boundary are put into a bucket (right=True)"
        if col_type == "numerical" and not right:
            msg = "Edit the prebucket mapping boundaries. "
            msg += "Values up to but not including the boundary are put into a bucket (right=False)"
        return [msg]

    @app.callback(
        [
            Output("pre_bucket_table", "data"),
            Output("input_map", "invalid"),
            Output("input_map_feedback", "children"),
        ],
        [Input("input_map", "value")],
        [State("input_column", "value")],
    )
    def get_prebucket_table(input_map, col):
        """Loads the table and the figure, when the input_map changes."""
        col_type = self.pre_pipeline.features_bucket_mapping_.get(col).type

        # Load the object from text input into python object
        if col_type == "numerical":
            try:
                input_map = json.loads(input_map)
                assert len(input_map) > 0
            except Exception:
                msg = "Make sure the input is properly formatted as a list"
                return no_update, True, [msg]
            # validate input
            if not is_increasing(input_map):
                return no_update, True, ["Make sure the list values are in increasing order"]
        else:
            try:
                # note using ast.literal_eval is not safe
                # for use when you don't trust the user input
                # in this case, it's a local user using his/her own kernel
                input_map = ast.literal_eval(input_map)
                # re-sort on value, key
                input_map = dict(sorted(input_map.items(), key=lambda x: (x[1], x[0])))
            except Exception:
                msg = "Make sure the input is properly formatted as a dictionary"
                return no_update, True, [msg]
            # validate input
            if not min(input_map.values()) == 0:
                msg = "Dictionary values (buckets) must start at 0"
                return no_update, True, [msg]
            if not is_sequential(list(input_map.values())):
                msg = "Dictionary values (buckets) must be sequentially increasing with steps of 1"
                return no_update, True, [msg]

        # Update the fit for this specific column
        # Note we passed X, y to add_bucketing_process_callbacks() so they are available here.
        # make sure to re-generate the summary table
        for step in self.pre_pipeline.steps:
            if col in step[1].variables:
                step[1]._update_column_fit(
                    X=X,
                    y=y,
                    feature=col,
                    special=self._prebucketing_specials.get(col, {}),
                    splits=input_map,
                    right=self.pre_pipeline.features_bucket_mapping_.get(col).right,
                    generate_summary=True,
                )

        self.prebucket_tables_[col] = build_bucket_table(
            X, y, column=col, bucket_mapping=self.pre_pipeline.features_bucket_mapping_.get(col)
        )
        # Re-calculate the BucketingProcess summary
        self._generate_summary(X, y)

        # Retrieve the new bucket tables and plots
        table = self.prebucket_table(col)
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        return table.to_dict("records"), False, no_update

    @app.callback(
        [
            Output("bucket_table", "data"),
            Output("graph-prebucket", "figure"),
            Output("graph-bucket", "figure"),
            Output("bucket-error-msg", "children"),
            Output("bucket-error-msg", "style"),
        ],
        [Input("pre_bucket_table", "data")],
        [State("input_column", "value")],
    )
    def get_bucket_table(prebucket_table, col):
        # Get the input from the prebucket table
        new_buckets = pd.DataFrame()
        new_buckets["pre_buckets"] = [row.get("pre-bucket") for row in prebucket_table]
        input_buckets = [int(row.get("bucket")) for row in prebucket_table]
        new_buckets["buckets"] = input_buckets

        # Input validation
        bucket_mapping = self.pipeline.features_bucket_mapping_.get(col)
        if not is_sequential(sorted(input_buckets)):
            msg = "Buckets must start at 0 and increase by 1"
            return no_update, no_update, no_update, [msg], {"display": "block"}

        # Determine the boundaries from the buckets set in the prebucket table
        boundaries = determine_boundaries(new_buckets, bucket_mapping)

        # Update the fit for this specific column
        # Note we passed X, y to add_bucketing_process_callbacks() so they are available here.
        # make sure to re-generate the summary table
        for step in self.pipeline.steps:
            if col in step[1].variables:
                step[1]._update_column_fit(
                    X=X,
                    y=y,
                    feature=col,
                    special=bucket_mapping.specials,
                    splits=boundaries,
                    right=bucket_mapping.right,
                    generate_summary=True,
                )

        # update the BucketingProcess bucket tables.
        X_prebucketed = self.pre_pipeline.transform(X)
        self.bucket_tables_[col] = build_bucket_table(
            X_prebucketed,
            y,
            column=col,
            bucket_mapping=self.pipeline.features_bucket_mapping_.get(col),
        )

        # Re-calculate the BucketingProcess summary
        self._generate_summary(X, y)

        # Get the updated tables and figures
        table = self.bucket_table(col)
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        prebucket_fig = self.plot_prebucket(col)
        prebucket_fig.update_layout(title="")
        fig = self.plot_bucket(col)
        fig.update_layout(title="")

        return table.to_dict("records"), prebucket_fig, fig, no_update, {"display": "none"}


def add_common_callbacks(self):
    """
    Add dash callbacks.

    Common callbacks for the normal bucketer app and the BucketingProcess app.
    """
    app = self.app

    @app.callback(
        [
            Output("column_title", "children"),
            Output("column_type", "children"),
        ],
        [
            Input("input_column", "value"),
        ],
    )
    def update_column_title(col):
        """Update the content title."""
        col_type = self.features_bucket_mapping_.get(col).type
        return [f"Feature '{col}'"], [col_type]

    @app.callback(
        [Output("code_export", "content")],
        [Input("input_map", "value")],
    )
    def update_code_export(input_map):
        return [f"UserInputBucketer({self.features_bucket_mapping_.as_dict()})"]
