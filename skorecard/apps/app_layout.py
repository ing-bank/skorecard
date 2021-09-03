from skorecard.utils.exceptions import NotInstalledError
from skorecard.apps.app_utils import perc_data_bars, colorize_cell

# Dash + dependencies
try:
    import dash_core_components as dcc
    import dash_html_components as html
    import dash_table
except ModuleNotFoundError:
    dcc = NotInstalledError("dash_core_components", "dashboard")
    html = NotInstalledError("dash_html_components", "dashboard")
    Input = NotInstalledError("dash", "dashboard")
    Output = NotInstalledError("dash", "dashboard")
    State = NotInstalledError("dash", "dashboard")
    dash_table = NotInstalledError("dash_table", "dashboard")

# Dash Bootstrap
try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")


# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    "padding": "20px 10px",
    "background-color": "#f8f9fa",
}
# the style arguments for the main content page.
CONTENT_STYLE = {"margin-left": "25%", "margin-right": "5%", "top": 0, "padding": "20px 10px"}
TEXT_STYLE = {"textAlign": "center", "color": "#191970"}
CARD_TEXT_STYLE = {"textAlign": "center", "color": "#0074D9"}


def add_basic_layout(self):
    """
    Adds a basic layout to self.app.
    """
    column_options = [{"label": o, "value": o} for o in self.features_bucket_mapping_.columns]

    sidebar = get_sidebar(self, column_options)

    content = html.Div(
        children=[
            dbc.Row(
                [
                    html.H3(
                        [
                            html.Span("Current column", id="column_title"),
                            dbc.Badge("tbd", id="column_type", className="ml-1"),
                        ],
                        style=TEXT_STYLE,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    html.H4("Bucket plot", style=TEXT_STYLE),
                ]
            ),
            dbc.Row([dcc.Graph(id="graph-bucket", style={"width": "100%"})]),
            dbc.Row(
                [
                    html.H4("Bucket table", style=TEXT_STYLE),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dash_table.DataTable(
                        id="bucket_table",
                        style_table={
                            "width": "100%",
                            "minWidth": "100%",
                        },
                        style_data={"whiteSpace": "normal", "height": "auto"},
                        style_cell={
                            "height": "auto",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "minWidth": "80px",
                            "textAlign": "left",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "range"}, "width": "180px"},
                        ],
                        page_size=20,
                        columns=[
                            {"name": "label", "id": "label", "editable": False},
                            {"name": "bucket", "id": "bucket", "editable": False},
                            {"name": "Count", "id": "Count", "editable": False},
                            {"name": "Count (%)", "id": "Count (%)", "editable": False},
                            {"name": "Non-event", "id": "Non-event", "editable": False},
                            {"name": "Event", "id": "Event", "editable": False},
                            {"name": "Event Rate", "id": "Event Rate", "editable": False},
                            {"name": "WoE", "id": "WoE", "editable": False},
                            {"name": "IV", "id": "IV", "editable": False},
                        ],
                        style_data_conditional=perc_data_bars("Count (%)")
                        + perc_data_bars("Event Rate")
                        + [
                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                            {
                                "if": {"column_editable": True},
                                "backgroundColor": "rgb(46,139,87)",
                                "color": "white",
                                "border": "1px solid black",
                            },
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                "border": "1px solid rgb(0, 116, 217)",
                            },
                        ]
                        + colorize_cell("bucket"),
                        style_header_conditional=[
                            {
                                "if": {"column_editable": True},
                                "border": "1px solid black",
                            }
                        ],
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                        },
                        editable=False,
                        fill_width=True,
                    ),
                ],
                style={"width": "100%"},
            ),
        ],
        style=CONTENT_STYLE,
    )

    self.app.layout = html.Div([sidebar, content])


def add_bucketing_process_layout(self):
    """
    Adds a layout to self.app.
    """
    column_options = [{"label": o, "value": o} for o in self.features_bucket_mapping_.columns]
    sidebar = get_sidebar(self, column_options)

    content = html.Div(
        children=[
            dbc.Row(
                [
                    html.H3(
                        [
                            html.Span("Current column", id="column_title"),
                            dbc.Badge("tbd", id="column_type", className="ml-1"),
                        ],
                        style=TEXT_STYLE,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    html.H4("Pre-bucket plot", style=TEXT_STYLE),
                ]
            ),
            dbc.Row([dcc.Graph(id="graph-prebucket", style={"width": "100%"})]),
            dbc.Row(
                [
                    html.H4("Pre-bucket table", style=TEXT_STYLE),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Badge(
                        "There is something wrong with your bucketing info",
                        color="danger",
                        id="bucket-error-msg",
                        style={"display": "none"},
                    ),
                    dash_table.DataTable(
                        id="pre_bucket_table",
                        style_table={
                            "width": "100%",
                            "minWidth": "100%",
                        },
                        style_data={"whiteSpace": "normal", "height": "auto"},
                        style_cell={
                            "height": "auto",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "minWidth": "80px",
                            "textAlign": "left",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "range"}, "width": "180px"},
                        ],
                        page_size=20,
                        columns=[
                            {"name": "label", "id": "label", "editable": False},
                            {"name": "pre-bucket", "id": "pre-bucket", "editable": False},
                            {"name": "Count", "id": "Count", "editable": False},
                            {"name": "Count (%)", "id": "Count (%)", "editable": False},
                            {"name": "Non-event", "id": "Non-event", "editable": False},
                            {"name": "Event", "id": "Event", "editable": False},
                            {"name": "Event Rate", "id": "Event Rate", "editable": False},
                            {"name": "WoE", "id": "WoE", "editable": False},
                            {"name": "IV", "id": "IV", "editable": False},
                            {"name": "bucket", "id": "bucket", "editable": True},
                        ],
                        style_data_conditional=perc_data_bars("Count (%)")
                        + perc_data_bars("Event Rate")
                        + [
                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                            {
                                "if": {"column_editable": True},
                                "backgroundColor": "rgb(46,139,87)",
                                "color": "white",
                                "border": "1px solid black",
                            },
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                "border": "1px solid rgb(0, 116, 217)",
                            },
                        ]
                        + colorize_cell("bucket"),
                        style_header_conditional=[
                            {
                                "if": {"column_editable": True},
                                "border": "1px solid black",
                            }
                        ],
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                        },
                        editable=False,
                        fill_width=True,
                    ),
                ],
                style={"width": "100%"},
            ),
            html.Br(),
            dbc.Row(
                [
                    html.H4("Bucket table", style=TEXT_STYLE),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dash_table.DataTable(
                        id="bucket_table",
                        style_table={
                            "width": "100%",
                            "minWidth": "100%",
                        },
                        style_data={"whiteSpace": "normal", "height": "auto"},
                        style_cell={
                            "height": "auto",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "minWidth": "80px",
                            "textAlign": "left",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "range"}, "width": "180px"},
                        ],
                        page_size=20,
                        columns=[
                            {"name": "bucket", "id": "bucket", "editable": False},
                            {"name": "label", "id": "label", "editable": False},
                            {"name": "Count", "id": "Count", "editable": False},
                            {"name": "Count (%)", "id": "Count (%)", "editable": False},
                            {"name": "Non-event", "id": "Non-event", "editable": False},
                            {"name": "Event", "id": "Event", "editable": False},
                            {"name": "Event Rate", "id": "Event Rate", "editable": False},
                            {"name": "WoE", "id": "WoE", "editable": False},
                            {"name": "IV", "id": "IV", "editable": False},
                        ],
                        style_data_conditional=perc_data_bars("Count (%)")
                        + perc_data_bars("Event Rate")
                        + [
                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                            {
                                "if": {"column_editable": True},
                                "backgroundColor": "rgb(46,139,87)",
                                "color": "white",
                                "border": "1px solid black",
                            },
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                "border": "1px solid rgb(0, 116, 217)",
                            },
                        ]
                        + colorize_cell("bucket"),
                        style_header_conditional=[
                            {
                                "if": {"column_editable": True},
                                "border": "1px solid black",
                            }
                        ],
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                        },
                        editable=False,
                        fill_width=True,
                    ),
                ],
                style={"width": "100%"},
            ),
            html.Br(),
            dbc.Row(
                [
                    html.H4("Bucket plot", style=TEXT_STYLE),
                ]
            ),
            dbc.Row([dcc.Graph(id="graph-bucket", style={"width": "100%"})]),
        ],
        style=CONTENT_STYLE,
    )

    self.app.layout = html.Div([sidebar, content])


def get_sidebar(self, column_options):
    """
    Layout for the sidebar in the dash app.
    """
    return html.Div(
        [
            html.H5(f"{type(self).__name__}", style=TEXT_STYLE),
            html.Hr(),
            dbc.Label("Select feature", html_for="input_column"),
            dcc.Dropdown(
                id="input_column",
                options=column_options,
                value=column_options[0].get("label"),
            ),
            html.Br(),
            dbc.FormGroup(
                [
                    dbc.Label("Map", html_for="input_map"),
                    dbc.Textarea(
                        id="input_map",
                        style={"minHeight": 160},
                    ),
                    dbc.FormText(
                        "You can manually set different boundaries", color="secondary", id="input_map_helptext"
                    ),
                    dbc.FormFeedback("Feedback will be given here", valid=False, id="input_map_feedback"),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Bucket assignment"),
                    dbc.FormText(
                        "You can edit bucket assignment in the highlighted right column of the prebucket table.",
                    ),
                ],
                id="bucketingprocess-helptext",
                style={"display": "none"},
            ),
            html.H5("Export"),
            html.P(
                [
                    "Your updated buckets are saved to the class instance, ",
                    "which means you can safely close this app. See also ",
                    dcc.Link(
                        "working with manually defined buckets",
                        href="https://ing-bank.github.io/skorecard/howto/using_manually_defined_buckets/",
                    ),
                ],
                className="small",
            ),
            html.P(
                [
                    "For convenience, you can also copy this generated code snippet defining a ",
                    dcc.Link(
                        html.Code("UserInputBucketer"),
                        href="https://ing-bank.github.io/skorecard/api/bucketers/UserInputBucketer/",
                    ),
                    dcc.Clipboard(content="hi", title="code snippet", style={"font-size": "120%"}, id="code_export"),
                ],
                className="small",
            ),
        ],
        style=SIDEBAR_STYLE,
    )
