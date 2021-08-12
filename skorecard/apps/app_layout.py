from skorecard.utils.exceptions import NotInstalledError

# Dash + dependencies
try:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State
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


from skorecard.apps.app_utils import perc_data_bars, colorize_cell


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
    column_options = [{"label": o, "value": o} for o in self.variables]

    sidebar = html.Div(
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
                        "You can manually set different boundaries",
                        color="secondary",
                    ),
                    dbc.FormFeedback("Feedback will be given here", valid=False, id="input_map_feedback"),
                ]
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
                ]
            ),
            html.P(
                [
                    "For convenience, you can also copy this snippet defining a skorecard.bucketers.UserInputBucketer:",
                    dcc.Clipboard(content="hi", title="code snippet", style={"font-size": "120%"}, id="code_export"),
                ]
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    content = html.Div(
        children=[
            dbc.Row(
                [
                    html.H3("Current column", id="column_title", style=TEXT_STYLE),
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
                            # 'overflowY': 'scroll',
                            # 'overflowX': 'scroll',
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
                        # style_as_list_view=True,
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
            )
            # dbc.Row(
            #     children=[
            #         dbc.Col(
            #             html.Div(
            #                 [
            #                 ],
            #                 # style={"padding": "0 1em 0 1em", "width": "100%"},
            #             ),
            #             # style={"margin": "0 1em 0 1em"},
            #             # width=5,
            #         ),
            #     ],
            #     no_gutters=False,
            #     justify="center",
            # ),
        ],
        style=CONTENT_STYLE,
    )

    self.app.layout = html.Div([sidebar, content])


def add_layout(self):
    """
    Returns a dash layout for the bucketing app.
    """
    column_options = [{"label": o, "value": o} for o in self.variables]

    navbar = dbc.NavbarSimple(
        children=[
            dbc.Button("AUC: t.b.d.", color="primary", className="mr-1", id="menu-model-performance"),
            dbc.Button("Boundaries", color="primary", className="mr-1", id="menu-boundaries"),
            dbc.Button("Saved versions", color="primary", className="mr-1", id="menu-save-versions"),
            # dbc.NavItem(dbc.NavLink("Boundaries", id="menu-boundaries", href="#")),
            dcc.Dropdown(
                id="input_column",
                options=column_options,
                value=column_options[0].get("label"),
                style={
                    "max-width": "300px",
                    "min-width": "250px",
                },
            ),
        ],
        brand="Skorecard | Manual Bucketing App",
        brand_href="#",
        color="primary",
        dark=True,
        fluid=True,
        fixed="top",
    )

    self.app.layout = html.Div(
        children=[
            dbc.Row(navbar, style={"padding-bottom": "3em"}),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(),
                        dbc.Col(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Boundaries"),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        [
                                                            "Original boundaries: ",
                                                            html.Code([""], id="original_boundaries"),
                                                            html.Br(),
                                                            "Updated boundaries: ",
                                                            html.Code([""], id="updated_boundaries"),
                                                        ],
                                                        className="card-text",
                                                    ),
                                                    dbc.Button(
                                                        "Reset boundaries",
                                                        id="reset-boundaries-button",
                                                        color="primary",
                                                        className="mr-2",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            className="mb-4",
                        ),
                    ]
                ),
                id="collapse-menu-boundaries",
            ),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(),
                        dbc.Col(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Save versions"),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        [
                                                            "TODO",
                                                        ],
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            className="mb-4",
                        ),
                    ]
                ),
                id="collapse-menu-save-versions",
            ),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(),
                        dbc.Col(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Model Performance"),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        [
                                                            "TODO",
                                                        ],
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            className="mb-4",
                        ),
                    ]
                ),
                id="collapse-menu-model-performance",
            ),
            dbc.Row(
                [dbc.Col(dcc.Graph(id="graph-prebucket"), width=6), dbc.Col(dcc.Graph(id="graph-bucket"), width=6)]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(children="pre-bucketing table"),
                                html.Div(children=[], id="pre-bucket-error"),
                                dash_table.DataTable(
                                    id="prebucket_table",
                                    style_data={"whiteSpace": "normal", "height": "auto"},
                                    style_cell={
                                        "height": "auto",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "maxWidth": 0,
                                        "textAlign": "center",
                                    },
                                    style_cell_conditional=[
                                        {"if": {"column_id": "range"}, "width": "180px"},
                                    ],
                                    style_as_list_view=True,
                                    page_size=20,
                                    columns=[
                                        {"name": "pre-bucket", "id": "pre-bucket", "editable": False},
                                        {"name": "label", "id": "label", "editable": False},
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
                                    editable=True,
                                ),
                            ],
                            style={"padding": "0 1em 0 1em", "width": "100%"},
                        ),
                        style={"margin": "0 1em 0 1em"},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(
                                    [
                                        "bucketing table",
                                    ]
                                ),
                                dbc.Collapse(
                                    dbc.Badge("is not monotonic", color="danger", className="mr-1"),
                                    id="is_not_monotonic_badge",
                                ),
                                dbc.Collapse(
                                    dbc.Badge("one or more buckets <5% obs", color="danger", className="mr-1"),
                                    id="has_5perc_badge",
                                ),
                                dash_table.DataTable(
                                    id="bucket_table",
                                    style_data={"whiteSpace": "normal", "height": "auto"},
                                    style_cell={
                                        "height": "auto",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "maxWidth": 0,
                                        "textAlign": "center",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                    },
                                    style_data_conditional=perc_data_bars("Count (%)")
                                    + perc_data_bars("Event Rate")
                                    + [
                                        {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                                        {
                                            "if": {"state": "active"},  # 'active' | 'selected'
                                            "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                            "border": "1px solid rgb(0, 116, 217)",
                                        },
                                    ]
                                    + colorize_cell("bucket"),
                                    style_as_list_view=True,
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
                                        # {"name": "bucket", "id": "bucket"},
                                        # {"name": "range", "id": "range"},
                                        # {"name": "count", "id": "count"},
                                        # {"name": "count %", "id": "count %"},
                                        # {"name": "Non-event", "id": "Non-event"},
                                        # {"name": "Event", "id": "Event"},
                                        # {"name": "Event Rate", "id": "Event Rate"},
                                        # {"name": "WoE", "id": "WoE"},
                                        # {"name": "IV", "id": "IV"},
                                    ],
                                    editable=False,
                                ),
                            ],
                            style={"padding": "0 1em 0 1em", "width": "100%"},
                        ),
                        style={"margin": "0 1em 0 1em"},
                    ),
                ],
                no_gutters=False,
                justify="center",
            ),
        ],
        style={"margin": "1em", "padding:": "1em"},
    )
