import pandas as pd

from skorecard.utils.exceptions import NotInstalledError

from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")

try:
    from IPython.display import Image
except ModuleNotFoundError:
    Image = NotInstalledError("psutil")


def make_plot_figure(bucket_table: pd.DataFrame):
    """
    Make a plotly object out of a table.
    """
    # To support both pre-buckets and buckets
    if "pre-bucket" in bucket_table.columns:
        buckets = [b for b in bucket_table["pre-bucket"].values]
    else:
        buckets = [b for b in bucket_table["bucket"].values]

    plotdf = pd.DataFrame(
        {
            "bucket": buckets,
            "label": [x for x in bucket_table["label"].values],
            "counts": [int(count) for count in bucket_table["Count"].values],
            "counts %": [float(count / 100) for count in bucket_table["Count (%)"].values],
        }
    )

    # If the bucket_table is built without any 'y' information
    # we won't know any event rate rates either
    # and thus we need to output a simpler plot
    if "Event Rate" in bucket_table.columns:
        plotdf["Event Rate"] = [event for event in bucket_table["Event Rate"].values]
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add trace with event rate
        fig.add_trace(
            go.Scatter(x=plotdf["label"], y=plotdf["Event Rate"], name="Event Rate", line=dict(color="#454c57")),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="bucket event rate", secondary_y=True, tickformat=",.0%")
    else:
        fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add trace with counts
    fig.add_trace(
        go.Bar(x=plotdf["label"], y=plotdf["counts %"], name="Bucket count percentage"),
        secondary_y=False,
    )
    fig.update_yaxes(title_text="bucket size", secondary_y=False, tickformat=",.0%")

    # If we have bucket information, use that to colorize the bars
    # that means a prebucket table without information on the corresponding buckets
    # wont have bars colorized.
    if "bucket" in bucket_table.columns:
        bar_colors = [get_bucket_color(i) for i in bucket_table["bucket"].values]
        fig.update_traces(marker=dict(color=bar_colors), selector=dict(type="bar"))

    # Other stuff
    fig.update_layout(transition_duration=50)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig


def plot_prebucket_table(prebucket_table, column="", format=None, scale=None, width=None, height=None):
    """
    Given the prebucketed data, plot the pre-buckets.

    Args:
        prebucket_table (pd.DataFrame): the table of the prebucketed data
        X (pd.DataFrame): [description]
        y ([type], optional): [description]. Defaults to None.
        column (str): The column to plot
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig
        scale: If format is specified, the scale of the image
        width: If format is specified, the width of the image
        height: If format is specified, the image of the image

    Returns:
        fig of desired format
    """
    fig = make_plot_figure(prebucket_table)

    fig.update_layout(title=f"pre-buckets: {column}".strip())
    fig.update_layout(xaxis_title=f"{column} pre-buckets".strip())

    if format:
        img_bytes = fig.to_image(format=format, scale=scale, width=width, height=height)
        fig = Image(img_bytes)
    return fig


def plot_bucket_table(bucket_table, column="", format=None, scale=None, width=None, height=None):
    """
    Given the bucketed data, plot the buckets with Event Rate.

    Args:
        bucket_table (pd.DataFrame): the table of the bucketed data
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig
        scale: If format is specified, the scale of the image
        width: If format is specified, the width of the image
        height: If format is specified, the image of the image

    Returns:
        plotly fig
    """
    fig = make_plot_figure(bucket_table)

    fig.update_layout(title=f"buckets: {column}".strip())
    fig.update_layout(xaxis_title=f"{column} buckets".strip())

    if format is not None:
        img_bytes = fig.to_image(format=format, scale=scale, width=width, height=height)
        fig = Image(img_bytes)

    return fig


def get_bucket_color(i):
    """Return diverging color for unique buckets.

    Generated using:

    ```python
    import seaborn as sns
    colors = sns.color_palette("Set2")
    rgbs = []
    for r,g,b in list(colors):
        rgbs.append(
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        )
    ```

    Args:
        i: number to return a color string for
    """
    if i < 0:
        return "rgb(189,189,189)"  # light grey for specials
    elif i % 8 == 0:
        return "rgb(102,194,165)"
    elif i % 8 == 1:
        return "rgb(252,141,98)"
    elif i % 8 == 2:
        return "rgb(141,160,203)"
    elif i % 8 == 3:
        return "rgb(231,138,195)"
    elif i % 8 == 4:
        return "rgb(166,216,84)"
    elif i % 8 == 5:
        return "rgb(255,217,47)"
    elif i % 8 == 6:
        return "rgb(229,196,148)"
    elif i % 8 == 7:
        return "rgb(179,179,179)"
    else:
        raise NotImplementedError("")


class PlotPreBucketMethod:
    """
    Add methods for plotting bucketing tables to another class.

    To be used with skorecard.pipeline.BucketingProcess and skorecard.bucketers.BaseBucketer
    """

    def plot_prebucket(self, column, format=None, scale=None, width=None, height=None):
        """
        Generates the prebucket table and produces a corresponding plotly plot.

        Args:
            column: The column we want to visualise
            format: The format of the image, such as 'png'. The default None returns a plotly image.
            scale: If format is specified, the scale of the image
            width: If format is specified, the width of the image
            height: If format is specified, the image of the image

        Returns:
            plot: plotly fig
        """
        if isinstance(self, Pipeline):
            check_is_fitted(self.steps[0][1])
        else:
            check_is_fitted(self)
        return plot_prebucket_table(
            prebucket_table=self.prebucket_table(column),
            column=column,
            format=format,
            scale=scale,
            width=width,
            height=height,
        )


class PlotBucketMethod:
    """
    Add methods for plotting bucketing tables to another class.

    To be used with skorecard.pipeline.BucketingProcess and skorecard.bucketers.BaseBucketer
    """

    def plot_bucket(self, column, format=None, scale=None, width=None, height=None):
        """
        Plot the buckets.

        Args:
            column: The column we want to visualise
            format: The format of the image, such as 'png'. The default None returns a plotly image.
            scale: If format is specified, the scale of the image
            width: If format is specified, the width of the image
            height: If format is specified, the image of the image

        Returns:
            plot: plotly fig
        """
        if isinstance(self, Pipeline):
            check_is_fitted(self.steps[0][1])
        else:
            check_is_fitted(self)

        return plot_bucket_table(
            bucket_table=self.bucket_table(column=column),
            column=column,
            format=format,
            scale=scale,
            width=width,
            height=height,
        )
