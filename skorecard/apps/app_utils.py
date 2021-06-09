import pandas as pd

from skorecard.bucket_mapping import BucketMapping
from skorecard.reporting.plotting import get_bucket_colors


def determine_boundaries(df: pd.DataFrame, bucket_mapping: BucketMapping) -> list:
    """
    Determine mapping boundaries.

    Given a dataframe with pre_bucket and bucket column, determine the boundaries
    that can be passed to the bucket_mapping.

    ```python
    import pandas as pd
    from skorecard.bucket_mapping import BucketMapping
    df = pd.DataFrame()
    df['pre_buckets'] = [0,1,2,3,4,5,6,7,8,9,10]
    df['buckets'] = [0,0,1,1,2,2,2,3,3,4,5]

    bucket_mapping = BucketMapping('feature1', 'numerical', map = [2,3,4,5])

    determine_boundaries(df, bucket_mapping)
    ```
    """
    assert "pre_buckets" in df.columns
    assert "buckets" in df.columns

    # if bucket_mapping.type != "numerical":
    #     raise NotImplementedError("todo")

    dfg = df.groupby(["buckets"]).agg(["max"])
    dfg.columns = dfg.columns.get_level_values(1)
    boundaries = dfg["max"]
    if bucket_mapping.right is False:
        # the prebuckets are integers
        # So we can safely add 1 to make sure the
        # map includes the right prebuckets
        boundaries += 1

    # Drop the last value,
    # This makes sure outlier values are in the same bucket
    # instead of a new one
    boundaries = list(boundaries)[:-1]

    # Removed this because of the categoricals
    # assert sorted(boundaries) == boundaries, "buckets must be sorted"
    return boundaries


def perc_data_bars(column):
    """
    Display bar plots inside a dash DataTable cell.

    Assumes a value between 0 - 100.

    Adapted from: https://dash.plotly.com/datatable/conditional-formatting
    """
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [float(x) for x in range(101)]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        # For odd rows
        styles.append(
            {
                "if": {
                    "filter_query": (
                        "{{{column}}} >= {min_bound}"
                        + (" && {{{column}}} < {max_bound}" if (i < len(bounds) - 1) else "")
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    "column_id": column,
                    "row_index": "odd",
                },
                "background": (
                    """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    rgb(248, 248, 248) {max_bound_percentage}%,
                    rgb(248, 248, 248) 100%)
                """.format(
                        max_bound_percentage=max_bound_percentage
                    )
                ),
                "paddingBottom": 2,
                "paddingTop": 2,
            }
        )
        # For even rows
        styles.append(
            {
                "if": {
                    "filter_query": (
                        "{{{column}}} >= {min_bound}"
                        + (" && {{{column}}} < {max_bound}" if (i < len(bounds) - 1) else "")
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    "column_id": column,
                    "row_index": "even",
                },
                "background": (
                    """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(
                        max_bound_percentage=max_bound_percentage
                    )
                ),
                "paddingBottom": 2,
                "paddingTop": 2,
            }
        )

    return styles


def colorize_cell(column):
    """Colourize the integer bucket number.

    We can safely assume max 20 buckets, as features are often binned to 3-7 buckets.
    We will cycle through them.
    """
    colors = get_bucket_colors()

    styles = []
    for i in range(21):
        styles.append(
            {
                "if": {
                    # 'row_index': i,  # number | 'odd' | 'even'
                    "filter_query": f"{{{column}}} = '{i}'",
                    "column_id": column,
                },
                "backgroundColor": colors[i % len(colors)],
                "color": "white",
            }
        )
    return styles
