import pandas as pd
import pytest

from skorecard.apps.app_utils import determine_boundaries
from skorecard.bucket_mapping import BucketMapping


@pytest.mark.parametrize(
    "map_input, right_input,",
    [([2, 3, 4, 5], True), ([2, 3, 4, 5], False), ([3], True), ([3], False), ([], True), ([], False)],
)
def test_determine_boundaries(map_input, right_input):
    """Tests function."""
    df = pd.DataFrame()
    df["pre_buckets"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    bucket_mapping = BucketMapping("feature1", "numerical", map=map_input, right=right_input)
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)
