import pandas as pd
from skorecard.bucket_mapping import BucketMapping
from skorecard.apps.app_utils import determine_boundaries


def test_determine_boundaries():
    """Tests function."""
    df = pd.DataFrame()
    df["pre_buckets"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    bucket_mapping = BucketMapping("feature1", "numerical", map=[2, 3, 4, 5])
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)

    bucket_mapping = BucketMapping("feature1", "numerical", map=[2, 3, 4, 5], right=False)
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)

    bucket_mapping = BucketMapping("feature1", "numerical", map=[3])
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)

    bucket_mapping = BucketMapping("feature1", "numerical", map=[3], right=False)
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)

    bucket_mapping = BucketMapping("feature1", "numerical", map=[])
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)

    bucket_mapping = BucketMapping("feature1", "numerical", map=[], right=False)
    df["buckets"] = bucket_mapping.transform(df["pre_buckets"])
    assert bucket_mapping.map == determine_boundaries(df, bucket_mapping)
