from sklearn.pipeline import make_pipeline
from skorecard.bucketers.bucketers import DecisionTreeBucketer, EqualWidthBucketer
import numpy as np
import pandas as pd
import pytest
from skorecard.bucket_mapping import BucketMapping, merge_bucket_mapping, MissingDict
from skorecard.features_bucket_mapping import FeaturesBucketMapping, merge_features_bucket_mapping


def test_bucket_mapping_numerical():
    """Tests numerical transforms."""
    bm = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=True)
    assert bm.labels == {0: "(-inf, 1.0]", 1: "(1.0, 3.0]", 2: "(3.0, 5.0]", 3: "(5.0, inf]", -1: "Missing"}
    assert all(bm.transform([1, 3, 5, 7]) == np.array([0, 1, 2, 3]))

    # with right=False
    bm = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=False)
    assert bm.labels == {0: "[-inf, 1.0)", 1: "[1.0, 3.0)", 2: "[3.0, 5.0)", 3: "[5.0, inf)", -1: "Missing"}
    assert all(bm.transform([1, 3, 5, 7]) == np.array([1, 2, 3, 3]))

    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4])
    assert all(np.equal(bucket.transform(x), np.digitize(x, [3, 4], right=True)))
    # array([0, 0, 0, 0, 1, 2])

    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert all(np.equal(bucket.transform(x), np.digitize(x, [2, 3, 4], right=True)))
    # array([0, 0, 0, 1, 2, 3])

    x = [0, 1, np.nan, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert np.allclose(bucket.transform(x), np.array([0, 0, -1, 1, 2, 3]), equal_nan=True)
    assert "Missing" in bucket.labels.values()


def test_bucket_mapping_categorical():
    """
    Tests categorical transforms.
    """
    # Make sure that the map outputs start at 0 and are incremental. Because it is skipping 2,it will raise an exception
    with pytest.raises(ValueError):
        bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0, "boat": 2})

    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={})
    other_category_encoding = -2
    assert all(np.equal(bucket.transform(x), [other_category_encoding] * 5))

    # Empty map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={})
    other_category_encoding = -2
    missing_cat = -1
    assert all(np.equal(bucket.transform(x), [other_category_encoding] * 5 + [missing_cat]))

    # # Limited map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = -2
    reference = [0, other_category_encoding, other_category_encoding, 0, 0]
    assert all(np.equal(bucket.transform(x), reference))

    # Limited map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = -2
    missing_cat = -1
    reference = [0, other_category_encoding, other_category_encoding, 0, 0, missing_cat]
    assert all(np.equal(bucket.transform(x), reference))

    # Specials defined
    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    assert a.transform([np.nan]) == -1

    a = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        missing_bucket=None,
        other_bucket=None,
        map={2: 0, 3: 1, 5: 2, 6: 3, 4: 4, 0: 5},
        right=True,
        specials={"ed 0": [1]},
    )
    assert a.transform([1]) == -3

    # Make sure dtypes is OK
    a = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        missing_bucket=None,
        other_bucket=None,
        map={2: 0, 1: 1, 3: 2},
        right=True,
        specials={},
    )
    # number of bits in .astype(int) differs per OS
    assert a.transform([1, 2, 3]).dtype in [np.dtype("int32"), np.dtype("int64")]


def test_cat_other_bucket():
    """
    Test using 'other_bucket' parameter.
    """
    with pytest.raises(AssertionError):
        bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0}, other_bucket=1)

    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 1}, other_bucket=1)
    assert bucket.labels == {0: "car", 1: "truck | Other", -1: "Missing"}
    assert bucket.transform(["bla"]).tolist() == [1]

    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 1})
    assert bucket.transform(["bla"]).tolist() == [-2]


def test_reserved_names():
    """
    Test using reserved names.
    """
    a = BucketMapping("feature1", "categorical", map={"other": 0, "truck": 1})
    assert a.transform(["other"]).tolist() == [0]
    assert a.transform(["sdfsfsa"]).tolist() == [-2]


def test_get_labels():
    """Make sure nicely formatting is returned."""
    bucket = BucketMapping("feature1", "numerical", map=[1, 3, 4], right=True)
    assert bucket.labels == {0: "(-inf, 1.0]", 1: "(1.0, 3.0]", 2: "(3.0, 4.0]", 3: "(4.0, inf]", -1: "Missing"}

    bucket = BucketMapping("feature1", "numerical", map=[1, 3, 4], right=False)
    assert bucket.labels == {0: "[-inf, 1.0)", 1: "[1.0, 3.0)", 2: "[3.0, 4.0)", 3: "[4.0, inf)", -1: "Missing"}

    bucket = BucketMapping("feature1", "numerical", map=[1], right=True)
    assert bucket.labels == {0: "(-inf, 1.0]", 1: "(1.0, inf]", -1: "Missing"}

    bucket = BucketMapping("feature1", "numerical", map=[1], right=False)
    assert bucket.labels == {0: "[-inf, 1.0)", 1: "[1.0, inf)", -1: "Missing"}

    bucket = BucketMapping("feature1", "numerical", map=[], right=True)
    assert bucket.labels == {0: "(-inf, inf]", -1: "Missing"}

    bucket = BucketMapping("feature1", "numerical", map=[], right=False)
    assert bucket.labels == {0: "[-inf, inf)", -1: "Missing"}


def test_specials_numerical():
    """Test that the specials are put in a special bin."""
    # test that a special case
    x = [0, 1, 2, 3, 4, 5, 2]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=2": [2]})
    assert all(np.equal(bucket.transform(x), np.array([0, 0, -3, 0, 1, 2, -3])))

    # test that calling transform again does not change the labelling
    assert bucket.labels == {0: "(-inf, 3.0]", 1: "(3.0, 4.0]", 2: "(4.0, inf]", -1: "Missing", -3: "Special: =2"}
    assert all(bucket.transform(x) == bucket.transform(x))
    assert bucket.labels == {0: "(-inf, 3.0]", 1: "(3.0, 4.0]", 2: "(4.0, inf]", -1: "Missing", -3: "Special: =2"}

    # Test that if special is not in x, nothing happens
    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=6": [6]})
    assert all(np.equal(bucket.transform(x), np.digitize(x, [3, 4], right=True)))

    # Test multiple specials
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=2": [2], "=3or4": [3, 4]})
    assert bucket.labels == {
        0: "(-inf, 3.0]",
        1: "(3.0, 4.0]",
        2: "(4.0, inf]",
        -1: "Missing",
        -3: "Special: =2",
        -4: "Special: =3or4",
    }


def test_labels():
    """Test that the labels are correct in different scenarios."""
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    bins = pd.Series(bucket.transform(x))

    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    # labels with specials for categorical are not so nice
    # we cannot include the special name because item become part of self.map
    assert labels[in_series == "truck"].equals(labels[labels == "Special: is truck"])
    assert labels[(in_series.isin(["car", "boat"]))].equals(labels[labels == "boat, car"])

    # test with numerical categories
    # Limited map with NA's
    x = [310, 311, 312, 313, 313, np.nan]
    bucket = BucketMapping("feature1", "categorical", map={310: 0, 311: 1, 312: 2}, specials={"is 313": [313]})
    bins = pd.Series(bucket.transform(x))

    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    assert labels[in_series == 313].equals(labels[labels == "Special: is 313"])
    assert labels[in_series == 311].equals(labels[labels == "311"])
    assert labels[in_series.isna()].equals(labels[labels == "Missing"])

    # test numerical labels
    # note that .transform() is not necessary to build the labels.
    x = [0, 1, 2, 3, 4, 5, 2, np.nan]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=2": [2]})
    assert bucket.labels == {0: "(-inf, 3.0]", 1: "(3.0, 4.0]", 2: "(4.0, inf]", -1: "Missing", -3: "Special: =2"}
    bins = pd.Series(bucket.transform(x))
    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    assert labels[(in_series <= 3) & (in_series != 2)].equals(labels[labels == "(-inf, 3.0]"])
    assert labels[(in_series <= 4) & (in_series > 3)].equals(labels[labels == "(3.0, 4.0]"])
    assert labels[in_series == 2].equals(labels[labels == "Special: =2"])
    assert labels[in_series > 4].equals(labels[labels == "(4.0, inf]"])


def test_error_is_raised_if_wrong_specials():
    """Self explanatory."""
    # Test that is the values of the dictionary
    with pytest.raises(AssertionError):
        BucketMapping("feature1", "numerical", map=[3, 4], specials={"special": 2})
    #
    with pytest.raises(AssertionError):
        BucketMapping("feature1", "numerical", map=[3, 4], specials={0: [2]})


def test_features_bucket_mapping():
    """
    Test contruction feature bucketing mapping.
    """
    a = BucketMapping("testfeat1", "numerical", map=[1, 3, 5], right=True)
    b = BucketMapping("testfeat2", "numerical", map=[1, 3, 5], right=False)
    fbm = FeaturesBucketMapping([a, b])
    assert fbm.get("testfeat1") == a
    assert fbm.get("testfeat2") == b
    assert len(fbm) == 2

    # features_dict = {
    #     'feature1': {'feature_name': 'feature1',
    #         'type': 'numerical',
    #         'map': [2, 3, 4, 5],
    #         'right': True},
    #     'feature2': {'feature_name': 'feature2',
    #         'type': 'numerical',
    #         'map': [5, 6, 7, 8],
    #         'right': True}
    # }
    # assert FeaturesBucketMapping(features_dict) == FeaturesBucketMapping().load_dict(features_dict)


def test_merge_bucket_mapping_numerical():
    """
    Test merging two numerical bucketmappings.
    """
    a = BucketMapping(
        feature_name="LIMIT_BAL",
        type="numerical",
        map=[
            25000.0,
            45000.0,
            55000.0,
            75000.0,
            85000.0,
            105000.0,
            145000.0,
            175000.0,
            225000.0,
            275000.0,
            325000.0,
            385000.0,
        ],
        right=False,
    )

    b = BucketMapping(
        feature_name="LIMIT_BAL",
        type="numerical",
        map=[1, 2, 4, 6, 7, 9, 10, 12],
        right=False,
    )

    c = merge_bucket_mapping(a, b)

    x = [25000, 45000]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    x = [
        -1,
        0,
        24999,
        25000.0,
        45000.0,
        55000.0,
        75000.0,
        85000.0,
        105000.0,
        145000.0,
        175000.0,
        225000.0,
        275000.0,
        325000.0,
        385000.0,
        400000,
    ]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    # Now first bucketer with right=True, second one with right=False
    a = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=True)
    assert all(a.transform([1, 3, 5, 7]) == np.array([0, 1, 2, 3]))
    b = BucketMapping("testfeat", "numerical", map=[1, 3], right=False)
    assert all(b.transform([1, 3, 5, 7]) == np.array([1, 2, 2, 2]))
    c = merge_bucket_mapping(a, b)
    x = [-1, 0, 0.99, 1, 1.001, 3, 3.1, 4.9, 5, 5.1]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    # Now with specials
    a = BucketMapping(
        feature_name="LIMIT_BAL",
        type="numerical",
        map=[
            25000.0,
            45000.0,
            55000.0,
            75000.0,
            85000.0,
            105000.0,
            145000.0,
            175000.0,
            225000.0,
            275000.0,
            325000.0,
            385000.0,
        ],
        right=False,
        specials={"=45000.0": [45000.0]},
    )

    b = BucketMapping(
        feature_name="LIMIT_BAL",
        type="numerical",
        map=[1, 2, 4, 6, 7, 9, 10, 12],
        right=False,
        specials={"=45000.0": [14]},
    )

    c = merge_bucket_mapping(a, b)
    assert all(c.transform(x) == b.transform(a.transform(x)))


def test_merge_bucket_mapping_categorical():
    """
    Test merging two categorical bucketmappings.
    """
    a = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        map={
            0: 6,
            1: 0,
            2: 1,
            3: 2,
            4: 5,
            5: 3,
            6: 4,
        },
        right=True,
        specials={},
    )
    b = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        map={
            0: 0,
            1: 2,
            2: 1,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        },
        right=False,
        specials={},
    )

    correct_map = {0: 0, 1: 0, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0}
    c = merge_bucket_mapping(a, b)
    assert c.map == correct_map

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    # another case
    a = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        map={2: 0, 1: 1, 3: 2},
        right=True,
        specials={},
    )
    b = BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        map={3: 0, 1: 0, 2: 1, 0: 2},
        right=False,
        specials={},
    )

    c = merge_bucket_mapping(a, b)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert all(c.transform(x) == b.transform(a.transform(x)))


def test_missing_dict():
    """
    Test MissingDict.
    """
    s = pd.Series(["cat", "dog", np.nan, "rabbit"])
    a = {"cat": "kitten", "dog": "puppy"}
    a = MissingDict(a)
    a.set_missing_value("bye")
    assert all(s.map(a) == pd.Series(["kitten", "puppy", "bye", "bye"]))
    assert pd.isnull(s.map(a, na_action="ignore")[2])


def test_missing_bucket():
    """
    Test missing values are treated properly.
    """
    # missing_bucket='separate'
    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    assert a.transform(x)[5] == -1

    # missing_bucket=<bucket that exists>
    for missing_bucket in [0]:
        a = BucketMapping(
            "feature1",
            "categorical",
            map={"car": 0, "boat": 0},
            specials={"is truck": ["truck"]},
            missing_bucket=missing_bucket,
        )
        x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
        assert a.transform(x)[5] == missing_bucket

    # missing_bucket=<bucket that does not exist in map>
    # should raise an error!
    with pytest.raises(AssertionError):
        a = BucketMapping(
            "feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]}, missing_bucket=7
        )

    # numerical. missing_bucket='separate'
    a = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=True)
    x = [1, 3, 5, np.nan, 7]
    assert all(a.transform(x) == np.array([0, 1, 2, -1, 3]))

    # numerical. missing_bucket=1
    for missing_bucket in range(3):
        a = BucketMapping("testfeat", "numerical", missing_bucket=missing_bucket, map=[1, 3, 5], right=True)
        x = [1, 3, 5, np.nan, 7]
        assert all(a.transform(x) == np.array([0, 1, 2, missing_bucket, 3]))

    # numerical missing_bucket=<bucket that does not exist in map>
    # should raise an error!
    with pytest.raises(AssertionError):
        a = BucketMapping("testfeat", "numerical", missing_bucket=4, map=[1, 3, 5], right=True)


def test_other_bucket():
    """
    Tests other_bucket param.
    """
    # other_bucket=<bucket that exists>
    for other_bucket in [0]:
        a = BucketMapping(
            "feature1",
            "categorical",
            map={"car": 0, "boat": 0},
            specials={"is truck": ["truck"]},
            other_bucket=other_bucket,
        )
        x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
        assert a.transform(x)[2] == other_bucket

    # other_bucket=<bucket that does not exist in map>
    # should raise an error!
    with pytest.raises(AssertionError):
        a = BucketMapping(
            "feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]}, other_bucket=7
        )


def test_missing_other_bucket():
    """
    Test transforms when both other_bucket and missing_bucket are defined.
    """
    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    assert a.labels == {0: "boat, car", -2: "Other", -1: "Missing", -3: "Special: is truck"}
    assert a.transform(x)[1] == -2  # other
    assert a.transform(x)[4] == -3  # special
    assert a.transform(x)[5] == -1  # missing

    # other_bucket=<bucket that exists>
    for bucket in [0]:
        a = BucketMapping(
            "feature1",
            "categorical",
            map={"car": 0, "boat": 0},
            specials={"is truck": ["truck"]},
            other_bucket=bucket,
            missing_bucket=bucket,
        )
        x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
        assert a.transform(x)[2] == bucket
        assert a.transform(x)[5] == bucket

    # other_bucket=<bucket that does not exist in map>
    # should raise an error!
    with pytest.raises(AssertionError):
        a = BucketMapping(
            "feature1",
            "categorical",
            map={"car": 0, "boat": 0},
            specials={"is truck": ["truck"]},
            other_bucket=7,
            missing_bucket=7,
        )


def test_merge_cats_with_missing():
    """
    Testing merges.
    """
    # Test wrong 'other_bucket' set
    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    b = BucketMapping(
        "feature1", "categorical", map={0: 0, 1: 1, 3: 1, 2: 0}, other_bucket=1, specials={"is truck": [-3]}
    )
    with pytest.raises(AssertionError):
        c = merge_bucket_mapping(a, b)

    # Test wrong 'missing_bucket' set
    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    b = BucketMapping(
        "feature1", "categorical", map={0: 0, 1: 1, 3: 1, 2: 0}, missing_bucket=1, specials={"is truck": [-3]}
    )
    with pytest.raises(AssertionError):
        c = merge_bucket_mapping(a, b)

    a = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    b = BucketMapping(
        "feature1", "categorical", map={0: 0, -1: 0, -2: 0}, missing_bucket=0, specials={"is truck": [-3]}
    )
    c = merge_bucket_mapping(a, b)
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert all(c.transform(x) == b.transform(a.transform(x)))

    # first bucket with lots of cats
    # second with very few
    a = BucketMapping(
        "feature1", "categorical", map={"car": 0, "boat": 1, "motorcycle": 2}, specials={"is truck": ["truck"]}
    )
    b = BucketMapping(
        "feature1",
        "categorical",
        map={0: 0, 1: 0, 2: 1},
        other_bucket=0,
        missing_bucket=0,
        specials={"is truck": [-3]},
    )

    c = merge_bucket_mapping(a, b)
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan, "something else"]
    assert all(c.transform(x) == b.transform(a.transform(x)))


def test_merge_features_bucket_mapping():
    """
    Tests merging two sets.
    """
    a = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=True)
    b = BucketMapping("testfeat", "numerical", map=[1, 3], right=False)
    c = merge_bucket_mapping(a, b)

    af = FeaturesBucketMapping([a])
    bf = FeaturesBucketMapping([b])

    assert FeaturesBucketMapping([c]) == merge_features_bucket_mapping(af, bf)

    # now wiht new keys in either
    a = BucketMapping("testfeat", "numerical", map=[1, 3, 5], right=True)
    b = BucketMapping("testfeat", "numerical", map=[1, 3], right=False)
    c = BucketMapping("new_feat", "numerical", map=[5, 6], right=False)

    af = FeaturesBucketMapping([a, c])
    bf = FeaturesBucketMapping([b])

    assert FeaturesBucketMapping([merge_bucket_mapping(a, b), c]) == merge_features_bucket_mapping(af, bf)


def test_merge_buckets_on_data(df):
    """
    Test on actual data.
    """
    df = df.drop(columns=["pet_ownership", "BILL_AMT1"])
    y = df["default"]
    X = df.drop(columns=["default"])

    buck_a = EqualWidthBucketer(n_bins=100)
    buck_b = DecisionTreeBucketer()

    pipe = make_pipeline(buck_a, buck_b)
    pipe.fit(X, y)

    a = pipe.steps[0][1].features_bucket_mapping_
    b = pipe.steps[1][1].features_bucket_mapping_
    c = merge_features_bucket_mapping(a, b)

    for feature in X.columns:
        assert all(c.get(feature).transform(X[feature]) == pipe.transform(X)[feature].values)


def test_item_assignment():
    """
    Test if assignment bucket mappings by feature name works properly.
    """
    bm = FeaturesBucketMapping()
    a = BucketMapping("feature1", "categorical", map={310: 0, 311: 1, 312: 2}, specials={"is 313": [313]})
    b = BucketMapping("feature1", "categorical", map={310: 0, 311: 1, 312: 2}, specials={"is 999": [313]})

    bm.append(a)
    bm["feature1"] = b
    assert len(bm) == 1
    assert bm.get("feature1") == b
