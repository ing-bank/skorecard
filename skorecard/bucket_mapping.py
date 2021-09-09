"""
Classes to store features mapping for bucketing.
"""
import dataclasses

from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional

import pandas as pd
import numpy as np


@dataclass
class BucketMapping:
    """Internal class to store all the bucketing info.

    This saves the information from a fitted bucketer and can transform any new data.

    Example:

    ```python
    from skorecard.bucket_mapping import BucketMapping

    # Manually a new bucket mapping for a feature
    bucket = BucketMapping('feature1', 'numerical', map = [2,3,4,5], specials={"special 0": [0]})
    print(bucket)

    # You can work with these classes as dicts as well
    bucket.as_dict()
    BucketMapping(**bucket.as_dict())

    # Transform new data
    bucket.transform(list(range(10)))
    ```

    Args:
        feature_name (str): Name of the feature
        type (str): Type of feature, one of ['categorical','numerical']
        missing_treatment (int): How missing values should be treated.
            If none, missing values get put in their separate bucket.
            If an int, this bucket number is where we will put the missing values.
        other_bucket (int):  Only for categoricals: If specified,
            the bucket number where any new values (not in the map) should be put.
        map (list or dict): The info needed to create the buckets (boundaries or cats)
        right (bool): parameter to np.digitize, used when map='numerical'.
        specials (dict): dictionary of special values to bin separately.
            The key is used for the labels, the value(s) is used to create a separate bucket.
        labels (dict): dictionary containing special values. It must be of the format:
            - keys: strings, containing the name (that will be used as labels) for the special values
            - values: lists, containing the special values
    """

    feature_name: str
    type: str
    missing_bucket: Optional[int] = None
    other_bucket: Optional[int] = None
    map: Union[Dict, List] = field(default_factory=lambda: [])
    right: bool = True
    specials: Dict[str, list] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """
        Do input validation, pre-calculations and build labels.
        """
        # Input validation
        assert self.type in ["numerical", "categorical"]
        assert len(self.map) is not None, "Please set a 'map' first"
        assert isinstance(self.specials, dict) or isinstance(self.specials, list)

        # Check specials
        assert all(
            [isinstance(k, str) for k in self.specials.keys()]
        ), f"The keys of the special dicionary must be \
        strings, got {self.specials.keys()} instead."
        assert all(
            [isinstance(k, list) for k in self.specials.values()]
        ), f"The values of the special dicionary must be a list of elements, got {self.specials}instead."
        # TODO: assert that special values are not present in multiple special buckets.

        # Make sure map is in correct format
        if isinstance(self.map, np.ndarray):
            self.map = self.map.tolist()

        # Determine the bucket numbers for reserved categories: 'missing'.
        # Missings are always -1 unless specified
        if self.type == "numerical":
            assert isinstance(self.map, list), "Map must be list"

            # 2 boundaries will give 3 buckets.
            # they are zero-indexed so max_bucket number equals len(self.map)
            max_bucket = len(self.map)

            if self.missing_bucket is not None:
                if not np.isnan(self.missing_bucket):
                    assert (
                        self.missing_bucket <= max_bucket
                    ), "map '%s' corresponds buckets 0-%s but missing_bucket is set to %s" % (
                        self.map,
                        max_bucket,
                        self.missing_bucket,
                    )
                self._missing_bucket = self.missing_bucket
            else:
                self._missing_bucket = -1

            # We leave -2 for 'other', whcih only applies to categoricals
            # There for consistency
            self._start_special_bucket = -3

            # Build labels
            self.labels = build_labels(
                self.map,
                right=self.right,
                missing_bucket=self._missing_bucket,
                specials=self.specials,
                start_special_bucket=self._start_special_bucket,
            )

        # Determine the bucket numbers for reserved categories: 'other' and 'missing'.
        if self.type == "categorical":

            assert isinstance(self.map, dict), "Map must be dict"

            # Assure the conversion from numpy.int to int
            new_dict = dict()
            for k, v in self.map.items():
                if isinstance(k, (np.int32, np.int64)):
                    k = int(k)
                if isinstance(v, (np.int32, np.int64)):
                    v = int(v)
                new_dict[k] = v

            self.map = new_dict
            self._validate_categorical_map()
            self.map = dict(self.map)  # type: Dict

            # Python 3.7+ has dicts that are OrderedDicts. Let's have a pretty ordering
            self.map = dict(sorted(self.map.items(), key=lambda x: (x[1], x[0])))

            # Set 'other' bucket
            if self.other_bucket is not None:
                assert self.other_bucket in self.map.values(), "other_bucket '%s' does not exist in map values: %s" % (
                    self.other_bucket,
                    self.map,
                )
                self._other_bucket = self.other_bucket
            else:
                self._other_bucket = -2

            # Set 'missing' bucket
            if self.missing_bucket is not None:
                # Allow -2, -1 Some missing_treatments (e.g. most_risky) add it here
                if self.missing_bucket not in [-2, -1, np.nan]:
                    assert (
                        self.missing_bucket in self.map.values()
                    ), "missing_bucket '%s' does not exist in map values: %s" % (self.missing_bucket, self.map)

                self._missing_bucket = self.missing_bucket
            else:
                self._missing_bucket = -1

            # Set specials bucket
            self._start_special_bucket = -3

            # Build labels
            self.labels = build_cat_labels(
                self.map,
                missing_bucket=self._missing_bucket,
                other_bucket=self._other_bucket,
                specials=self.specials,
                start_special_bucket=self._start_special_bucket,
            )

    def _validate_categorical_map(self):
        """
        Validate map structure.

        Assures that the provided mapping starts at 0 and that it has an incremental trend.
        """
        values = [v for v in self.map.values()]
        if len(values) > 0:
            if not np.array_equal(np.unique(values), np.arange(max(values) + 1)):
                err_msg = (
                    f"Mapping dictionary must start at 0 and be incremental. "
                    f"Found the following mappings {np.unique(values)}, and expected {np.arange(max(values) + 1)}"
                )
                raise ValueError(err_msg)

    def transform(self, x):
        """
        Apply binning using a boundaries map.

        Note:
        - resulting bins are zero-indexed

        ```python
        import numpy as np
        bins = np.array([-np.inf, 1, np.inf])
        x = np.array([-1,0,.5, 1, 1.5, 2,10, np.nan, 0])
        new = np.digitize(x, bins)
        np.where(np.isnan(x), np.nan, new)
        ```
        """
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        if isinstance(x, list):
            x = pd.Series(x)
        assert isinstance(x, pd.core.series.Series)
        
        # Workaround for missings
        def to_int(x):
            if not np.isnan(x):
                return int(x)
            else:
                return x

        # Transform using self.map
        if self.type == "numerical":
            buckets = np.digitize(x, self.map, right=self.right).astype(int)
            buckets = pd.Series(buckets)
        if self.type == "categorical":
            buckets = self._apply_cat_mapping(x)

        # Deal with missing values
        buckets = np.where(x.isnull(), self._missing_bucket, buckets)

        # Ensure dtype is integer buckets
        buckets = [to_int(x) for x in buckets]

        # Deal with special values
        # Both categorical & numerical
        special_counter = self._start_special_bucket
        for k, v in self.specials.items():
            buckets = np.where(x.isin(v), special_counter, buckets)
            special_counter -= 1
        return np.array(buckets)

    def _apply_cat_mapping(self, x):

        mapping = MissingDict(self.map)
        mapping.set_missing_value(self._other_bucket)  # This was 'other' but you cannot mix integers and strings

        # Note that we need to add .astype('Int64')
        # so that we have a nullable integer series.
        # This is because na_action='ignore' will convert from Int64 series to float64
        # Support for nullable integpyer arrays is a pandas 'gotcha'
        # Details discussed here:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#support-for-integer-na

        return x.map(mapping, na_action="ignore").astype("Int64")

    def as_dict(self) -> dict:
        """Return data in class as a dict.

        Returns:
            dict: data in class
        """
        return dataclasses.asdict(self)


class MissingDict(dict):
    """
    Deal with missing entries in a dict map.

    Because Pandas .map() uses the __missing__ method
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html

    Note the word 'missing' is confusing: It refers to the situation when a value is not
    in the dict map, and sets a default value instead.

    Missing values can be dealt with separately using map(na_action='ignore')

    Example usage:

    ```python
    s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
    a = {'cat': 'kitten', 'dog': 'puppy'}
    s.map(a)
    a = MissingDict(a)
    a.set_missing_value("bye")
    s.map(a)
    # make sure to ignore missing values!
    s.map(a, na_action='ignore')
    ```
    """

    def set_missing_value(self, value):
        """Setter for a missing value."""
        self.missing_value = value

    def __missing__(self, key):
        """Adds a default for missing values."""
        assert self.missing_value is not None, "Use .set_missing_value(key) first"
        return self.missing_value


def build_labels(
    boundaries,
    right: bool,
    missing_bucket: int,
    start_special_bucket: int,
    specials: Optional[Dict[str, list]],
) -> Dict[int, str]:
    """
    Build a nice label dict from a boundary.

    ```python
    assert build_labels([1,2,3]) == {
        0: '(-inf, 1.0]',
        1: '(1.0, 3.0]',
        2: '(3.0, 5.0]',
        3: '(5.0, inf]'
     }
    ```
    """
    boundaries = np.hstack([-np.inf, boundaries, np.inf]).tolist()
    labels = {}

    if right:
        b_left = "("
        b_right = "]"
    else:
        b_left = "["
        b_right = ")"

    for i, boundary in enumerate(boundaries):
        if i != len(boundaries) - 1:
            labels[i] = f"{b_left}{boundary}, {boundaries[i+1]}{b_right}"

    # reserve a label for missing values
    if missing_bucket in labels.keys():
        labels[missing_bucket] += " | Missing"
    else:
        labels[missing_bucket] = "Missing"

    # labels for specials
    if specials:
        for k, v in specials.items():
            labels[start_special_bucket] = "Special: " + str(k)
            start_special_bucket -= 1

    return labels


def build_cat_labels(
    boundaries,
    other_bucket: int,
    missing_bucket: int,
    start_special_bucket: int,
    specials: Optional[Dict[str, list]],
) -> Dict[int, str]:
    """
    Build categorical labels.
    """
    assert other_bucket is not None
    assert missing_bucket is not None

    # Switch value and key
    labels: Dict = {}
    for key in boundaries.keys():
        labels.setdefault(boundaries[key], []).append(str(key))

    # Merge lists into string
    for key in labels:
        labels[key] = ", ".join(labels[key])

    # add label for 'other'
    if other_bucket in labels.keys():
        labels[other_bucket] += " | Other"
    else:
        labels[other_bucket] = "Other"

    # reserve a label for missing values
    if missing_bucket in labels.keys():
        labels[missing_bucket] += " | Missing"
    else:
        labels[missing_bucket] = "Missing"

    # Add specials
    if specials:
        for k, v in specials.items():
            labels[start_special_bucket] = "Special: " + str(k)
            start_special_bucket -= 1

    # Make sure labels dict is nicely sorted by value, key
    labels = dict(sorted(labels.items(), key=lambda x: (x[1], x[0])))

    return labels


def merge_bucket_mapping(a, b):
    """
    Merges two bucketmappings into one.

    Assumption here is that one is for prebucketing and the other is for bucketing.
    In other words, one bucketmapping builds on the other one.
    """
    msg = f"Feature '{a.feature_name}' has variable_type '{a.type}' in a, but '{b.type}' in b."
    msg += "\nDid you set variable_type correctly in your (pre)bucketing pipeline?"
    assert a.type == b.type, msg

    if a.type == "categorical":

        if b.other_bucket:
            assert (
                b.other_bucket in a.labels.keys()
            ), f"b.other_bucket set to {b.other_bucket} but not present in any of a's buckets ({a.labels})"
        if b.missing_bucket:
            assert (
                b.missing_bucket in a.labels.keys()
            ), f"b.other_bucket set to {b.missing_bucket} but not present in any of a's buckets ({a.labels})"

        new_boundaries = {}
        for category, bucket in a.map.items():
            new_boundaries[category] = int(b.transform([bucket]))

        # let's also see where the 'other' category is assigned
        something_random = "84a088e251d2fa058f37145222e536dc"
        new_other_bucket = int(b.transform(a.transform([something_random])).tolist()[0])
        # if 'other' is put together with an existing bucket
        # manually assign that.
        if new_other_bucket in new_boundaries.values():
            other_bucket = new_other_bucket
        else:
            other_bucket = None

        # let's see where the missing category is assigned
        new_missing_bucket = int(b.transform(a.transform([np.nan])).tolist()[0])
        if new_missing_bucket in new_boundaries.values():
            missing_bucket = new_missing_bucket
        else:
            missing_bucket = None

        return BucketMapping(
            feature_name=a.feature_name,
            type=a.type,
            missing_bucket=missing_bucket,
            other_bucket=other_bucket,
            map=new_boundaries,
            specials=a.specials,
        )

    if a.type == "numerical":

        # This should hold for numerical maps
        assert len(a.map) >= len(b.map)

        # Add infinite edges to boundary map
        ref_map = [-np.inf] + a.map + [np.inf]

        new_buckets = list(b.transform(a.transform(ref_map)))

        # We take a.map and add inf edges, f.e.
        #  [-np.inf, 1,3,5, np.inf]
        # We can run it through both bucketers and get f.e.
        # [0,0,1,1,2]
        # If a.right = True, then we take the max per group
        # If a.right = False, then we take the min per group
        # Finally, remove any infinites
        new_boundaries = []

        if a.right:
            for i, new_bucket in enumerate(new_buckets):
                if i == 0:
                    if len(new_buckets) == 1:
                        new_boundaries.append(ref_map[0])
                        continue
                if i == len(new_buckets) - 1:
                    continue
                if new_buckets[i + 1] > new_bucket:
                    new_boundaries.append(ref_map[i])

        if not a.right:
            for i, new_bucket in enumerate(new_buckets):
                if i == 0:
                    if len(new_buckets) == 1:
                        new_boundaries.append(ref_map[0])
                        continue
                else:
                    if new_buckets[i - 1] < new_bucket:
                        new_boundaries.append(ref_map[i])

        new_boundaries = [x for x in new_boundaries if x != -np.inf]
        new_boundaries = [x for x in new_boundaries if x != np.inf]

        return BucketMapping(
            feature_name=a.feature_name,
            type=a.type,
            missing_bucket=a.missing_bucket,
            map=new_boundaries,
            specials=a.specials,
            right=a.right,
        )
