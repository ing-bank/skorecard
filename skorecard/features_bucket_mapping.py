import yaml
import dataclasses

from skorecard.bucket_mapping import BucketMapping, merge_bucket_mapping


class FeaturesBucketMapping:
    """Stores a collection of features BucketMapping.

    ```python
    from skorecard.bucket_mapping import BucketMapping
    from skorecard.features_bucket_mapping import FeaturesBucketMapping

    # Working with collections of BucketMappings
    bucket1 = BucketMapping(feature_name='feature1', type='numerical', map=[2, 3, 4, 5])
    bucket2 = BucketMapping(feature_name='feature2', type='numerical', map=[5,6,7,8])
    features_bucket_mapping = FeaturesBucketMapping([bucket1, bucket2])
    print(features_bucket_mapping)

    # You can also work with class as dict
    features_bucket_mapping.as_dict()

    features_dict = {
        'feature1': {'feature_name': 'feature1',
            'type': 'numerical',
            'map': [2, 3, 4, 5],
            'right': True},
        'feature2': {'feature_name': 'feature2',
            'type': 'numerical',
            'map': [5, 6, 7, 8],
            'right': True}
    }

    features_bucket_mapping = FeaturesBucketMapping()
    features_bucket_mapping.load_dict(features_dict)
    # Or directly from dict
    FeaturesBucketMapping(features_dict)
    # See columns
    features_bucket_mapping.columns
    ```
    """

    def __init__(self, maps=[]):
        """Takes list of bucketmappings and stores as a dict.

        Args:
            maps (list): list of BucketMapping. Defaults to [].
        """
        self.maps = {}
        if isinstance(maps, list):
            for bucketmap in maps:
                self.append(bucketmap)

        if isinstance(maps, dict):
            for _, bucketmap in maps.items():
                if not isinstance(bucketmap, BucketMapping):
                    bucketmap = BucketMapping(**bucketmap)
                self.append(bucketmap)

    def __repr__(self):
        """Pretty print self.

        Returns:
            str: reproducable object representation.
        """
        class_name = self.__class__.__name__
        maps = list(self.maps.values())
        return f"{class_name}({maps})"

    def __len__(self):
        """
        Length of the map.
        """
        return len(self.maps)

    def __eq__(self, other):
        """
        Define equality.
        """
        return self.maps == other.maps

    def __getitem__(self, key):
        """
        Retrieve BucketMappings by feature name.
        """
        return self.maps[key]

    def __setitem__(self, key, value):
        """
        Set a bucketmapping using the feature name.
        """
        self.maps[key] = value

    def get(self, col: str):
        """Get BucketMapping for a column.

        Args:
            col (str): Name of column

        Returns:
            mapping (BucketMapping): BucketMapping for column
        """
        return self.maps[col]

    def append(self, bucketmap: BucketMapping) -> None:
        """Add a BucketMapping to the collection.

        Args:
            bucketmap (BucketMapping): map of a feature
        """
        assert isinstance(bucketmap, BucketMapping)
        self.maps[bucketmap.feature_name] = bucketmap

    def load_yml(self) -> None:
        """Should load in data from a yml.

        Returns:
            None: nothing
        """
        raise NotImplementedError("todo")

    def save_yml(self, file) -> None:
        """Should write data to a yml.

        Returns:
            None: nothing
        """
        if isinstance(file, str):
            file = open(file, "w")
        yaml.safe_dump(self.as_dict(), file)

    def load_dict(self, obj):
        """Should load in data from a python dict.

        Args:
            obj (dict): Dict with names of features and their BucketMapping

        Returns:
            None: nothing
        """
        assert isinstance(obj, dict)

        self.maps = {}
        for feature, bucketmap in obj.items():
            self.append(BucketMapping(**bucketmap))

    def as_dict(self):
        """Returns data in class as a dict.

        Returns:
            dict: Data in class
        """
        return {k: dataclasses.asdict(v) for k, v in self.maps.items()}

    @property
    def columns(self):
        """Returns the columns that have a bucket_mapping."""
        return list(self.as_dict().keys())


def merge_features_bucket_mapping(a: FeaturesBucketMapping, b: FeaturesBucketMapping) -> FeaturesBucketMapping:
    """
    Merge two sets of sequentual FeatureBucketMapping.
    """
    assert isinstance(a, FeaturesBucketMapping)
    assert isinstance(b, FeaturesBucketMapping)
    assert a.maps.keys() == b.maps.keys()

    features_bucket_mapping = FeaturesBucketMapping()

    for feature_name, bm in a.maps.items():
        c = merge_bucket_mapping(bm, b.get(feature_name))
        features_bucket_mapping.append(c)

    return features_bucket_mapping
