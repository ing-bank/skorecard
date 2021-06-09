from skorecard.bucketers import AgglomerativeClusteringBucketer


def test_kwargs_are_saved(df):
    """Test that the kwargs fed to the AgglomerativeClusteringBucketer are saved."""
    X = df
    y = df["default"].values

    ab = AgglomerativeClusteringBucketer(variables=["LIMIT_BAL"], n_bins=7, compute_distances=True)
    ab.fit(X, y)

    assert ab.binners["LIMIT_BAL"].distances_ is not None
