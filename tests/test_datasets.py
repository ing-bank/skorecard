from skorecard import datasets


def test_load_data_as_frame():
    """Test setting as_frame to True."""
    df = datasets.load_uci_credit_card(return_X_y=False, as_frame=True)
    assert df.shape == (6000, 5)

    return None


def test_load_data_as_frame_and_X_y():
    """Test setting both params to True gives an error."""
    try:
        X, y = datasets.load_uci_credit_card(return_X_y=True, as_frame=True)

    except ValueError:
        assert True

    return None


def test_load_data_X_y():
    """Test we get X and y numpy arrays."""
    X, y = datasets.load_uci_credit_card(return_X_y=True, as_frame=False)
    assert X.shape == (6000, 4)
    assert y.shape == (6000,)

    return None


def test_load_data_dict():
    """Test that setting both arguments to False returns a dict."""
    X_y_dict = datasets.load_uci_credit_card(return_X_y=False, as_frame=False)
    assert type(X_y_dict) == dict
    assert X_y_dict["data"].shape == (6000, 4)
    assert X_y_dict["target"].shape == (6000,)

    return None
