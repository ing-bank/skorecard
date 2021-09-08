from sklearn.tree import DecisionTreeClassifier
import pytest

from skorecard.utils.validation import check_args


def test_checks_args():
    """
    Tests checking arguments.
    """
    args = {"hi": 1}
    with pytest.warns(UserWarning):
        check_args(args, DecisionTreeClassifier)

    # check no warning is raised with valid arg
    args = {"min_samples_leaf": 1}
    with pytest.warns(None) as record:
        check_args(args, DecisionTreeClassifier)

    assert len(record) == 0
