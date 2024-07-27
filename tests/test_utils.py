import pytest
from sklearn.tree import DecisionTreeClassifier
import warnings

from skorecard.utils.validation import check_args


@pytest.mark.parametrize("args, nr_of_warnings", [({"hi": 1}, 1), ({"min_samples_leaf": 1}, 0)])
def test_checks_args(args, nr_of_warnings):
    """
    Tests checking arguments.
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        check_args(args, DecisionTreeClassifier)
        return len(caught_warnings) == nr_of_warnings
