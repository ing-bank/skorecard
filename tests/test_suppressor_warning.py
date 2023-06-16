import warnings

from skorecard import Skorecard
from skorecard.datasets import load_uci_credit_card


def test_suppressor_warning():
    """Checks suppressor effect warning on Skorecard fit."""
    # Load the data. Construct datasets with and without suppressor effect occurring
    data = load_uci_credit_card()
    y = data["target"]

    X_no_suppression = data["data"]
    model = Skorecard()

    X_suppression = X_no_suppression.copy()
    X_suppression["suppressor"] = X_suppression[X_suppression.columns[0]] - X_suppression[X_suppression.columns[1]]
    model_suppression = Skorecard()

    with warnings.catch_warnings(record=True) as w:
        # Check that the suppressor warning is not issued and that no coefficient has an unexpected sign
        model = model.fit(X_no_suppression, y)
        relevant_warning_issued = 0
        if len(w) > 0:
            latest_warning = str(w[-1].message)
            msg = (
                "Features found with coefficient-sign that is contrary to what is expected based on weight-of-evidence."
            )
            relevant_warning_issued = msg in latest_warning
        coefs = model.coef_[0]
        suppression = any(c < 0 for c in coefs)
        assert (not relevant_warning_issued) & (not suppression)

        # Check that the suppressor warning is issued and that there is a coefficient with an unexpected sign
        model_suppression = model_suppression.fit(X_suppression, y)
        relevant_warning_issued = 0
        if len(w) > 0:
            latest_warning = str(w[-1].message)
            msg = (
                "Features found with coefficient-sign that is contrary to what is expected based on weight-of-evidence."
            )
            relevant_warning_issued = msg in latest_warning
        coefs = model_suppression.coef_[0]
        suppression = any(c < 0 for c in coefs)
        assert (relevant_warning_issued) & (suppression)
