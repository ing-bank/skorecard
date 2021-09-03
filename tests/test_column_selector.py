import pytest
from skorecard.preprocessing import ColumnSelector
import pandas as pd

from skorecard.bucketers import DecisionTreeBucketer, OrdinalCategoricalBucketer
from skorecard.preprocessing import WoeEncoder
from skorecard.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def test_column_selector(df):
    """Test that the column selector works."""
    features = ["LIMIT_BAL", "BILL_AMT1"]
    X = df
    # y = df["default"]

    cs = ColumnSelector(variables=features)
    X_sel = cs.fit_transform(X)

    assert isinstance(X_sel, pd.DataFrame)
    assert X_sel.shape[1] == len(features)

    # test that it raises an exception if X is not a pd DataFrame
    npX = X.values
    cs = ColumnSelector(variables=features)
    with pytest.raises(KeyError):
        cs.fit_transform(npX)


def test_column_selector_in_pipeline(df):
    """Test that the column selector works well in a pipeline."""
    features = ["LIMIT_BAL", "BILL_AMT1"]
    X = df.drop("default", axis=1)
    y = df["default"]

    model_pipe = make_pipeline(
        DecisionTreeBucketer(variables=["LIMIT_BAL"], max_n_bins=5),
        OrdinalCategoricalBucketer(variables=["MARRIAGE"], tol=0.05),
        WoeEncoder(),
        ColumnSelector(variables=features),
        LogisticRegression(calculate_stats=True),
    ).fit(X, y)

    # test that the model pipe has as many coefficients as there are features.
    assert len(model_pipe[-1].coef_[0]) == len(features)

    # test that the stats of the model pipe represent the correct features with a selector.
    model_stats = model_pipe[-1].get_stats().index.tolist()
    model_stats.remove("const")
    assert model_stats == features
