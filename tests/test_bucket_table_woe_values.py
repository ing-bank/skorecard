import numpy as np
import pandas as pd

from skorecard import Skorecard
from skorecard.datasets import load_uci_credit_card


def test_bucket_table_woe_values():
    """Checks whether or not the WoE-values of bucket_table()are equivalent to those in the transformed data."""
    data = load_uci_credit_card()
    X = data["data"]
    y = data["target"]

    model = Skorecard()
    model = model.fit(X, y)
    X_woe = model.woe_transform(X)
    for c in X.columns:
        bucket_table = model.bucket_table(c)
        b_tab_woes = set(bucket_table["WoE"])
        b_tab_woes = {x for x in b_tab_woes if pd.notna(x)}
        data_woes = set(np.round(X_woe[c].value_counts().index, 3))
        assert b_tab_woes == data_woes
