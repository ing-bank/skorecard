from skorecard.bucketers import DecisionTreeBucketer
from skorecard.reporting import build_bucket_table
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def test_report_decision_tree(df):
    """Test the reporting module."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]
    tbt = DecisionTreeBucketer(max_n_bins=4, min_bin_size=0.1, variables=["LIMIT_BAL", "BILL_AMT1"])
    tbt.fit(X, y)
    tbt.transform(X)

    df_out = build_bucket_table(X, y, column="LIMIT_BAL", bucketer=tbt)
    assert df_out.shape == (5, 9)
    # Make sure bucket table equals feature bucket mapping dict
    assert (
        dict(zip(df_out["bucket_id"].values, df_out["label"].values))
        == tbt.features_bucket_mapping_.get("LIMIT_BAL").labels
    )

    expected = pd.DataFrame(
        {"bucket_id": {0: -1, 1: 0, 2: 1, 3: 2, 4: 3}, "Count": {0: 0.0, 1: 849, 2: 676, 3: 1551, 4: 2924}}
    )
    pd.testing.assert_frame_equal(df_out[["bucket_id", "Count"]], expected)

    np.testing.assert_array_equal(
        df_out.columns.ravel(),
        np.array(
            [
                "bucket_id",
                "label",
                "Count",
                "Count (%)",
                "Non-event",
                "Event",
                "Event Rate",
                # "% Event",
                # "% Non Event",
                "WoE",
                "IV",
            ]
        ),
    )


def test_report_consinstency(df):
    """
    Test that the reported defaults match the ones in the sample.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]
    tbt = DecisionTreeBucketer(max_n_bins=4, min_bin_size=0.1, variables=["LIMIT_BAL", "BILL_AMT1"])
    tbt.fit(X, y)
    tbt.transform(X)

    df_out = build_bucket_table(X, y, column="LIMIT_BAL", bucketer=tbt)
    assert df_out["Event"].sum() == y.sum()

    # adds random selection in the indexing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    df_out = build_bucket_table(X_train, y_train, column="LIMIT_BAL", bucketer=tbt)
    assert df_out["Event"].sum() == y_train.sum()

    df_out = build_bucket_table(X_test, y_test, column="LIMIT_BAL", bucketer=tbt)
    assert df_out["Event"].sum() == y_test.sum()
