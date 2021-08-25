import random
import numpy as np
import pytest

from skorecard import datasets


@pytest.fixture()
def df():
    """Generate dataframe."""
    df = datasets.load_uci_credit_card(as_frame=True)
    # Add a fake categorical
    pets = ["no pets"] * 3000 + ["cat lover"] * 1500 + ["dog lover"] * 1000 + ["rabbit"] * 498 + ["gold fish"] * 2
    random.Random(42).shuffle(pets)
    df["pet_ownership"] = pets

    return df


@pytest.fixture()
def df_with_missings(df):
    """
    Add missing values to above df.
    """
    df_with_missings = df.copy()

    for col in ["EDUCATION", "MARRIAGE", "BILL_AMT1", "LIMIT_BAL", "pet_ownership"]:
        df_with_missings.loc[df_with_missings.sample(frac=0.2, random_state=42).index, col] = np.nan

    # Make sure there are 8 unique values (7 unique plus some NA)
    assert len(df_with_missings["EDUCATION"].unique()) == 8
    # Make sure there are NAs
    assert any([np.isnan(x) for x in df_with_missings["EDUCATION"].unique().tolist()])

    return df_with_missings
