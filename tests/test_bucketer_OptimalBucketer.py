import pytest
import numpy as np
from optbinning import OptimalBinning
from sklearn.pipeline import make_pipeline

from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.bucket_mapping import BucketMapping
from skorecard.utils import NotPreBucketedError


def test_optimal_binning_prebinning(df):
    """Ensure we have prevented prebinning correctly.

    optbinning.OptimalBinning() does pre-binning by default.
    In skorecard, we want more control, so force the user to do this explicitly.

    Extracting only the optimizer from optbinning is quite involved.
    Instead, we'll let the prebinning (a DecisionTreeClassifier) use our user defined splits.
    These splits are simply the unique values from the prebucketed feature.

    This tests checks the output is not changed.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"].values

    # optbinning saves the user_defined_splits for prebinning
    # https://github.com/guillermo-navas-palencia/optbinning/blob/396b9bed97581094167c9eb4744c2fd1fb5c7408/optbinning/binning/binning.py#L741
    # then applies np.digitize(right=False)
    # https://github.com/guillermo-navas-palencia/optbinning/blob/396b9bed97581094167c9eb4744c2fd1fb5c7408/optbinning/binning/binning.py#L1048
    # If these are the prebuckted values
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Internally this happens:
    binned_x = np.digitize(x=x, bins=x, right=False)
    # which means a shift by 1
    assert all(binned_x == x + 1)
    # which means internally it is shifted by 1, but that doesn't really matter,
    # as tests below show, because still same unique buckets.

    # transform a feature using OptBinning's prebucket + optimal binning
    binner = OptimalBinning(
        name="BILL_AMT1",
        dtype="numerical",
        solver="cp",
        monotonic_trend="auto_asc_desc",
        min_prebin_size=0.05,
        max_n_prebins=20,
        min_bin_size=0.05,
        max_n_bins=10,
        cat_cutoff=0.05,
        time_limit=25,
    )
    old = binner.fit_transform(X["BILL_AMT1"], y, metric="indices")

    # transform a feature using skorecard prebinning + optimal binning
    pipe = make_pipeline(
        DecisionTreeBucketer(variables=["BILL_AMT1"], max_n_bins=20, min_bin_size=0.05),
        OptimalBucketer(variables=["BILL_AMT1"], max_n_bins=10, min_bin_size=0.05),
    )
    new = pipe.fit_transform(X, y)["BILL_AMT1"]

    assert all(np.equal(old, new.values))

    # now for categoricals


def test_optimal_binning_numerical(df):
    """Tests the wrapper of optbinning.OptimalBinning()."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"].values

    X_prebucketed = DecisionTreeBucketer(max_n_bins=20, min_bin_size=0.05).fit_transform(X, y)

    obt = OptimalBucketer(variables=["LIMIT_BAL", "BILL_AMT1"])
    with pytest.raises(NotPreBucketedError):
        obt.fit(X, y)
    obt.fit(X_prebucketed, y)
    X_trans = obt.transform(X_prebucketed)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9
    assert len(X_trans["BILL_AMT1"].unique()) == 6

    obt = OptimalBucketer(variables=["LIMIT_BAL"])
    obt.fit(X_prebucketed, y)
    X_trans = obt.transform(X_prebucketed)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9

    # Note our bins were 1-indexed in the past
    # This unit test is there to make sure our bins are zero-indexed
    obt.fit(X, y)
    assert all(obt.features_bucket_mapping_.get("LIMIT_BAL").transform([0, 30_000]) == np.array([0, 1]))

    # optb.binning_table.build()
    # optb.splits
    # obt.features_bucket_mapping_.get('LIMIT_BAL')


def test_optimal_binning_categorical(df):
    """Test categoricals."""
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION"]]
    y = df["default"].values

    obt = OptimalBucketer(variables=["EDUCATION"], variables_type="categorical")
    obt.fit(X, y)
    X_trans = obt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 3
    assert obt.features_bucket_mapping_.get("EDUCATION") == BucketMapping(
        feature_name="EDUCATION",
        type="categorical",
        map={0: 0, 6: 0, 4: 0, 5: 0, 1: 0, 3: 1, 2: 2},
        right=False,
    )

    optb = OptimalBinning(
        name="EDUCATION", dtype="categorical", solver="cp", cat_cutoff=None, max_n_prebins=100, max_n_bins=10
    )
    optb.fit(X["EDUCATION"], y)
    ref = optb.transform(X["EDUCATION"], metric="indices")

    assert all(X_trans["EDUCATION"].values == ref)


def test_raises_prebucketing_error(df):
    """
    Test prebucketing error.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION"]]
    y = df["default"].values

    obt = OptimalBucketer(variables=["BILL_AMT1"])
    with pytest.raises(NotPreBucketedError):
        obt.fit_transform(X, y)


def test_optbinning_with_specials(df):
    """
    Test adding specials.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION"]]
    y = df["default"].values

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    obt = OptimalBucketer(variables=["LIMIT_BAL"], specials=specials)
    obt.fit_transform(X, y)


def test_optimal_binning_categorical_specials(df):
    """
    Test categoricals with specials.
    """
    # WIP - currently not implemented yet
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "pet_ownership"]]
    y = df["default"].values

    specials = {"EDUCATION": {"special_one": [0]}}
    obt = OptimalBucketer(variables=["EDUCATION"], variables_type="categorical", specials=specials)
    obt.fit(X, y)
    X_trans = obt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 4

    assert obt.transform(X)["EDUCATION"][X["EDUCATION"] == 0].shape[0] == 1

    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "pet_ownership"]]
    y = df["default"].values
    specials = {"pet_ownership": {"indoor pets outside water": ["dog lover", "cat lover"]}}
    obt = OptimalBucketer(variables=["pet_ownership"], max_n_bins=2, variables_type="categorical", specials=specials)
    X_trans = obt.fit_transform(X, y)
    # import pdb; pdb.set_trace();
    assert len(X_trans.pet_ownership.unique()) == 2 + 1  # 2 bins plus 1 speclias


def test_optimal_bucketing_cats(df):
    """
    Test with categorical strings.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "pet_ownership"]]
    y = df["default"].values
    # specials = {'pet_ownership': {"indoor pets outside water": ['dog lover','cat lover']}}

    # obt.features_bucket_mapping_.get('pet_ownership')
    obt = OptimalBucketer(variables=["pet_ownership"], variables_type="categorical")
    X_trans = obt.fit_transform(X, y)
    # assert len(X_trans.pet_ownership.unique()) == 5

    obt = OptimalBucketer(variables=["pet_ownership"], max_n_bins=3, variables_type="categorical")
    X_trans = obt.fit_transform(X, y)
    # assert len(X_trans.pet_ownership.unique()) == 3+1

    obt = OptimalBucketer(variables=["pet_ownership"], max_n_bins=2, variables_type="categorical")
    X_trans = obt.fit_transform(X, y)
    # assert len(X_trans.pet_ownership.unique()) == 2+1

    obt = OptimalBucketer(variables=["LIMIT_BAL"], max_n_bins=2)
    X_trans = obt.fit_transform(X, y)
    assert len(X_trans["LIMIT_BAL"].unique()) == 2


def test_missing_categorical_manual(df_with_missings):
    """
    Test.
    """
    X = df_with_missings[["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "pet_ownership"]]
    y = df_with_missings["default"].values
    obt = OptimalBucketer(
        variables=["pet_ownership", "EDUCATION"],
        max_n_bins=2,
        variables_type="categorical",
        missing_treatment={"pet_ownership": 0},
    )
    obt.fit(X[["pet_ownership", "EDUCATION"]], y)
    X_trans = obt.transform(X[["pet_ownership", "EDUCATION"]])  # ['pet_ownership']
    X["pet_ownership_trans"] = X_trans["pet_ownership"]
    X["default"] = y
    assert (
        X[X["pet_ownership"].isnull()]["pet_ownership_trans"].sum() == 0
    )  # Sums to 0 because all missings in bucket 0


def test_missing_numerical_manual(df_with_missings):
    """
    Test.
    """
    X = df_with_missings[["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "pet_ownership"]]
    y = df_with_missings["default"].values
    obt = OptimalBucketer(
        variables=["LIMIT_BAL"], max_n_bins=2, variables_type="numerical", missing_treatment={"LIMIT_BAL": 0}
    )
    obt.fit(X[["LIMIT_BAL"]], y)
    X["LIMIT_BAL_trans"] = obt.transform(X[["LIMIT_BAL"]])
    assert X[np.isnan(X["LIMIT_BAL"])]["LIMIT_BAL_trans"].sum() == 0  # Sums to 0 because all missings in bucket 0
