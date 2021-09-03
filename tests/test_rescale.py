from skorecard.rescale import calibrate_to_master_scale
from skorecard.bucketers import DecisionTreeBucketer, OrdinalCategoricalBucketer
from sklearn.pipeline import make_pipeline
from skorecard import Skorecard
from skorecard.rescale import ScoreCardPoints
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def probas():
    """Proba array fixture."""
    return np.array([0.1 * i for i in range(1, 10)])


def test_master_scale_calibration(probas):
    """Test that the master scale calibrations follows the expected logic."""
    scores = calibrate_to_master_scale(probas, pdo=20, ref_odds=1, ref_score=100)
    # add odds 1:1, te score is 100. This are probas of 50%

    # When proba is 0.2, the odds are 4:1. This means that the score should have increased by twice the pdo
    assert scores[1] == 140

    # When proba is 0.8, the odds are 1:4. This means that the score should have decreased by twice the pdo
    assert scores[7] == 60

    scores = calibrate_to_master_scale(probas, pdo=30, ref_odds=1, ref_score=100)
    # add odds 1:1, te score is 100. This are probas of 50%

    # When proba is 0.2, the odds are 4:1. This means that the score should have increased by twice the pdo
    assert scores[1] == 160

    # When proba is 0.8, the odds are 1:4. This means that the score should have decreased by twice the pdo
    assert scores[7] == 40


def test_scorecard_rescaling(df):
    """Test scorecard rescaling."""
    features = ["LIMIT_BAL", "BILL_AMT1", "EDUCATION", "MARRIAGE"]

    X = df[features]
    y = df["default"]

    scorecard_model = Skorecard(
        bucketing=make_pipeline(
            DecisionTreeBucketer(variables=features[:2]),
            OrdinalCategoricalBucketer(variables=features[2:]),
        )
    )
    with pytest.raises(NotFittedError):
        ScoreCardPoints(skorecard_model=scorecard_model, pdo=25, ref_score=400, ref_odds=20)

    scorecard_model.fit(X, y)

    scorecard = ScoreCardPoints(
        skorecard_model=scorecard_model, pdo=25, ref_score=400, ref_odds=20
    ).get_scorecard_points()

    # Contains all the features and the intercept
    assert len(scorecard.loc[:, "feature"].unique().tolist()) == len(features) + 1

    # The intercept must have 0 points
    assert scorecard.loc[scorecard["feature"] == "Intercept", "Points"].values[0] == 0

    # Make sure transform works
    sc = ScoreCardPoints(skorecard_model=scorecard_model, pdo=25, ref_score=400, ref_odds=20)
    assert isinstance(sc.transform(X), pd.DataFrame)
