import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from skorecard.preprocessing import WoeEncoder
from category_encoders.woe import WOEEncoder


from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin
from skorecard import Skorecard


def calibrate_to_master_scale(y_pred, *, pdo, ref_score, ref_odds, epsilon=1e-6):
    """Calibrate the score to the master scale.

    It's common practice to represent the model predictions on a 'master scale'.
    The rescaling step is defined as follows:
        A reference score is set that corresponds to some reference odds of being class 0.
        By adding pdo (points to double the odds), the odds will double.

    Example:
        ref_score = 300 for ref_odds 50:1, and pdo = 25
        if the odds become 100:1, the score is 325.
        if the odds become 200:1, the score is 350.
        if the odds become 25:1, the score is 275.

    Args:
        y_pred: predicted probabilities
        pdo: number of points necessary to double the odds
        ref_score: reference score set for the reference odds
        ref_odds: odds that correspond to the ref_score
        epsilon: float (1e-6), correction to avoid infinite odds if y_pred = 0

    Returns:
        pd.Series (or np.array if y_pred was a np.array), rescaled integer scores
    """
    is_np = False
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
        is_np = True
    odd_func = lambda x: (1 - x + epsilon) / (x + epsilon)
    odds = y_pred.apply(odd_func)

    factor = pdo / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)

    master_scale = odds.apply(lambda x: _map_to_scale(x, factor, offset))

    if is_np:
        return master_scale.values
    else:
        return master_scale


def _map_to_scale(x, factor, offset):
    """Helper function."""
    try:
        return int(round(factor * np.log(x) + offset))
    except ValueError:
        return -999


class ScoreCardPoints(BaseEstimator, TransformerMixin):
    """Transformer to map the the buckets from the skorecard model and maps them to the rescaled points.

    Examples:

    ```python
    from skorecard import Skorecard
    from skorecard.rescale import ScoreCardPoints
    from skorecard.datasets import load_uci_credit_card

    X,y = load_uci_credit_card(return_X_y=True)
    model = Skorecard(variables = ["LIMIT_BAL", "BILL_AMT1","EDUCATION", "MARRIAGE"])
    model.fit(X, y)

    scp = ScoreCardPoints(model)
    scp.transform(X)
    ```

    """

    def __init__(self, skorecard_model, *, pdo=20, ref_score=100, ref_odds=1):
        """
        Args:
            skorecard_model: the fitted Skorecard class
            pdo: number of points necessary to double the odds
            ref_score: reference score set for the reference odds
            ref_odds: odds that correspond to the ref_score
        """
        assert isinstance(skorecard_model, Skorecard), (
            f"The skorecard_model must be an instance of "
            f"skorecard.Skorecard, got {skorecard_model.__class__.__name__} instead."
        )
        check_is_fitted(skorecard_model)
        self.skorecard_model = skorecard_model
        # self.pipeline = skorecard_model.pipeline
        self.pdo = pdo
        self.ref_score = ref_score
        self.ref_odds = ref_odds
        self._get_pipeline_elements()
        self._calculate_scorecard_points()

    def _get_pipeline_elements(self):

        bucketers = self.skorecard_model.pipeline_.named_steps["bucketer"]
        woe_enc = self.skorecard_model.pipeline_.named_steps["encoder"]
        self.features = self.skorecard_model.variables
        self.model = self.skorecard_model.pipeline_.named_steps["model"]

        assert hasattr(self.model, "predict_proba"), (
            f"Expected a model at the end of the pipeline, " f"got {self.model.__class__}"
        )
        if not (isinstance(woe_enc, WoeEncoder) or isinstance(woe_enc, WOEEncoder)):
            raise ValueError("Pipeline must have WoE encoder")

        fbm = bucketers.features_bucket_mapping_

        if len(self.features) == 0:
            # there is no feature selector
            self.features = fbm.columns
        woe_dict = woe_enc.mapping

        self.buckets = {k: fbm.get(k) for k in fbm.columns if k in self.features}
        self.woes = {k: woe_dict[k] for k in woe_dict.keys() if k in self.features}

    def _calculate_scorecard_points(self):

        # Put together the features in a list of table, containing all the buckets.
        list_dfs = list()
        for ix, col in enumerate(self.features):
            df_ = (
                pd.concat([pd.Series(self.buckets[col].labels), pd.Series(self.woes[col])], axis=1)
                .reset_index()
                .rename(columns={"index": "bin_index", 0: "map", 1: "woe"})
            )
            df_.loc[:, "feature"] = col

            df_.loc[:, "coef"] = self.model.coef_[0][ix]
            #
            list_dfs.append(df_)

        # Reduce the list of tables, to build the final scorecard feature points
        scorecard = reduce(lambda x, y: pd.concat([x, y]), list_dfs)
        scorecard = scorecard.append(
            {"feature": "Intercept", "coef": self.model.intercept_[0], "bin_index": 0, "map": 0, "woe": 0},
            ignore_index=True,
        )
        #     return buckets, woes
        scorecard["contribution"] = scorecard["woe"] * scorecard["coef"]

        self.scorecard = _scale_scorecard(
            scorecard, pdo=self.pdo, ref_score=self.ref_score, ref_odds=self.ref_odds, features=self.features
        )

        self.points_mapper = dict()
        for feat in self.scorecard["feature"].unique():
            one_feat_df = self.scorecard.loc[self.scorecard["feature"] == feat, ["bin_index", "Points"]]
            self.points_mapper[feat] = {
                k: v for k, v in zip(one_feat_df["bin_index"].values, one_feat_df["Points"].values)
            }

    def get_scorecard_points(self):
        """Get the scorecard points."""
        return self.scorecard

    def transform(self, X):
        """Transform the features to the points."""
        X_buckets = self.skorecard_model.pipeline_.named_steps["bucketer"].transform(X)

        bin_points = pd.concat(
            [
                X_buckets[feat].map(self.points_mapper[feat])
                for feat in self.points_mapper.keys()
                if feat != "Intercept"
            ],
            axis=1,
        )
        bin_points.index = X.index

        return bin_points


def _scale_scorecard(df, *, pdo, ref_score, ref_odds, features):
    """Equations to scale the feature scorecards.
    Args:
        df: Pandas DataFrame
        pdo: number of points necessary to double the odds
        ref_score: reference score set for the reference odds
        ref_odds: odds that correspond to the ref_score
        features (list): The list of features

    """
    df = df[~df["woe"].isnull()]

    factor = pdo / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)

    intercept = df.loc[df["feature"] == "Intercept"]["coef"].values[0]
    n = len(features)

    df_out = df.copy()
    df_out["Points"] = (offset / n) - (df_out["contribution"] + (intercept / n)) * factor
    df_out.loc[df_out["feature"] == "Intercept", "Points"] = 0

    df_out["Points"] = df_out["Points"].apply(lambda x: int(round(x, 0)))

    return df_out
