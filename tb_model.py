import numpy as np
import pandas as pd
import math

class BayesianTBModel:
    def __init__(self):
        self.feature_probs = {}
        self.prior_tb = 0.5

    def fit(self, df: pd.DataFrame, pseudo: float = 0.5) -> None:
        features = [c for c in df.columns if c != "TB"]
        self.prior_tb = float(df["TB"].mean())

        for f in features:
            tb = df[df["TB"] == 1][f]
            ntb = df[df["TB"] == 0][f]
            self.feature_probs[f] = {
                1: (tb.sum() + pseudo) / (len(tb) + 2 * pseudo),   # P(f=1 | TB=1)
                0: (ntb.sum() + pseudo) / (len(ntb) + 2 * pseudo), # P(f=1 | TB=0)
            }

    def predict_proba(self, x: dict) -> float:
        log_odds = math.log(self.prior_tb / (1 - self.prior_tb))

        for f, v in x.items():
            p1 = min(max(float(self.feature_probs[f][1]), 0.01), 0.99)
            p0 = min(max(float(self.feature_probs[f][0]), 0.01), 0.99)

            if v == 1:
                log_odds += math.log(p1 / p0)
            else:
                log_odds += math.log((1 - p1) / (1 - p0))

        log_odds = max(-8.0, min(8.0, log_odds))
        return 1.0 / (1.0 + math.exp(-log_odds))