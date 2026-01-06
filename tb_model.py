import numpy as np

class BayesianTBModel:
    def __init__(self):
        self.feature_probs = {}
        self.prior_tb = 0.5

    def fit(self, df, pseudo=0.5):
        features = df.columns.drop("TB")
        self.prior_tb = df["TB"].mean()

        for f in features:
            tb = df[df["TB"] == 1][f]
            ntb = df[df["TB"] == 0][f]

            self.feature_probs[f] = {
                1: (tb.sum() + pseudo) / (len(tb) + 2*pseudo),
                0: (ntb.sum() + pseudo) / (len(ntb) + 2*pseudo)
            }

    def predict_proba(self, x):
        log_odds = np.log(self.prior_tb / (1 - self.prior_tb))
        
        neg_weight = 0.3  # tune 0..1

        for f, v in x.items():
            p1 = self.feature_probs[f][1]
            p0 = self.feature_probs[f][0]

            p1 = min(max(p1, 0.01), 0.99)
            p0 = min(max(p0, 0.01), 0.99)

            log_odds += np.log(p1 / p0) if v == 1 else np.log((1-p1)/(1-p0))

        log_odds = np.clip(log_odds, -8, 8)
        return 1 / (1 + np.exp(-log_odds))
