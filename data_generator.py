# data_generator.py
# Dataset generation + Bayesian model + evaluation helpers.

import math
import numpy as np
import pandas as pd

# ---------------------------
# 1) Original BayesianTBModel
# ---------------------------
from tb_model import BayesianTBModel



# ---------------------------
# 2) Testcases + helpers
# ---------------------------
ALL_FEATURES = [
    "cough",
    "cough_gt_2w",
    "blood_in_sputum",
    "breathing_problem",
    "fever",
    "night_sweats",
    "fatigue",
    "weight_loss",
    "loss_of_appetite",
    "chest_pain",
    "contact_with_TB",
]

TOKEN_TO_FEATURE = {
    "c": "cough",
    "c2": "cough_gt_2w",
    "b": "blood_in_sputum",
    "f": "fever",
    "w": "weight_loss",
    "ns": "night_sweats",
    "cp": "chest_pain",
    "bp": "breathing_problem",
    "fat": "fatigue",
    "loa": "loss_of_appetite",
    "conta": "contact_with_TB",
}

TESTS = [
    ("c,c2,b", "High"),
    ("c,c2", "Low"),
    ("c,c2,w", "High"),
    ("c,c2,ns", "High"),
    ("c,c2,cp,ns", "High"),
    ("c,c2,f", "Moderate"),
    ("c,c2,conta", "Moderate"),
    ("c,f,ns,fat", "Moderate"),
    ("c,w,loa,fat", "Moderate"),
    ("w,loa,fat", "Low"),
]


def make_case(tokens_csv: str) -> dict:
    x = {f: 0 for f in ALL_FEATURES}
    tokens = [t.strip() for t in tokens_csv.split(",") if t.strip()]
    for t in tokens:
        if t == "n":  # allow shorthand
            t = "ns"
        if t not in TOKEN_TO_FEATURE:
            raise ValueError(f"Unknown token '{t}'. Valid: {sorted(TOKEN_TO_FEATURE.keys())}")
        x[TOKEN_TO_FEATURE[t]] = 1
    return x


def risk_label(p: float, high_thr: float = 0.70, mod_thr: float = 0.30) -> str:
    if p >= high_thr:
        return "High"
    if p >= mod_thr:
        return "Moderate"
    return "Low"


def evaluate_df(df: pd.DataFrame, high_thr: float = 0.70, mod_thr: float = 0.30):
    model = BayesianTBModel()
    model.fit(df, pseudo=0.5)

    passed = 0
    rows = []
    for toks, exp in TESTS:
        p = float(model.predict_proba(make_case(toks)))
        pred = risk_label(p, high_thr=high_thr, mod_thr=mod_thr)
        ok = (pred == exp)
        passed += int(ok)
        rows.append((toks, exp, pred, p, ok))

    return passed, rows


# ---------------------------
# 3) Parametric dataset generator
#    (pulmonary + systemic TB phenotypes)
# ---------------------------
def create_dataset(
    n: int = 8000,
    tb_prevalence: float = 0.55,
    seed: int = 42,
    params: dict | None = None,
) -> pd.DataFrame:
    if params is None:
        raise ValueError("params is required")

    rng = np.random.default_rng(seed)
    tb = rng.binomial(1, tb_prevalence, size=n).astype(int)
    systemic = (rng.binomial(1, params["tb_systemic_frac"], size=n) * tb).astype(int)

    def pick(p_tb_pulm, p_tb_sys, p_notb):
        p = np.where(tb == 1, np.where(systemic == 1, p_tb_sys, p_tb_pulm), p_notb)
        return rng.binomial(1, p).astype(int)

    # cough
    cough = pick(params["p_cough_tb_pulm"], params["p_cough_tb_sys"], params["p_cough_notb"])

    # cough_gt_2w and blood depend on cough
    cough_gt_2w = pick(params["p_c2_tb_pulm"], params["p_c2_tb_sys"], params["p_c2_notb"])
    cough_gt_2w = (cough_gt_2w * cough).astype(int)

    blood = pick(params["p_blood_tb_pulm"], params["p_blood_tb_sys"], params["p_blood_notb"])
    blood = (blood * cough).astype(int)

    breathing_problem = pick(params["p_bp_tb_pulm"], params["p_bp_tb_sys"], params["p_bp_notb"])
    chest_pain = pick(params["p_cp_tb_pulm"], params["p_cp_tb_sys"], params["p_cp_notb"])

    fever = pick(params["p_fever_tb_pulm"], params["p_fever_tb_sys"], params["p_fever_notb"])
    night_sweats = pick(params["p_ns_tb_pulm"], params["p_ns_tb_sys"], params["p_ns_notb"])
    fatigue = pick(params["p_fat_tb_pulm"], params["p_fat_tb_sys"], params["p_fat_notb"])

    weight_loss = pick(params["p_w_tb_pulm"], params["p_w_tb_sys"], params["p_w_notb"])
    loss_of_appetite = pick(params["p_loa_tb_pulm"], params["p_loa_tb_sys"], params["p_loa_notb"])

    contact_with_TB = pick(params["p_conta_tb_pulm"], params["p_conta_tb_sys"], params["p_conta_notb"])

    df = pd.DataFrame(
        {
            "cough": cough,
            "cough_gt_2w": cough_gt_2w,
            "blood_in_sputum": blood,
            "breathing_problem": breathing_problem,
            "fever": fever,
            "night_sweats": night_sweats,
            "fatigue": fatigue,
            "weight_loss": weight_loss,
            "loss_of_appetite": loss_of_appetite,
            "chest_pain": chest_pain,
            "contact_with_TB": contact_with_TB,
            "TB": tb,
        }
    ).astype(int)

    return df
