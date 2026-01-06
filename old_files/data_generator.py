import numpy as np
import pandas as pd

FEATURES = [
    "cough", "cough_gt_2w", "blood_in_sputum",
    "fever", "weight_loss", "night_sweats",
    "chest_pain", "breathing_problem",
    "fatigue", "loss_of_appetite",
    "contact_with_TB"
]

def generate_data(n=10000, tb_prevalence=0.6, seed=42):
    np.random.seed(seed)
    rows = []

    for _ in range(n):
        is_tb = np.random.rand() < tb_prevalence
        row = {}

        if is_tb:
            probs = {
                "cough": 0.95, "cough_gt_2w": 0.90, "blood_in_sputum": 0.80,
                "fever": 0.75, "weight_loss": 0.80, "night_sweats": 0.75,
                "chest_pain": 0.50, "breathing_problem": 0.60,
                "fatigue": 0.75, "loss_of_appetite": 0.65,
                "contact_with_TB": 0.70
            }
        else:
            probs = {
                "cough": 0.25, "cough_gt_2w": 0.08, "blood_in_sputum": 0.01,
                "fever": 0.20, "weight_loss": 0.10, "night_sweats": 0.08,
                "chest_pain": 0.15, "breathing_problem": 0.20,
                "fatigue": 0.30, "loss_of_appetite": 0.15,
                "contact_with_TB": 0.05
            }

        for f in FEATURES:
            row[f] = int(np.random.rand() < probs[f])

        # enforce cough logic
        if row["cough"] == 0:
            row["cough_gt_2w"] = 0
            row["blood_in_sputum"] = 0

        row["TB"] = int(is_tb)
        rows.append(row)

    return pd.DataFrame(rows)
