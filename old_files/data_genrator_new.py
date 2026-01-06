import numpy as np
import pandas as pd


def create_dataset(n=8000, tb_prevalence=0.55, seed=42):
    np.random.seed(seed)
    rows = []

    for _ in range(n):
        # 1. TB ground truth
        tb = np.random.binomial(1, tb_prevalence)

        # 2. Primary symptom: cough
        #cough = np.random.binomial(1, 0.85 if tb else 0.35)
        cough = np.random.binomial(1, 0.95 if tb else 0.35)

        if cough == 1:
            #cough_gt_2w = np.random.binomial(1, 0.65 if tb else 0.20)
            cough_gt_2w = np.random.binomial(1, 0.85 if tb else 0.20)
            blood_in_sputum = np.random.binomial(1, 0.50 if tb else 0.05)
        else:
            cough_gt_2w = 0
            blood_in_sputum = 0

        # 3. Respiratory involvement
        breathing_problem = np.random.binomial(1, 0.55 if tb else 0.15)
        chest_pain = np.random.binomial(1, 0.45 if tb else 0.10)

        # 4. Systemic symptoms
        fever = np.random.binomial(1, 0.60 if tb else 0.20)
        fatigue = np.random.binomial(1, 0.65 if tb else 0.30)
        weight_loss = np.random.binomial(1, 0.60 if tb else 0.10)
        loss_of_appetite = np.random.binomial(1, 0.55 if tb else 0.15)
        night_sweats = np.random.binomial(1, 0.55 if tb else 0.15)

        # 5. Epidemiological risk
        contact_with_TB = np.random.binomial(1, 0.50 if tb else 0.05)
        rows.append({
            "cough": cough,
            "cough_gt_2w": cough_gt_2w,
            "blood_in_sputum": blood_in_sputum,
            "breathing_problem": breathing_problem,
            "fever": fever,
            "night_sweats": night_sweats,
            "fatigue": fatigue,
            "weight_loss": weight_loss,
            "loss_of_appetite": loss_of_appetite,
            "chest_pain": chest_pain,
            "contact_with_TB": contact_with_TB,
            "TB": tb
        })

    df = pd.DataFrame(rows)

    # Absolute safety
    df = df.fillna(0)

    return df.astype(int)
if __name__ == "__main__":

    OUTPUT_PATH = "data/tb_training_data.csv"
    N_SAMPLES = 8000
    TB_PREVALENCE = 0.55

    import os
    os.makedirs("data", exist_ok=True)

    df = create_dataset(
        n=N_SAMPLES,
        tb_prevalence=TB_PREVALENCE,
        seed=42
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(" Dataset generated")
    print(f" Path: {OUTPUT_PATH}")
    print(f" Samples: {len(df)}")
    print(f" TB prevalence: {df['TB'].mean():.2%}")
