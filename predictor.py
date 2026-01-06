import pickle

FEATURES = [
    "cough", "cough_gt_2w", "blood_in_sputum",
    "fever", "weight_loss", "night_sweats",
    "chest_pain", "breathing_problem",
    "fatigue", "loss_of_appetite",
    "contact_with_TB"
]

def load_model(path="model/tb_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, user_input):
    x = {f: int(user_input.get(f, 0)) for f in FEATURES}

    # enforce cough logic
    if x["cough"] == 0:
        x["cough_gt_2w"] = 0
        x["blood_in_sputum"] = 0

    # pulmonary issue rule (your requirement)
    if x["breathing_problem"] == 1 and sum(x.values()) == 1:
        return {"risk": "Pulmonary Issue", "tb_probability": 0.0}

    prob = model.predict_proba(x)

    # sensitive thresholds
    if prob >= 0.75:
        risk = "High TB Risk"
    elif prob >= 0.40:
        risk = "Moderate TB Risk"
    #elif prob >= 0.25:
    #    risk = "Low TB Risk"
    else:
        risk = "Low TB Risk"

    return {"risk": risk, "tb_probability": round(prob, 3)}
