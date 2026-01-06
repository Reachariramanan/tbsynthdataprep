# import pickle
# from data_generator import generate_data
# from tb_model import BayesianTBModel
# import pandas as pd

# df = generate_data()
# model = BayesianTBModel()
# model.fit(df)

# with open("model/tb_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# df.to_csv("data/synthetic_tb_data_with_pred.csv", index=False)
# print("Model trained & saved")

import pickle
import pandas as pd
from tb_model import BayesianTBModel

# 1. Path to your existing dataset
# DATA_PATH = "data/tb_training_data.csv"   
DATA_PATH = "data/tb_training_data_tuned_v2.csv"   

# 2. Load data from CSV
df = pd.read_csv(DATA_PATH)

# (Optional) Basic validation
if df.empty:
    raise ValueError("Training data is empty")

# 3. Initialize and train the model
model = BayesianTBModel()
model.fit(df)

# 4. Save trained model
with open("model/tb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained using CSV data & saved successfully")
