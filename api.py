from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import load_model, predict

app = FastAPI(title="TB Risk Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
model = load_model()

class SymptomInput(BaseModel):
    cough: int = 0
    cough_gt_2w: int = 0
    blood_in_sputum: int = 0
    fever: int = 0
    weight_loss: int = 0
    night_sweats: int = 0
    chest_pain: int = 0
    breathing_problem: int = 0
    fatigue: int = 0
    loss_of_appetite: int = 0
    contact_with_TB: int = 0

@app.get("/")
def health_check():
    return {"status": "TB backend running"}

@app.post("/predict")
def predict_tb(symptoms: dict):
    return predict(model, symptoms)
