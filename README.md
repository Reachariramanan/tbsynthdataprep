# TB Risk Prediction Tool

A simple web application for assessing tuberculosis (TB) risk based on symptoms.

## Features

- User-friendly web interface for symptom input
- Real-time TB risk assessment using machine learning
- Risk categories: Low, Moderate, High, and Pulmonary Issue
- Probability scores for informed decision making

## Setup and Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Backend Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI backend server:
   ```bash
   uvicorn api:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend

The frontend is a single HTML file that can be opened directly in any modern web browser.

## Usage

1. Start the backend server as described above
2. Open `index.html` in your web browser
3. Check the symptoms that apply to you
4. Click "Check TB Risk" to get your assessment

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Make a TB risk prediction

### Prediction Input

The API expects a JSON object with the following symptom fields (0 for absent, 1 for present):

```json
{
  "cough": 0,
  "cough_gt_2w": 0,
  "blood_in_sputum": 0,
  "fever": 0,
  "weight_loss": 0,
  "night_sweats": 0,
  "chest_pain": 0,
  "breathing_problem": 0,
  "fatigue": 0,
  "loss_of_appetite": 0,
  "contact_with_TB": 0
}
```

### Prediction Output

```json
{
  "risk": "High TB Risk",
  "tb_probability": 0.85
}
```

## Important Disclaimer

This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.

## Project Structure

```
TB_new/
├── api.py              # FastAPI backend
├── predictor.py        # ML prediction logic
├── index.html          # Frontend interface
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Training data
├── model/             # Trained model
└── *.py               # Other Python files
