# TB Synthetic Data Preparation and Risk Prediction Tool

A comprehensive project for generating synthetic tuberculosis (TB) datasets, training machine learning models, and deploying a web-based TB risk assessment tool. The system uses a custom Bayesian probabilistic model to predict TB risk based on common symptoms.

## Overview

This project combines synthetic data generation, model training, and web deployment to create an educational tool for TB risk assessment. It includes:

- **Synthetic Data Generation**: Parametric generation of TB datasets with pulmonary and systemic TB phenotypes
- **Model Tuning**: Optimization of data parameters to match predefined test cases
- **Bayesian Modeling**: Custom probabilistic model for risk prediction
- **Web Application**: FastAPI backend with HTML frontend for user interaction

## Features

- Parametric synthetic TB data generation with realistic symptom correlations
- Automated parameter tuning to match clinical test cases
- Custom Bayesian TB risk prediction model
- RESTful API for predictions
- User-friendly web interface for symptom input
- Risk categorization: Low, Moderate, High TB Risk, and Pulmonary Issue
- Probability scores for informed decision making

## Project Structure

```
tbsynthdataprep/
├── api.py                    # FastAPI backend server
├── predictor.py              # ML prediction logic and model loading
├── tb_model.py               # Bayesian TB model implementation
├── train.py                  # Model training script
├── tuner.py                  # Parameter tuning for synthetic data
├── data_generator.py         # Synthetic data generation and evaluation
├── index.html               # Frontend web interface
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/
│   ├── tb_training_data_tuned_v2.csv    # Tuned training dataset
│   └── old/                            # Previous versions
├── model/
│   └── tb_model.pkl                    # Trained model pickle
└── old_files/                          # Archived scripts
```

## Installation and Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Reachariramanan/tbsynthdataprep.git
   cd tbsynthdataprep
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Generation and Model Training

The project includes pre-generated data and a trained model. To regenerate:

1. **Generate Synthetic Data** (optional):
   ```bash
   python tuner.py
   ```
   This tunes parameters and generates `data/tb_training_data_tuned.csv`

2. **Train the Model**:
   ```bash
   python train.py
   ```
   This loads the data and saves the trained model to `model/tb_model.pkl`

### Running the Web Application

1. **Start the Backend**:
   ```bash
   uvicorn api:app --reload
   ```
   The API will be available at `http://localhost:8000`

2. **Open the Frontend**:
   Open `index.html` in any modern web browser

3. **Use the Application**:
   - Check symptoms that apply
   - Click "Check TB Risk"
   - View risk assessment and probability

## API Documentation

### Endpoints

- `GET /` - Health check
  - Response: `{"status": "TB backend running"}`

- `POST /predict` - TB risk prediction
  - Input: JSON object with symptom fields (0/1)
  - Output: `{"risk": "High TB Risk", "tb_probability": 0.85}`

### Input Format

```json
{
  "cough": 1,
  "cough_gt_2w": 0,
  "blood_in_sputum": 0,
  "fever": 1,
  "weight_loss": 0,
  "night_sweats": 1,
  "chest_pain": 0,
  "breathing_problem": 0,
  "fatigue": 1,
  "loss_of_appetite": 0,
  "contact_with_TB": 0
}
```

### Output Format

```json
{
  "risk": "Moderate TB Risk",
  "tb_probability": 0.65
}
```

## Model Details

### Bayesian TB Model

The system uses a custom Bayesian probabilistic model that:

- Learns conditional probabilities P(symptom|TB) and P(symptom|no TB) from training data
- Uses Laplace smoothing for robust probability estimation
- Computes log-odds ratios for prediction
- Applies rule-based logic for specific cases (e.g., pulmonary issues)

### Risk Thresholds

- **High TB Risk**: Probability ≥ 0.75
- **Moderate TB Risk**: Probability ≥ 0.40
- **Low TB Risk**: Probability < 0.40
- **Pulmonary Issue**: Breathing problem as only symptom

### Synthetic Data Generation

Data is generated using a parametric model with:

- **TB Prevalence**: Configurable (default 55%)
- **Phenotypes**: Pulmonary vs Systemic TB with different symptom patterns
- **Dependencies**: Cough-dependent symptoms (cough >2w, blood in sputum)
- **Parameter Tuning**: Automated optimization to match clinical test cases

## Test Cases

The model is validated against predefined test cases:

| Symptoms | Expected Risk |
|----------|---------------|
| cough, cough>2w, blood in sputum | High |
| cough, cough>2w | Low |
| cough, cough>2w, weight loss | High |
| cough, cough>2w, night sweats | High |
| cough, cough>2w, chest pain, night sweats | High |
| cough, cough>2w, fever | Moderate |
| cough, cough>2w, contact with TB | Moderate |
| cough, fever, night sweats, fatigue | Moderate |
| cough, weight loss, loss of appetite, fatigue | Moderate |
| weight loss, loss of appetite, fatigue | Low |

## Data Format

Training data is stored as CSV with binary features:

```csv
cough,cough_gt_2w,blood_in_sputum,breathing_problem,fever,night_sweats,fatigue,weight_loss,loss_of_appetite,chest_pain,contact_with_TB,TB
1,0,0,0,0,1,0,0,1,0,0,0
1,0,0,0,0,1,1,0,1,0,0,1
...
```

- All features: 0 (absent) or 1 (present)
- TB: 0 (no TB) or 1 (TB positive)

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- fastapi: Web framework
- uvicorn: ASGI server
- scikit-learn: Not used in current model but included

## Important Disclaimer

**This tool is for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. The predictions are based on simplified models and synthetic data. Always consult with qualified healthcare providers for any health concerns.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test against the predefined test cases
5. Submit a pull request

## License

This project is open-source. Please check the repository for license details.
