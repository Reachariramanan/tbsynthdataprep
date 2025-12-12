"""
TB Detection Web Interface
A simple web UI for TB risk assessment using the trained model.
"""

from flask import Flask, render_template, request, flash, jsonify
import os
import pickle
from typing import Optional

app = Flask(__name__)
app.secret_key = 'tb_detection_system_secret_key'

# Define demographic questions
DEMOGRAPHIC_QUESTIONS = [
    {
        'id': 'age',
        'question': 'What is your age?',
        'type': 'number',
        'min': 1,
        'max': 120,
        'placeholder': 'Enter age in years'
    },
    {
        'id': 'gender',
        'question': 'What is your gender?',
        'type': 'select',
        'options': [
            {'value': 'M', 'label': 'Male'},
            {'value': 'F', 'label': 'Female'}
        ]
    },
    {
        'id': 'height',
        'question': 'What is your height?',
        'type': 'number',
        'min': 50,
        'max': 250,
        'step': 0.1,
        'placeholder': 'Height in cm',
        'description': 'Your height in centimeters'
    },
    {
        'id': 'weight',
        'question': 'What is your weight?',
        'type': 'number',
        'min': 20,
        'max': 300,
        'step': 0.1,
        'placeholder': 'Weight in kg',
        'description': 'Your weight in kilograms'
    }
]

# Define symptom questions with descriptions
SYMPTOM_QUESTIONS = [
    {
        'id': 'cough',
        'question': 'Do you have a cough?',
        'description': 'Any cough lasting more than a few days'
    },
    {
        'id': 'cough_gt_2w',
        'question': 'Has your cough lasted more than 2 weeks?',
        'description': 'Cough persisting for longer than 2 weeks'
    },
    {
        'id': 'blood_in_sputum',
        'question': 'Have you noticed blood in your sputum/phlegm?',
        'description': 'Coughing up blood or blood-streaked sputum'
    },
    {
        'id': 'fever',
        'question': 'Have you had a fever?',
        'description': 'Elevated body temperature > 37.5°C (99.5°F)'
    },
    {
        'id': 'low_grade_fever',
        'question': 'Have you had a persistent low-grade fever?',
        'description': 'Mild fever that persists over days or weeks'
    },
    {
        'id': 'weight_loss',
        'question': 'Have you lost weight unintentionally?',
        'description': 'Significant weight loss (>5% of body weight) without dieting'
    },
    {
        'id': 'night_sweats',
        'question': 'Do you experience night sweats?',
        'description': 'Excessive sweating during sleep'
    },
    {
        'id': 'chest_pain',
        'question': 'Do you have chest pain?',
        'description': 'Pain or discomfort in the chest area'
    },
    {
        'id': 'breathing_problem',
        'question': 'Do you have shortness of breath?',
        'description': 'Difficulty breathing or shortness of breath'
    },
    {
        'id': 'fatigue',
        'question': 'Do you feel unusually tired or fatigued?',
        'description': 'Extreme tiredness or lack of energy'
    },
    {
        'id': 'loss_of_appetite',
        'question': 'Have you lost your appetite?',
        'description': 'Reduced desire to eat'
    },
    {
        'id': 'contact_with_TB',
        'question': 'Have you had close contact with someone who has TB?',
        'description': 'Close contact with confirmed TB patient in the last 2 years'
    }
]

def load_model() -> Optional[object]:
    """Load the trained TB detection model."""
    model_path = "models/tb_detector.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data['model']
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

# Load model globally
MODEL = load_model()

@app.route('/')
def home():
    """Home page with symptom questionnaire."""
    return render_template('index.html',
                         demographic_questions=DEMOGRAPHIC_QUESTIONS,
                         symptom_questions=SYMPTOM_QUESTIONS)

@app.route('/assess', methods=['POST'])
def assess_risk():
    """Process symptom assessment and return risk evaluation."""
    if not MODEL:
        error_msg = "TB Detection Model not available. Please ensure the model is properly trained and saved."
        return render_template('error.html', error=error_msg)

    # Collect demographics from form
    demographics = {}
    try:
        # Get demographic info
        demographics['age'] = int(request.form.get('age', 30))
        demographics['gender'] = request.form.get('gender', 'M')
        demographics['height'] = float(request.form.get('height', 170))
        demographics['weight'] = float(request.form.get('weight', 70))

        # Calculate BMI
        height_m = demographics['height'] / 100  # Convert cm to meters
        demographics['bmi_value'] = round(demographics['weight'] / (height_m ** 2), 1)

        # Classify BMI category
        if demographics['bmi_value'] < 18.5:
            demographics['bmi_category'] = 'underweight'
        elif demographics['bmi_value'] < 25:
            demographics['bmi_category'] = 'normal'
        elif demographics['bmi_value'] < 30:
            demographics['bmi_category'] = 'overweight'
        else:
            demographics['bmi_category'] = 'obese'

        # Collect symptoms from form
        symptoms = {}
        for question in SYMPTOM_QUESTIONS:
            symptom_id = question['id']
            value = request.form.get(symptom_id, '0')
            symptoms[symptom_id] = int(value)

        # Get assessment (Note: Current model doesn't use demographics for prediction,
        # but we collect them for future enhancement and display)
        risk_category, probability, base_prob = MODEL.predict_with_dynamic_adjustment(
            symptoms, apply_penalty=False
        )

        # Get symptom importance
        symptom_importance = MODEL.get_symptom_importance(symptoms)
        key_symptoms = sorted(symptom_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Prepare result data
        result_data = {
            'risk_category': risk_category,
            'probability': round(probability * 100, 1),
            'symptoms_present': sum(symptoms.values()),
            'key_symptoms': key_symptoms,
            'clinical_interpretation': get_clinical_interpretation(risk_category, probability, symptoms),
            'demographics': demographics
        }

        return render_template('result.html', result=result_data, symptoms=symptoms)

    except Exception as e:
        error_msg = f"Error during risk assessment: {str(e)}"
        return render_template('error.html', error=error_msg)

@app.route('/api/assess', methods=['POST'])
def api_assess_risk():
    """API endpoint for risk assessment (JSON response)."""
    if not MODEL:
        return jsonify({'error': 'Model not available'}), 500

    try:
        data = request.get_json()
        symptoms = data.get('symptoms', {})

        # Convert symptoms to integers
        for key in symptoms:
            symptoms[key] = int(symptoms[key])

        risk_category, probability, base_prob = MODEL.predict_with_dynamic_adjustment(
            symptoms, apply_penalty=False
        )

        return jsonify({
            'risk_category': risk_category,
            'probability': round(probability * 100, 1),
            'symptoms_present': sum(symptoms.values()),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

def get_clinical_interpretation(risk_category, probability, symptoms):
    """Provide clinical interpretation based on risk assessment."""
    if risk_category == "High TB Risk":
        return """⚠️ HIGH RISK INDICATION: Strong clinical evidence suggesting tuberculosis. Immediate evaluation by a healthcare professional is highly recommended. Should undergo diagnostic testing including chest X-ray, sputum examination, and TB test."""

    elif risk_category == "Moderate TB Risk":
        return """🟡 MODERATE RISK: Several symptom combinations suggest possible tuberculosis. Please consult a healthcare provider for further evaluation, which may include TB screening tests."""

    elif risk_category == "Low TB Risk":
        return """🟢 LOW RISK: Some symptoms present but combination does not strongly suggest tuberculosis. Consider monitoring symptoms and consult healthcare provider if symptoms worsen or persist."""

    elif risk_category == "Pulmonary Issue":
        return """🔵 OTHER PULMONARY CONCERN: Symptoms suggest a respiratory condition. While TB risk appears low, please consult a healthcare provider for proper diagnosis of the underlying cause."""

    else:  # Healthy
        return """🟢 LOW CONCERN: Symptom profile indicates minimal risk for tuberculosis. Continue healthy lifestyle practices and regular medical check-ups."""

@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html')

@app.route('/medical-disclaimer')
def medical_disclaimer():
    """Medical disclaimer page."""
    return render_template('disclaimer.html')

if __name__ == '__main__':
    print("🚀 Starting TB Detection Web Interface...")
    print("📊 Available endpoints:")
    print("   Home: http://localhost:5000/")
    print("   API: http://localhost:5000/api/assess")
    print("   About: http://localhost:5000/about")
    print("   Disclaimer: http://localhost:5000/medical-disclaimer")

    if MODEL:
        print("✅ TB Detection Model loaded successfully")
    else:
        print("⚠️ No TB Detection Model found - please run train_save_model.py first")

    app.run(debug=True, host='0.0.0.0', port=5000)
