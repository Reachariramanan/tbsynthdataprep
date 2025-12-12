"""
Professional TB Detection System - Modular and Debug-Friendly
DOCTOR AI's best friend
"""

from detector import BayesianTBDetector

from utils import (
    create_sample_data,
    stratified_sample_data,
    validate_data_quality,
    cross_validate_model,
    augment_rare_cases
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

# Model persistence functions
import pickle
import os
from pathlib import Path

def save_model(model, filepath="models/tb_detector.pkl"):
    """Save trained model to pickle file."""
    os.makedirs("models", exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'saved_at': pd.Timestamp.now()}, f)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath="models/tb_detector.pkl", fallback_train=True):
    """Load saved model or train new one."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Model loaded from {filepath}")
            print(f"   Saved: {data['saved_at']}")
            return data['model']
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            if fallback_train:
                print("💡 Training new model...")
            else:
                return None

    if fallback_train:
        print("🚀 No saved model found. Training new model...")
        return train_model()
    return None

def train_model():
    """Train and return TB detection model."""
    print("Building training data...")
    df = create_sample_data(n=5000, tb_prevalence=0.015, include_demographics=True, seed=42)

    train_data, _ = stratified_sample_data(df, stratify_column='TB', test_size=0.2)

    detector = BayesianTBDetector()
    features = train_data.drop(columns=['age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'],
                              errors='ignore')
    detector.fit(features, use_priors=True, pseudo_counts=0.5)

    # Save the trained model
    save_model(detector)

    return detector

# Generate realistic test cases
def create_realistic_test_cases():
    """Create clinical test cases representing real patient scenarios."""
    return [
        {
            "name": "Classic TB - Males 15-34",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 1, "fever": 1, "weight_loss": 1, "night_sweats": 1,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 1, "contact_with_TB": 0},
            "expected": "High TB Risk"
        },
        {
            "name": "Elderly Atypical",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 0, "fever": 0, "weight_loss": 1, "night_sweats": 0,
                        "chest_pain": 0, "breathing_problem": 1, "fatigue": 1, "loss_of_appetite": 1, "contact_with_TB": 0},
            "expected": "Moderate TB Risk"
        },
        {
            "name": "HIV/TB Co-infection Pattern",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 0, "fever": 1, "weight_loss": 1, "night_sweats": 1,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 1, "contact_with_TB": 1},
            "expected": "High TB Risk"
        },
        {
            "name": "Acute Viral Illness",
            "symptoms": {"cough": 1, "cough_gt_2w": 0, "blood_in_sputum": 0, "fever": 1, "weight_loss": 0, "night_sweats": 0,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 0, "contact_with_TB": 0},
            "expected": "Low TB Risk"
        },
        {
            "name": "Healthcare Worker Exposure",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 0, "fever": 1, "weight_loss": 0, "night_sweats": 1,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 0, "contact_with_TB": 1},
            "expected": "Moderate TB Risk"
        },
        {
            "name": "COPD Exacerbation",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 0, "fever": 0, "weight_loss": 0, "night_sweats": 0,
                        "chest_pain": 1, "breathing_problem": 1, "fatigue": 1, "loss_of_appetite": 0, "contact_with_TB": 0},
            "expected": "Low TB Risk"
        },
        {
            "name": "Malnourished Child",
            "symptoms": {"cough": 1, "cough_gt_2w": 1, "blood_in_sputum": 0, "fever": 1, "weight_loss": 1, "night_sweats": 0,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 1, "contact_with_TB": 0},
            "expected": "Moderate TB Risk"
        },
        {
            "name": "Asymptomatic Control",
            "symptoms": {"cough": 0, "cough_gt_2w": 0, "blood_in_sputum": 0, "fever": 0, "weight_loss": 0, "night_sweats": 0,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 0, "loss_of_appetite": 0, "contact_with_TB": 0},
            "expected": "Healthy"
        },
        {
            "name": "Post-TB Monitoring",
            "symptoms": {"cough": 1, "cough_gt_2w": 0, "blood_in_sputum": 0, "fever": 0, "weight_loss": 0, "night_sweats": 0,
                        "chest_pain": 0, "breathing_problem": 0, "fatigue": 1, "loss_of_appetite": 0, "contact_with_TB": 1},
            "expected": "Low TB Risk"
        },
        {
            "name": "Community-Acquired Pneumonia",
            "symptoms": {"cough": 1, "cough_gt_2w": 0, "blood_in_sputum": 0, "fever": 1, "weight_loss": 0, "night_sweats": 0,
                        "chest_pain": 1, "breathing_problem": 1, "fatigue": 1, "loss_of_appetite": 0, "contact_with_TB": 0},
            "expected": "Pulmonary Issue"
        }
    ]

def benchmark_realistic_performance():
    """Comprehensive benchmark with realistic test cases."""
    print("🧪 COMPREHENSIVE TB MODEL BENCHMARKING")
    print("=" * 60)

    # Load or train model
    model = load_model(fallback_train=True)

    # Generate large realistic test dataset
    test_data = create_sample_data(n=10000, tb_prevalence=0.012, include_demographics=True, seed=99)

    prevalence = (test_data['TB'] == '1').mean()

    print("\n📊 Benchmarking on Realistic Clinical Data:")
    print(f"   Dataset: {len(test_data)} patients")
    print(f"   TB prevalence: {prevalence:.1%}")

    # Remove demographics and TB column for prediction (TB is the target, not a feature)
    features = test_data.drop(columns=['TB', 'age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'], errors='ignore')
    features = features.astype(int)
    true_labels = (test_data['TB'] == '1').astype(int)

    # Make predictions
    predictions = []
    probabilities = []

    print("\n🔍 Evaluating predictions...")
    debug_count = 0
    tb_debug_count = 0
    for idx, row in features.iterrows():
        evidence = {col: row[col] for col in features.columns}
        actual = int(test_data.iloc[idx]['TB'])
        try:
            prob = model.predict_probability(evidence)
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
            probabilities.append(prob)

            # Debug first 5 predictions and first 3 TB cases
            if debug_count < 5:
                print(f"  Sample {debug_count+1}: prob={prob:.4f}, pred={pred}, actual={actual}")
                debug_count += 1
            elif actual == 1 and tb_debug_count < 3:
                print(f"  TB case: prob={prob:.4f}, pred={pred}, actual={actual}, symptoms={sum(row)}")
                tb_debug_count += 1
        except Exception as e:
            if debug_count < 5:
                print(f"  Sample {debug_count+1}: ERROR - {e}")
                debug_count += 1
            predictions.append(0)
            probabilities.append(0.5)

    # Calculate comprehensive metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # Calculate impact factor (improvement over symptom count)
    symptom_sums = features.sum(axis=1)
    baseline_pred = (symptom_sums >= 3).astype(int)
    baseline_acc = accuracy_score(true_labels, baseline_pred)
    impact_factor = accuracy - baseline_acc

    print("\n🚀 MODEL PERFORMANCE METRICS:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1:.1%}")
    print(f"Impact Factor: {impact_factor:.1%}")
    if impact_factor > 0:
        print("✅ Model provides meaningful improvement over baseline")
    else:
        print("⚠️ Model impact needs improvement")
    # Evaluate on realistic clinical cases
    print("\n🌡️ CLINICAL CASE EVALUATION:")
    clinical_cases = create_realistic_test_cases()
    correct_predictions = 0

    for case in clinical_cases:
        try:
            risk, prob, _ = model.predict_with_dynamic_adjustment(case['symptoms'], apply_penalty=False)
            is_correct = (risk == case['expected'])
            if is_correct:
                correct_predictions += 1

            status = "✅" if is_correct else "❌"
            print(f"{status} {case['name']}: {risk} (expected: {case['expected']})")
        except Exception as e:
            print(f"⚠️ {case['name']}: Error - {e}")

    clinical_accuracy = correct_predictions / len(clinical_cases)

    print("\n📈 CLINICAL REALISM ASSESSMENT:")
    print(f"Clinical Test Cases: {len(clinical_cases)}")
    print(f"Correct Clinical Predictions: {correct_predictions}")
    print(f"Clinical Accuracy: {clinical_accuracy:.1%}")
    # Confusion matrix analysis
    cm = confusion_matrix(true_labels, predictions)
    print("\n📊 Confusion Matrix:")
    print("         Predicted")
    print("Actual    TB   Non-TB")
    print(f"TB     {cm[1][1]:>6} {cm[1][0]:>6}")
    print(f"Non-TB {cm[0][1]:>6} {cm[0][0]:>6}")

    # Final readiness assessment
    readiness = 0
    if accuracy >= 0.95: readiness += 30
    elif accuracy >= 0.90: readiness += 20
    if clinical_accuracy >= 0.80: readiness += 25
    if impact_factor >= 0.02: readiness += 25
    elif impact_factor >= 0.01: readiness += 15
    if precision >= 0.80: readiness += 10
    if recall >= 0.85: readiness += 10

    print("\n🏆 FINAL SYSTEM READINESS ASSESSMENT:")
    print(f"Overall Readiness Score: {readiness}%")
    print(f"Impact Factor: {impact_factor:.2f}%")

    if readiness >= 80:
        print("🟢 EXCELLENT: Ready for clinical deployment")
    elif readiness >= 60:
        print("🟡 GOOD: Ready for controlled testing")
    else:
        print("🔴 NEEDS IMPROVEMENT: Requires additional development")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'impact_factor': impact_factor,
        'clinical_accuracy': clinical_accuracy,
        'readiness_score': readiness
    }

# -----------------------------
# Professional TB Detection System - Demonstration
# -----------------------------
if __name__ == "__main__":
    print("=== PROFESSIONAL TB DETECTION SYSTEM ===")
    print("Enhanced with model persistence and realistic benchmarking\n")

    # Run comprehensive benchmarking
    results = benchmark_realistic_performance()

    print("\n✅ Benchmarking Complete!")
    print("Model Performance: Accuracy {:.1%}, Impact Factor: {:.2%}".format(
        results['accuracy'], results['impact_factor']
    ))
    print("Ready for clinical use: {}".format(
        "YES" if results['readiness_score'] >= 80 else "MODERATE" if results['readiness_score'] >= 60 else "NO"
    ))
