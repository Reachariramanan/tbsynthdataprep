"""
Simple script to train and save TB detection model as .pkl
Creates the models directory and saves the trained model.
"""

import os
import pandas as pd
from detector import BayesianTBDetector
from utils import create_sample_data, stratified_sample_data
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 TRAINING AND SAVING TB DETECTION MODEL")
    print("=" * 50)

    # Create training data
    print("📊 Generating training data...")
    df = create_sample_data(n=5000, tb_prevalence=0.015, include_demographics=True, seed=42)

    # Train/test split
    train_data, _ = stratified_sample_data(df, test_size=0.2, random_state=42)

    # Train model
    print("🤖 Training Bayesian TB detector...")
    model = BayesianTBDetector()
    features = train_data.drop(columns=['age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'], errors='ignore')
    model.fit(features, use_priors=True, pseudo_counts=0.5)

    # Create models directory if it doesn't exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created directory: {model_dir}/")

    # Save the trained model as .pkl
    model_path = os.path.join(model_dir, "tb_detector.pkl")
    model_data = {
        'model': model,
        'version': '1.0.0',
        'saved_at': pd.Timestamp.now(),
        'training_size': len(features),
        'tb_prevalence': (features['TB'] == '1').mean()
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved successfully to: {model_path}")
    print(f"   Version: {model_data['version']}")
    print(f"   Trained on: {model_data['training_size']} samples")
    print(f"   TB prevalence in training: {model_data['tb_prevalence']:.1%}")

    # Verify the model can be loaded
    print("\n🔍 Verifying model loading...")
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(loaded_data['model']).__name__}")
    print(f"   Saved time: {loaded_data['saved_at']}")

    # Test prediction
    print("\n🧪 Testing prediction on loaded model...")
    test_evidence = {
        'cough': 1,
        'cough_gt_2w': 1,
        'blood_in_sputum': 1,
        'fever': 1,
        'weight_loss': 1,
        'night_sweats': 1
    }

    risk, prob, base = loaded_data['model'].predict_with_dynamic_adjustment(test_evidence, apply_penalty=False)
    print(f"   Test prediction: {risk} (probability: {prob:.3f})")

    print("\n🎉 MODEL TRAINING AND SAVING COMPLETE!")
    print("📁 Location: models/tb_detector.pkl")
if __name__ == "__main__":
    main()
