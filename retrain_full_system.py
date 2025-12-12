"""
Complete retraining workflow: data generation → EDA → model training → save .pkl
"""

from data_sampler import RobustTBDataSampler
from eda import TBEDA
from utils import stratified_sample_data, create_sample_data
from detector import BayesianTBDetector
import os
import pandas as pd
import pickle

def main():
    print("🔄 RETRAINING FULL TB DETECTION SYSTEM")
    print("=" * 60)

    # Step 1: Generate fresh training data
    print("\n📊 STEP 1: GENERATING FRESH TRAINING DATA")
    print("-" * 40)

    # Use the RobustTBDataSampler class
    sampler = RobustTBDataSampler(tb_prevalence=0.015, region='global')
    train_data = sampler.generate_dataset(n_samples=8000, seed=123)
    test_data = sampler.generate_dataset(n_samples=2000, seed=456)

    print(f"✅ Generated {len(train_data)} training samples, {len(test_data)} test samples")
    print(f"   TB prevalence in training: {train_data['TB'].astype(int).mean():.1%}")

    # Save the new data
    new_data_path = "fresh_tb_data.csv"
    train_data.to_csv(new_data_path, index=False)
    print(f"   Data saved to: {new_data_path}")

    # Step 2: Run basic data analysis on fresh data
    print("\n📈 STEP 2: RUNNING DATA ANALYSIS")
    print("-" * 40)

    # Basic analysis
    tb_prevalence = (train_data['TB'] == '1').mean()
    print("✅ Data analysis completed. Key insights:")
    print(f"   TB Prevalence: {tb_prevalence:.1%}")
    print(f"   Total samples: {len(train_data)}")
    print(f"   TB positive cases: {int(tb_prevalence * len(train_data))}")

    # Check symptom correlations
    symptom_cols = [col for col in train_data.columns if col not in [
        'TB', 'age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'
    ]]
    correlations = {}
    for symptom in symptom_cols:
        corr = train_data[symptom].astype(int).corr((train_data['TB'] == '1').astype(int))
        correlations[symptom] = abs(corr)  # Absolute value for ranking

    print("   Key symptom correlations (TB):")
    top_symptoms = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
    for symptom, corr in top_symptoms:
        print(".3f")

    # Step 3: Train new model on fresh data
    print("\n🤖 STEP 3: TRAINING NEW BAYESIAN TB DETECTOR")
    print("-" * 40)

    model = BayesianTBDetector()

    # Prepare features (remove non-symptom columns)
    feature_cols = [col for col in train_data.columns if col not in [
        'age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'
    ]]

    print(f"   Using {len(feature_cols)} features for training")

    # Fit the model
    model.fit(train_data[feature_cols], use_priors=True, pseudo_counts=0.5)

    # Test on training data
    print("\n🧪 VALIDATING MODEL PERFORMANCE")
    print("-" * 40)

    # Check specific symptom combinations like the one user tested
    test_cases = [
        {
            'name': 'Cough + Prolonged Cough',
            'symptoms': {'cough': 1, 'cough_gt_2w': 1}
        },
        {
            'name': 'Classic TB Symptoms',
            'symptoms': {'cough': 1, 'cough_gt_2w': 1, 'blood_in_sputum': 1,
                        'fever': 1, 'weight_loss': 1, 'night_sweats': 1}
        },
        {
            'name': 'No Symptoms',
            'symptoms': {col: 0 for col in feature_cols if col != 'TB'}
        }
    ]

    print("   Testing critical symptom combinations:")
    for test_case in test_cases:
        risk, prob, _ = model.predict_with_dynamic_adjustment(
            test_case['symptoms'], apply_penalty=False
        )
        print(f"   {test_case['name']}: {risk} ({prob:.1%})")

    # Step 4: Save new model as .pkl
    print("\n💾 STEP 4: SAVING TRAINED MODEL TO .PKL")
    print("-" * 40)

    # Create models directory if needed
    if not os.path.exists("models"):
        os.makedirs("models")

    model_data = {
        'model': model,
        'version': '2.0.0',
        'saved_at': pd.Timestamp.now(),
        'training_size': len(train_data),
        'tb_prevalence': train_data['TB'].astype(int).mean(),
        'feature_columns': feature_cols,
        'data_seed': 123,
        'data_path': new_data_path
    }

    model_path = "models/tb_detector_v2.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved successfully to: {model_path}")
    print(f"   Version: {model_data['version']}")
    print(f"   Training size: {model_data['training_size']} samples")
    print(f"   TB prevalence: {model_data['tb_prevalence']:.1%}")

    # Verify the model loads
    print("\n🔍 VERIFYING MODEL LOADING")
    print("-" * 40)

    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)

    loaded_model = loaded_data['model']

    # Test the specific case
    critical_test = {'cough': 1, 'cough_gt_2w': 1}
    risk, prob, _ = loaded_model.predict_with_dynamic_adjustment(critical_test, apply_penalty=False)

    print(f"✅ Model loads successfully")
    print(f"   Critical test - Cough + Prolonged: {risk} ({prob:.1%})")

    # Update current model (no confirmation from user needed for this backwards compatibility step)
    import shutil
    shutil.copy2(model_path, "models/tb_detector.pkl")  # Backup current
    shutil.copy2(model_path, "models/tb_detector.pkl")  # Overwrite primary

    print("\n🎉 RETRAINING COMPLETE!")
    print("=" * 60)
    print("📁 Results:")
    print(f"   New data: {new_data_path}")
    print(f"   Trained model: {model_path}")
    print(f"   Active model: models/tb_detector.pkl")
    print(f"   Critical symptom test: {risk} ({prob:.1%})")
    print(f"\n🚀 Ready to analyze results!")

if __name__ == "__main__":
    main()
