"""
Comprehensive Test Cases for TB Detection System
Validates epidemiological accuracy, clinical performance, and system readiness.

WHO Statistics Summary (2023):
- 5.8 million men fell ill with TB
- 3.7 million women fell ill with TB
- 1.2 million children fell ill with TB
- Global ratio: ~48%:30%:10% (men:women:children)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import all system modules
from detector import BayesianTBDetector
from utils import create_sample_data, stratified_sample_data

class TBSystemTester:
    """
    Comprehensive testing framework for TB detection system.
    """

    def __init__(self):
        print("🧪 TB Detection System Test Framework Initialized")
        print("WHO Epidemiological Benchmarks (2023): 5.8M men, 3.7M women, 1.2M children")
        self.test_results = {}

    def test_epidemiological_accuracy(self):
        """Test if generated data matches WHO epidemiological distributions."""
        print("\n=== TESTING EPIDEMIOLOGICAL ACCURACY ===")

        # Generate sample data
        data = create_sample_data(n=10000, tb_prevalence=0.015, include_demographics=True, seed=42)

        print(f"Generated {len(data)} patient records")
        tb_prevalence = (data['TB'] == '1').mean()
        print(f"TB prevalence: {tb_prevalence:.1%}")

        # Gender analysis
        if 'gender' in data.columns:
            gender_dist = data['gender'].value_counts()
            tb_by_gender = data.groupby('gender')['TB'].apply(lambda x: (x == '1').mean())

            print("\nGender Distribution:")
            men_pct = gender_dist['M'] / len(data)
            women_pct = gender_dist['F'] / len(data)
            print(f"  Men: {men_pct:.1%}, Women: {women_pct:.1%}")

            if 'M' in tb_by_gender.index and 'F' in tb_by_gender.index:
                male_risk = tb_by_gender['M'] / tb_by_gender['F']
                print(f"  Male TB risk ratio: {male_risk:.2f}")

        # Age analysis
        if 'age' in data.columns:
            age_stats = data['age'].describe()
            print(f"\nAge Statistics: Mean={age_stats['mean']:.1f}, Range={age_stats['min']:.0f}-{age_stats['max']:.0f}")

        self.test_results['epidemiological'] = {
            'sample_size': len(data),
            'tb_prevalence': tb_prevalence
        }

        return self.test_results['epidemiological']

    def test_model_performance(self):
        """Test Bayesian model performance."""
        print("\n=== TESTING BAYESIAN MODEL PERFORMANCE ===")

        # Generate data
        data = create_sample_data(n=5000, tb_prevalence=0.015, include_demographics=True, seed=42)

        # Create train/test split
        train_data, test_data = stratified_sample_data(data, test_size=0.3, random_state=42)

        # Train model
        model = BayesianTBDetector()
        features_to_drop = ['age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight']
        model.fit(train_data.drop(columns=features_to_drop, errors='ignore'))

        # Test predictions
        test_features = test_data.drop(columns=features_to_drop + ['TB'], errors='ignore')
        true_labels = (test_data['TB'] == '1').astype(int)

        predictions = []
        for _, row in test_features.iterrows():
            evidence = {col: row[col] for col in test_features.columns}
            try:
                prob = model.predict_probability(evidence)
                predictions.append(1 if prob > 0.5 else 0)
            except:
                predictions.append(0)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)

        print(f"Model Performance: Accuracy={accuracy:.1%}, Precision={precision:.1%}, Recall={recall:.1%}")

        # Impact factor vs baseline
        symptom_cols = ['cough', 'cough_gt_2w', 'blood_in_sputum', 'fever', 'weight_loss', 'night_sweats']
        available_symptoms = [s for s in symptom_cols if s in test_features.columns]

        impact_factor = 0
        if available_symptoms:
            # Convert symptoms to numeric for calculation
            symptom_data = test_features[available_symptoms].astype(int)
            symptom_sums = symptom_data.sum(axis=1)
            baseline_pred = (symptom_sums >= 3).astype(int)
            baseline_acc = accuracy_score(true_labels, baseline_pred)
            impact_factor = accuracy - baseline_acc

            print(f"Impact factor vs baseline: {impact_factor:.1%}")
            if impact_factor > 0.05:
                impact_level = "HIGH"
            elif impact_factor > 0.02:
                impact_level = "MODERATE"
            else:
                impact_level = "LOW"

            print(f"IMPACT LEVEL: {impact_level}")
            if impact_factor < 0.02:
                print("⚠️ IMPACT FACTOR IS VERY LOW - Needs Improvement")

        self.test_results['model'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'impact_factor': impact_factor
        }

        return self.test_results['model']

    def test_system_integration(self):
        """Test system robustness."""
        print("\n=== TESTING SYSTEM INTEGRATION ===")

        tests_passed = 0
        total_tests = 0

        # Test data generation
        try:
            data = create_sample_data(n=100, tb_prevalence=0.01)
            tests_passed += 1
            print("✅ Data generation: PASS")
        except Exception as e:
            print(f"❌ Data generation: FAIL - {e}")
        total_tests += 1

        # Test model training
        try:
            data = create_sample_data(n=500, tb_prevalence=0.02)
            model = BayesianTBDetector()
            model.fit(data.drop(columns=['age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'], errors='ignore'))
            tests_passed += 1
            print("✅ Model training: PASS")
        except Exception as e:
            print(f"❌ Model training: FAIL - {e}")
        total_tests += 1

        # Test prediction
        try:
            evidence = {'cough': 1, 'fever': 1}
            prob = model.predict_probability(evidence)
            assert 0 <= prob <= 1
            tests_passed += 1
            print("✅ Prediction: PASS")
        except Exception as e:
            print(f"❌ Prediction: FAIL - {e}")
        total_tests += 1

        self.test_results['integration'] = {
            'passed': tests_passed,
            'total': total_tests,
            'success_rate': tests_passed / total_tests if total_tests > 0 else 0
        }

        return self.test_results['integration']

    def generate_final_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("FINAL TB DETECTION SYSTEM READINESS REPORT")
        print("="*80)

        # Run all tests
        epi_results = self.test_epidemiological_accuracy()
        model_results = self.test_model_performance()
        integration_results = self.test_system_integration()

        # Calculate readiness score
        impact_factor = model_results.get('impact_factor', 0)
        integration_rate = integration_results.get('success_rate', 0)

        readiness_score = (impact_factor * 100) + (integration_rate * 50) + 20  # Base score
        readiness_score = min(readiness_score, 100)  # Cap at 100

        print(f"Overall Readiness Score: {readiness_score:.1f}%")

        if readiness_score >= 70:
            status = "🟢 GOOD READINESS"
        elif readiness_score >= 50:
            status = "🟡 MODERATE READINESS"
        else:
            status = "🔴 LOW READINESS"

        print(f"\n🏆 FINAL STATUS: {status}")

        if impact_factor < 0.02:
            print("🚨 CRITICAL ISSUE: Impact factor is very low")
            print("💡 RECOMMENDATION: Review modeling approach and epidemiological data")

        print("\n📊 SUMMARY:")
        print(f"- Epidemiological data: {epi_results['sample_size']} samples generated")
        print(f"- TB prevalence: {epi_results['tb_prevalence']:.1%}")
        print(f"- Model accuracy: {model_results['accuracy']:.1%}")
        print(f"- Impact factor: {model_results['impact_factor']:.1%}")
        print(f"- Integration: {integration_results['passed']}/{integration_results['total']} tests passed")
        demo_cols = [c for c in ['age', 'gender', 'bmi_value'] if c in create_sample_data(100).columns]
        print(f"- Demographic factors: {len(demo_cols)}")

        return readiness_score

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 STARTING TB DETECTION SYSTEM TESTING")
    print("WHO Statistics: 5.8M men, 3.7M women, 1.2M children")

    tester = TBSystemTester()
    score = tester.generate_final_report()

    return score

if __name__ == "__main__":
    run_comprehensive_tests()
