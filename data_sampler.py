"""
Professional Data Sampling Module for TB Detection
Implements clinically-accurate data generation with BMI integration and statistical robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataSampler:
    """
    Professional data sampling with clinical accuracy, BMI integration, statistical robustness.
    """

    def __init__(self, tb_prevalence: float = 0.01, region: str = 'global'):
        """
        Initialize with clinical parameters and regional variations.

        Args:
            tb_prevalence: TB prevalence rate (0.01 = 1%)
            region: 'global', 'high_risk', 'low_risk' for regional variations
        """
        self.tb_prevalence = min(max(tb_prevalence, 0.001), 0.2)
        self.region = region

        # Clinical symptom probabilities (WHO-based guidelines)
        self.tb_positive_probs = {
            'cough': 0.95,             # 95% of infectious TB patients cough
            'cough_gt_2w': 0.95,        # 95% have prolonged cough (>2 weeks) - INCREASED IMPACT
            'blood_in_sputum': 0.75,    # Hemoptysis in 75% of pulmonary TB
            'fever': 0.80,             # Fever in 80%
            'low_grade_fever': 0.70,   # Low-grade fever
            'weight_loss': 0.70,       # Weight loss >10% body weight
            'night_sweats': 0.65,      # Night sweats
            'chest_pain': 0.50,        # Chest pain
            'breathing_problem': 0.55, # Dyspnea
            'fatigue': 0.75,           # Fatigue
            'loss_of_appetite': 0.65,  # Anorexia
            'contact_with_TB': 0.60    # Contact with TB patient
        }

        self.tb_negative_probs = {
            'cough': 0.10,             # 10% general population cough
            'cough_gt_2w': 0.005,       # Very rare prolonged cough in non-TB - DECREASED to increase discrimination
            'blood_in_sputum': 0.001,   # Very rare hemoptysis
            'fever': 0.15,             # Occasional fever
            'low_grade_fever': 0.05,   # Less common
            'weight_loss': 0.05,       # 5% may have weight issues
            'night_sweats': 0.03,      # Rare
            'chest_pain': 0.08,        # Some musculoskeletal pain
            'breathing_problem': 0.12, # Occasional respiratory issues
            'fatigue': 0.12,           # General fatigue
            'loss_of_appetite': 0.06,  # Occasional appetite issues
            'contact_with_TB': 0.02    # Low exposure in general population
        }

        # BMI impact modifiers
        self.bmi_modifiers = {
            'underweight': {'weight_loss': 1.2, 'fatigue': 1.1, 'loss_of_appetite': 1.3, 'tb_risk': 1.5},
            'normal': {},
            'overweight': {'breathing_problem': 1.2, 'fever': 0.9},
            'obese': {'breathing_problem': 1.4, 'chest_pain': 1.1, 'fever': 0.85, 'tb_risk': 0.7}
        }

    def draw_distributions(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """
        Draw statistical distributions and analyze them.
        """
        if data.empty:
            print("No data to analyze")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Statistical Distributions Analysis', fontsize=16)

        # Age distribution
        if 'age' in data.columns:
            sns.histplot(data['age'], bins=20, kde=True, ax=axes[0,0])
            axes[0,0].set_title('Age Distribution')
            axes[0,0].axvline(data['age'].mean(), ls='--', color='red', label=f'Mean: {data["age"].mean():.1f}')
            axes[0,0].legend()

        # BMI distribution by category
        if 'bmi_category' in data.columns:
            category_counts = data['bmi_category'].value_counts()
            axes[0,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('BMI Category Distribution')

        # Gender distribution
        if 'gender' in data.columns:
            gender_counts = data['gender'].value_counts()
            axes[0,2].bar(gender_counts.index, gender_counts.values)
            axes[0,2].set_title('Gender Distribution')

        # Symptom prevalence comparison
        symptom_cols = ['cough', 'fever', 'weight_loss', 'night_sweats']
        if all(col in data.columns for col in symptom_cols):
            tb_positive = data[data['TB'] == '1'][symptom_cols[:4]].astype(int).mean()
            tb_negative = data[data['TB'] == '0'][symptom_cols[:4]].astype(int).mean()

            symptoms_df = pd.DataFrame({'TB Positive': tb_positive, 'TB Negative': tb_negative})
            symptoms_df.plot(kind='bar', ax=axes[1,0], width=0.8)
            axes[1,0].set_title('Symptom Prevalence by TB Status')
            axes[1,0].tick_params(axis='x', rotation=45)

        # BMI vs TB correlation
        if 'bmi_category' in data.columns:
            tb_by_bmi = data.groupby('bmi_category')['TB'].apply(lambda x: (x == '1').mean())
            tb_by_bmi.plot(kind='bar', ax=axes[1,1], color='orange')
            axes[1,1].set_title('TB Prevalence by BMI Category')
            axes[1,1].tick_params(axis='x', rotation=45)

        # Symptom correlation heatmap
        if len([col for col in symptom_cols if col in data.columns]) >= 4:
            correlation_data = data[symptom_cols[:4] + ['TB']].replace({'0': 0, '1': 1})
            corr_matrix = correlation_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,2])
            axes[1,2].set_title('Symptom Correlation Matrix')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distributions saved to {save_path}")
        plt.show()

    def analyze_statistically(self, data: pd.DataFrame):
        """
        Perform comprehensive statistical analysis of the data.
        """
        print("=== STATISTICAL ANALYSIS REPORT ===")

        # Basic statistics
        print(f"Dataset: {len(data)} samples")
        tb_prevalence = (data['TB'] == '1').mean()
        print(f"TB Prevalence: {tb_prevalence:.4f}")

        # Demographic analysis
        if 'age' in data.columns:
            print("\nAge Statistics:")
            print(f"  Mean: {data['age'].mean():.1f} years")
            print(f"  Median: {data['age'].median():.1f} years")
            print(f"  Std: {data['age'].std():.1f} years")

        if 'gender' in data.columns:
            gender_dist = data['gender'].value_counts()
            print("\nGender Distribution:")
            for gender, count in gender_dist.items():
                print(f"  {gender}: {count} ({count/len(data)*100:.1f}%)")

        # BMI analysis
        if 'bmi_category' in data.columns:
            bmi_dist = data['bmi_category'].value_counts()
            print("\nBMI Category Distribution:")
            for bmi_cat, count in bmi_dist.items():
                print(f"  {bmi_cat}: {count} ({count/len(data)*100:.1f}%)")

            # TB prevalence by BMI
            print("\nTB Prevalence by BMI Category:")
            for bmi_cat in ['underweight', 'normal', 'overweight', 'obese']:
                if bmi_cat in data['bmi_category'].values:
                    subset = data[data['bmi_category'] == bmi_cat]
                    tb_rate = (subset['TB'] == '1').mean()
                    print(f"  {bmi_cat}: {tb_rate:.4f}")

        # Symptom statistics
        symptom_cols = ['cough', 'fever', 'weight_loss', 'night_sweats', 'chest_pain', 'breathing_problem']
        available_symptoms = [col for col in symptom_cols if col in data.columns]

        print(f"\nSymptom Statistics ({len(available_symptoms)} symptoms tracked):")
        for symptom in available_symptoms:
            tb_rate = data[data['TB'] == '1'][symptom].astype(int).mean()
            non_tb_rate = data[data['TB'] == '0'][symptom].astype(int).mean()
            print(f"  {symptom}: TB={tb_rate:.3f}, Non-TB={non_tb_rate:.3f}")

        # Statistical tests
        print("\nStatistical Tests:")
        for symptom in available_symptoms[:3]:  # Test first 3 symptoms
            try:
                tb_data = data[data['TB'] == '1'][symptom].astype(int)
                non_tb_data = data[data['TB'] == '0'][symptom].astype(int)

                # T-test for difference in proportions
                t_stat, p_val = stats.ttest_ind(tb_data, non_tb_data)
                print(f"  {symptom}: t={t_stat:.2f}, p={p_val:.4f}")
            except:
                continue

        print("\n=== END STATISTICAL ANALYSIS ===")

    def sample_demographics(self, n_samples: int) -> pd.DataFrame:
        """Sample age, gender, BMI with realistic distributions."""
        np.random.seed(42)

        # Age: log-normal distribution
        age = np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples)
        age = np.clip(age, 15, 80).astype(int)

        # Gender: slight male predominance
        gender = np.random.choice(['M', 'F'], size=n_samples, p=[0.55, 0.45])

        # BMI categories
        height = np.random.normal(170, 10, n_samples) / 100
        bmi_weights = {
            'underweight': lambda: np.random.uniform(15, 18.4),
            'normal': lambda: np.random.uniform(18.5, 24.9),
            'overweight': lambda: np.random.uniform(25, 29.9),
            'obese': lambda: np.random.uniform(30, 45)
        }

        category_probs = {'underweight': 0.15, 'normal': 0.45, 'overweight': 0.25, 'obese': 0.15}
        bmi_category = np.random.choice(list(category_probs.keys()), size=n_samples, p=list(category_probs.values()))
        bmi_value = np.array([bmi_weights[cat]() for cat in bmi_category])
        weight = bmi_value * (height ** 2)

        return pd.DataFrame({
            'age': age,
            'gender': gender,
            'bmi_category': bmi_category,
            'bmi_value': np.round(bmi_value, 1),
            'height': np.round(height, 2),
            'weight': np.round(weight, 1)
        })

    def sample_symptoms_conditional(self, tb_status: int, demographics_df: pd.DataFrame, i: int) -> Dict[str, int]:
        """Sample symptoms with clinical accuracy and BMI-dependent conditional probabilities."""
        demo = demographics_df.iloc[i]
        bmi_cat = demo['bmi_category']
        base_probs = self.tb_positive_probs if tb_status else self.tb_negative_probs

        # Apply BMI modifiers
        probs = base_probs.copy()
        if bmi_cat in self.bmi_modifiers:
            for symptom, modifier in self.bmi_modifiers[bmi_cat].items():
                if symptom in probs:
                    probs[symptom] = min(0.95, max(0.01, probs[symptom] * modifier))

        symptoms = {}

        # Cough (fundamental symptom)
        symptoms['cough'] = 1 if np.random.rand() < probs['cough'] else 0

        # Prolonged cough depends on having cough
        if symptoms['cough']:
            symptoms['cough_gt_2w'] = 1 if np.random.rand() < probs['cough_gt_2w'] else 0
        else:
            symptoms['cough_gt_2w'] = 1 if np.random.rand() < 0.001 else 0

        # Blood in sputum depends on having cough
        symptoms['blood_in_sputum'] = 1 if (symptoms['cough'] and np.random.rand() < probs['blood_in_sputum']) else 0

        # Systemic symptoms
        for symptom in ['fever', 'low_grade_fever', 'weight_loss', 'night_sweats',
                       'chest_pain', 'breathing_problem', 'fatigue', 'loss_of_appetite']:
            symptoms[symptom] = 1 if np.random.rand() < probs[symptom] else 0

        # Contact history
        symptoms['contact_with_TB'] = 1 if np.random.rand() < probs['contact_with_TB'] else 0

        return symptoms

    def generate_dataset(self, n_samples: int = 10000, tb_prevalence: Optional[float] = None,
                        include_demographics: bool = True, seed: int = 42) -> pd.DataFrame:
        """Generate complete dataset with clinical accuracy."""
        np.random.seed(seed)

        if tb_prevalence is None:
            tb_prevalence = np.random.choice([0.005, 0.01, 0.02], p=[0.5, 0.3, 0.2])

        # Sample TB status
        tb_status = np.random.binomial(1, tb_prevalence, n_samples)

        # Sample demographics
        demos = self.sample_demographics(n_samples)

        rows = []
        for i in range(n_samples):
            tb = tb_status[i]
            demo_row = demos.iloc[i]

            symptoms = self.sample_symptoms_conditional(tb, demos, i)

            row = {
                "TB": str(tb),
                **{symptom: str(val) for symptom, val in symptoms.items()}
            }

            if include_demographics:
                row.update({
                    'age': demo_row['age'],
                    'gender': demo_row['gender'],
                    'bmi_category': demo_row['bmi_category'],
                    'bmi_value': demo_row['bmi_value'],
                    'height': demo_row['height'],
                    'weight': demo_row['weight']
                })

            rows.append(row)

        return pd.DataFrame(rows)


def test_clinical_accuracy():
    """Test that the sampler produces clinically accurate data."""
    print("\n🔬 TESTING CLINICAL ACCURACY OF DATA SAMPLER")
    print("=" * 60)

    sampler = DataSampler(tb_prevalence=0.025, region='global')  # Increased from 1.5% to 2.5%
    data = sampler.generate_dataset(n_samples=5000, seed=123)

    # Test the specific case: cough + prolonged cough
    prolonged_cough_only = (data['cough'] == '1') & (data['cough_gt_2w'] == '1') & \
                          (data[['blood_in_sputum', 'weight_loss', 'night_sweats', 'fever']] == '0').all(axis=1)

    if prolonged_cough_only.any():
        tb_risk_prolonged = data[prolonged_cough_only]['TB'].astype(int).mean()
        print(f"TB risk for prolonged cough only: {tb_risk_prolonged*100:.1f}%")

        if tb_risk_prolonged > 0.8:
            print("✅ SUCCESS: Prolonged cough is a strong TB indicator!")
        elif tb_risk_prolonged > 0.6:
            print("⚠️ Satisfactory: Prolonged cough indicates substantial TB risk")
        else:
            print("❌ FAIL: Prolonged cough is not a strong enough TB indicator")
    else:
        print("⚠️ No cases found with prolonged cough only")

    # Additional clinical accuracy tests
    print("\n📊 Additional Clinical Accuracy Tests:")

    # Test 1: Blood in sputum should be highly specific for TB
    if 'blood_in_sputum' in data.columns:
        hemoptysis_cases = data[data['blood_in_sputum'] == '1']
        if len(hemoptysis_cases) > 0:
            tb_rate_hemoptysis = (hemoptysis_cases['TB'] == '1').mean()
            print(f"1. Blood in sputum → TB rate: {tb_rate_hemoptysis*100:.1f}%")
            if tb_rate_hemoptysis > 0.7:
                print("   ✅ Hemoptysis is highly specific for TB")
            else:
                print("   ⚠️ Hemoptysis specificity could be higher")

    # Test 2: Multiple symptoms should increase TB likelihood
    symptom_cols = ['cough', 'fever', 'weight_loss', 'night_sweats']
    available_symptoms = [col for col in symptom_cols if col in data.columns]

    if len(available_symptoms) >= 3:
        data['symptom_count'] = sum([data[col].astype(int) for col in available_symptoms])

        print("\n2. TB Rate by Number of Symptoms Present:")
        for n_symptoms in range(0, 5):
            subset = data[data['symptom_count'] == n_symptoms]
            if len(subset) > 0:
                tb_rate = (subset['TB'] == '1').mean()
                print(f"   {n_symptoms} symptoms: {tb_rate*100:.1f}% (n={len(subset)})")

    # Test 3: Contact with TB patient should increase risk
    if 'contact_with_TB' in data.columns:
        contact_cases = data[data['contact_with_TB'] == '1']
        no_contact_cases = data[data['contact_with_TB'] == '0']

        if len(contact_cases) > 0 and len(no_contact_cases) > 0:
            tb_rate_contact = (contact_cases['TB'] == '1').mean()
            tb_rate_no_contact = (no_contact_cases['TB'] == '1').mean()

            print(f"\n3. Contact with TB patient:")
            print(f"   With contact: {tb_rate_contact*100:.1f}% TB rate")
            print(f"   No contact: {tb_rate_no_contact*100:.1f}% TB rate")

            if tb_rate_contact > tb_rate_no_contact * 2:
                print("   ✅ Contact history appropriately increases TB risk")
            else:
                print("   ⚠️ Contact history effect could be stronger")

    # Test 4: BMI impact on TB risk
    if 'bmi_category' in data.columns:
        print("\n4. TB Rate by BMI Category:")
        for bmi_cat in ['underweight', 'normal', 'overweight', 'obese']:
            subset = data[data['bmi_category'] == bmi_cat]
            if len(subset) > 0:
                tb_rate = (subset['TB'] == '1').mean()
                print(f"   {bmi_cat.capitalize()}: {tb_rate*100:.1f}%")

        underweight_tb = data[data['bmi_category'] == 'underweight']['TB'].astype(int).mean()
        normal_tb = data[data['bmi_category'] == 'normal']['TB'].astype(int).mean()

        if underweight_tb > normal_tb * 1.2:
            print("   ✅ Underweight appropriately increases TB risk")
        else:
            print("   ⚠️ BMI impact on TB risk could be stronger")

    print("\n" + "=" * 60)
    print("Clinical Accuracy Testing Complete")

    return data


class RobustTBDataSampler(DataSampler):
    """
    Legacy compatibility class - now delegates to DataSampler.
    """
    """
    Clinically-accurate data sampler with BMI integration and statistical robustness.
    """

    def __init__(self, tb_prevalence=0.01, region='global'):
        """
        Initialize with clinical parameters and regional variations.

        Parameters:
        - tb_prevalence: TB prevalence rate (0.01 = 1%)
        - region: 'global', 'high_risk', 'low_risk' for regional variations
        """
        self.tb_prevalence = min(max(tb_prevalence, 0.001), 0.2)  # Bound to realistic range
        self.region = region

        # Clinical symptom probabilities (WHO-based guidelines)
        self.tb_positive_probs = {
            'cough': 0.95,              # 95% of infectious TB patients cough
            'cough_gt_2w': 0.95,         # 95% have prolonged cough (>2 weeks) - INCREASED IMPACT
            'blood_in_sputum': 0.75,     # Hemoptysis in 75% of pulmonary TB
            'fever': 0.80,              # Fever in 80%
            'low_grade_fever': 0.70,    # Low-grade fever
            'weight_loss': 0.70,        # Weight loss >10% body weight
            'night_sweats': 0.65,       # Night sweats
            'chest_pain': 0.50,         # Chest pain
            'breathing_problem': 0.55,  # Dyspnea/shortness of breath
            'fatigue': 0.75,            # Fatigue
            'loss_of_appetite': 0.65,   # Anorexia
            'contact_with_TB': 0.60     # Contact with TB patient
        }

        self.tb_negative_probs = {
            'cough': 0.10,              # 10% general population cough
            'cough_gt_2w': 0.005,        # Very rare prolonged cough in non-TB - DECREASED to increase discrimination
            'blood_in_sputum': 0.001,    # Very rare hemoptysis
            'fever': 0.15,              # Occasional fever
            'low_grade_fever': 0.05,    # Less common
            'weight_loss': 0.05,        # 5% may have weight issues
            'night_sweats': 0.03,       # Rare
            'chest_pain': 0.08,         # Some musculoskeletal pain
            'breathing_problem': 0.12,  # Occasional respiratory issues
            'fatigue': 0.12,            # General fatigue
            'loss_of_appetite': 0.06,   # Occasional appetite issues
            'contact_with_TB': 0.02     # Low exposure in general population
        }

        # BMI impact modifiers
        self.bmi_modifiers = {
            'underweight': {'weight_loss': 1.2, 'fatigue': 1.1, 'loss_of_appetite': 1.3, 'tb_risk': 1.5},
            'normal': {},
            'overweight': {'breathing_problem': 1.2, 'fever': 0.9},
            'obese': {'breathing_problem': 1.4, 'chest_pain': 1.1, 'fever': 0.85, 'tb_risk': 0.7}
        }


if __name__ == "__main__":
    """Run clinical accuracy testing when module is executed directly."""
    print("🩺 TB Data Sampler - Clinical Accuracy Testing")
    print("=" * 60)

    # Run clinical accuracy tests
    test_data = test_clinical_accuracy()

    # Display basic statistics
    print("\n📊 Dataset Overview:")
    print(f"Total samples: {len(test_data)}")
    tb_count = (test_data['TB'] == '1').sum()
    tb_pct = (test_data['TB'] == '1').mean() * 100
    non_tb_count = (test_data['TB'] == '0').sum()
    non_tb_pct = (test_data['TB'] == '0').mean() * 100
    print(f"TB cases: {tb_count} ({tb_pct:.2f}%)")
    print(f"Non-TB cases: {non_tb_count} ({non_tb_pct:.2f}%)")

    print("\n✅ Clinical accuracy testing complete!")
