"""
Utility Functions for TB Detection System
Includes validation, cross-validation, sampling utilities, and data processing helpers.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')


def create_sample_data(n=10000, tb_prevalence=None, include_demographics=True, seed=42):
    """
    Create robust synthetic TB dataset with clinical accuracy, BMI integration, and statistical validity.

    Parameters:
    - n: Number of samples
    - tb_prevalence: TB prevalence (0.01 = 1%), if None uses default based on region
    - include_demographics: Include age, gender, BMI features
    - seed: Random seed for reproducibility

    Returns:
    - DataFrame with TB and symptom columns, optionally demographics
    """
    from data_sampler import DataSampler

    np.random.seed(seed)

    # Initialize sampler
    if tb_prevalence is None:
        tb_prevalence = np.random.choice([0.005, 0.01, 0.02], p=[0.5, 0.3, 0.2])  # Realistic distribution
    sampler = DataSampler(tb_prevalence=tb_prevalence, region='global')

    # Sample TB status with prevalence
    tb_status = np.random.binomial(1, tb_prevalence, n)

    # Sample demographics
    demos = sampler.sample_demographics(n)

    rows = []
    for i in range(n):
        tb = tb_status[i]
        demo_row = demos.iloc[i]

        # Sample symptoms conditionally
        symptoms = sampler.sample_symptoms_conditional(tb, demos, i)

        # Create data row
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

    df = pd.DataFrame(rows)

    # Validate data quality
    validate_data_quality(df)

    return df


def stratified_sample_data(data, stratify_column='TB', n_strata=5, test_size=0.2, random_state=42):
    """
    Create stratified train/test splits maintaining class balance.

    Parameters:
    - data: DataFrame to split
    - stratify_column: Column to stratify on
    - n_strata: Number of stratification bins
    - test_size: Proportion for test set
    - random_state: Random state for reproducibility

    Returns:
    - train_data, test_data
    """
    # Create stratification labels
    if stratify_column in data.columns:
        # For numerical columns, create bins for stratification
        if data[stratify_column].dtype in ['int64', 'float64']:
            data['strata'] = pd.cut(data[stratify_column], bins=n_strata, labels=False)
            stratify_col = 'strata'
        else:
            stratify_col = stratify_column
    else:
        data['strata'] = pd.qcut(range(len(data)), q=n_strata, labels=False)
        stratify_col = 'strata'

    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[stratify_col],
        random_state=random_state
    )

    # Remove temporary stratification column
    if 'strata' in train_data.columns:
        train_data = train_data.drop('strata', axis=1)
        test_data = test_data.drop('strata', axis=1)

    return train_data, test_data


def validate_data_quality(data):
    """
    Perform statistical validation on generated data.

    Checks:
    - Symptom independence (chi-squared tests)
    - BMI distribution
    - Clinical consistency
    """
    symptoms = ['cough', 'cough_gt_2w', 'blood_in_sputum', 'fever', 'weight_loss',
                'night_sweats', 'chest_pain', 'breathing_problem', 'fatigue', 'loss_of_appetite']

    print(f"Data quality validation for {len(data)} samples:")

    # Check TB prevalence
    tb_prevalence = data['TB'].astype(int).mean()
    print(f"TB Prevalence: {tb_prevalence:.4f}")

    # Check symptom prevalence in TB vs non-TB
    for symptom in symptoms:
        tb_rate = data[data['TB'] == '1'][symptom].astype(int).mean()
        non_tb_rate = data[data['TB'] == '0'][symptom].astype(int).mean()
        print(f"  - {symptom}: TB={tb_rate:.3f}, non-TB={non_tb_rate:.3f}")

    # Chi-squared test for independence (cough vs other symptoms)
    try:
        contingency = pd.crosstab(data['cough'], data['cough_gt_2w'])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        print(f"  - Chi-squared (cough → cough_gt_2w): {chi2:.2f}, p={p:.4f}")
    except Exception as e:
        print(f"  - Could not compute chi-squared: {e}")

    # BMI distribution if present
    if 'bmi_category' in data.columns:
        bmi_dist = data['bmi_category'].value_counts(normalize=True)
        print("  - BMI distribution:")
        for cat, pct in bmi_dist.items():
            print(f"    {cat}: {pct:.3f}")


def cross_validate_model(data, model_class, n_splits=5, fit_params=None):
    """
    Perform cross-validation on the data with the TB detector.

    Parameters:
    - data: DataFrame with features and TB target
    - model_class: The BayesianTBDetector class
    - n_splits: Number of CV folds
    - fit_params: Dictionary of parameters for model.fit() method (e.g., {'use_priors': True})

    Returns:
    - scores: Dictionary with cross-validation results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    symptom_cols = ['cough', 'cough_gt_2w', 'blood_in_sputum', 'fever', 'low_grade_fever',
                   'weight_loss', 'night_sweats', 'chest_pain', 'breathing_problem',
                   'fatigue', 'loss_of_appetite', 'contact_with_TB']

    data_copy = data.copy()
    # Ensure data is in string format to match model state names
    for col in symptom_cols + ['TB']:
        if col in data_copy.columns:
            data_copy[col] = data_copy[col].astype(str)

    data_cv = data_copy[symptom_cols + ['TB']].copy()

    # Default fit parameters
    if fit_params is None:
        fit_params = {}

    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    fold = 1
    for train_idx, test_idx in skf.split(data_cv, data_cv['TB']):
        print(f"CV Fold {fold}/{n_splits}")

        train_data = data_cv.iloc[train_idx]
        test_data = data_cv.iloc[test_idx]

        # Train model
        model = model_class()
        model.fit(train_data, **fit_params)

        # Evaluate on test set (simplified: predict based on evidence)
        correct_predictions = 0
        true_positives = 0
        predicted_positives = 0
        actual_positives = 0

        # Count actual positives first
        for _, row in test_data.iterrows():
            if int(row['TB']) == 1:
                actual_positives += 1

        # Make predictions
        for _, row in test_data.iterrows():
            evidence = {col: row[col] for col in symptom_cols if col in row}
            pred_prob = model.predict_probability(evidence)
            pred = 1 if pred_prob > 0.5 else 0
            actual = int(row['TB'])

            if pred == actual:
                correct_predictions += 1

            if pred == 1:
                predicted_positives += 1

            if pred == 1 and actual == 1:
                true_positives += 1

        accuracy = correct_predictions / len(test_data)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

        fold += 1

    return {
        'accuracy': {
            'mean': np.mean(accuracy_scores),
            'std': np.std(accuracy_scores),
            'scores': accuracy_scores
        },
        'precision': {
            'mean': np.mean(precision_scores),
            'std': np.std(precision_scores),
            'scores': precision_scores
        },
        'recall': {
            'mean': np.mean(recall_scores),
            'std': np.std(recall_scores),
            'scores': recall_scores
        }
    }


def augment_rare_cases(data, rare_case_threshold=50, augmentation_factor=3):
    """
    Augment rare case combinations to improve model training.

    Parameters:
    - data: DataFrame
    - rare_case_threshold: Minimum frequency for a symptom combo to be considered common
    - augmentation_factor: How many times to duplicate rare cases

    Returns:
    - Augmented DataFrame
    """
    symptom_cols = ['cough', 'cough_gt_2w', 'blood_in_sputum', 'fever', 'weight_loss']

    # Find rare combinations
    combo_counts = data.groupby(symptom_cols)['TB'].count()

    # Assume TB=1 cases with unusual symptom combinations are rare
    tb_data = data[data['TB'] == '1']
    tb_combos = tb_data.groupby(symptom_cols)['TB'].count()

    rare_combos = tb_combos[tb_combos < rare_case_threshold]

    print(f"Found {len(rare_combos)} rare TB-case combinations for augmentation")

    # Augment by duplicating
    augmented_rows = []
    for combo, count in rare_combos.items():
        rare_cases = tb_data[
            (tb_data[symptom_cols[0]] == str(combo[0])) &
            (tb_data[symptom_cols[1]] == str(combo[1])) &
            (tb_data[symptom_cols[2]] == str(combo[2])) &
            (tb_data[symptom_cols[3]] == str(combo[3])) &
            (tb_data[symptom_cols[4]] == str(combo[4]))
        ]

        # Duplicate rare cases
        for _ in range(augmentation_factor - 1):  # Already have 1
            augmented_rows.extend(rare_cases.to_dict('records'))

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        data = pd.concat([data, augmented_df], ignore_index=True)
        print(f"Added {len(augmented_rows)} augmented samples")

    return data
