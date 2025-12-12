# Changelog - TB Detection System

## [1.1.0] - 2025-12-12

### Fixed - Critical Bugs Resolved

#### 1. Data Type Mismatch in Bayesian Network State Names
**File:** `detector.py` (lines 25-39)

**Problem:**
- State names were defined as integers `[0, 1]`
- Training data used string values `['0', '1']`
- Caused silent failures in Bayesian inference, returning default probabilities

**Solution:**
```python
# Before
self.state_names = {
    'TB': [0, 1],
    'cough': [0, 1],
    # ... other symptoms
}

# After
self.state_names = {
    'TB': ['0', '1'],
    'cough': ['0', '1'],
    # ... other symptoms
}
```

**Impact:** Model can now properly perform Bayesian inference on evidence

---

#### 2. Impossible Probability Threshold
**File:** `detector.py` (lines 169-180)

**Problem:**
- Risk classification required `probability >= 1.0` for "High TB Risk"
- Probabilities are in range [0, 1), never reaching 1.0
- Model could never classify any case as "High TB Risk"

**Solution:**
```python
# Before
if probability >= 1.0:
    return "High TB Risk"

# After
if probability > 0.80:
    return "High TB Risk"
```

**Impact:** Model can now properly classify high-risk TB cases

---

#### 3. Target Variable Included in Evidence (CRITICAL)
**File:** `run.py` (line 158)

**Problem:**
- The 'TB' column (target variable) was included in the feature set
- Bayesian inference queried P(TB | ..., TB), which is undefined
- Error: "Can't have the same variables in both `variables` and `evidence`"
- Model defaulted to probability 0.5, causing all predictions to be "Non-TB"

**Solution:**
```python
# Before
features = test_data.drop(columns=['age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'], errors='ignore')

# After
features = test_data.drop(columns=['TB', 'age', 'gender', 'bmi_category', 'bmi_value', 'height', 'weight'], errors='ignore')
```

**Impact:** Model now makes actual predictions instead of failing silently

---

#### 4. Evidence Type Conversion Enhancement
**File:** `detector.py` (lines 121-129)

**Problem:**
- Evidence conversion logic was unnecessarily complex
- Potential type mismatches when converting evidence values

**Solution:**
```python
# Before
observed_evidence[k] = str(int(v) if isinstance(v, (int, float)) else v)

# After
if isinstance(v, str):
    observed_evidence[k] = v
else:
    observed_evidence[k] = str(int(v))
```

**Impact:** Cleaner, more reliable evidence handling

---

#### 5. Confusion Matrix Display Error
**File:** `run.py` (lines 227-231)

**Problem:**
- Confusion matrix rows/columns were incorrectly labeled
- TB and Non-TB rows were swapped
- Made interpretation of results confusing

**Solution:**
```python
# Before
print("TB    " + " ".join(f"{x:>6}" for x in cm[1]))
print("Non-TB" + " ".join(f"{x:>6}" for x in cm[0]))

# After
print("         Predicted")
print("Actual    TB   Non-TB")
print(f"TB     {cm[1][1]:>6} {cm[1][0]:>6}")
print(f"Non-TB {cm[0][1]:>6} {cm[0][0]:>6}")
```

**Impact:** Accurate confusion matrix display for proper evaluation

---

### Added - Debugging Features

#### Debug Output for Predictions
**File:** `run.py` (lines 167-190)

**Added:**
- Sample predictions display (first 5 cases)
- TB case-specific debug output (first 3 TB cases)
- Error handling with detailed exception messages
- Symptom count display for TB cases

**Benefit:** Easier diagnosis of prediction issues

---

### Performance Improvements

#### Before Fix
```
Accuracy:   98.7%
Precision:   0.0%  ❌ (Model predicted everything as Non-TB)
Recall:      0.0%  ❌
F1-Score:    0.0%  ❌
Readiness:   55%   (Needs Improvement)
```

#### After Fix
```
Accuracy:   99.9%  ✅
Precision:  96.8%  ✅
Recall:     95.3%  ✅
F1-Score:   96.1%  ✅
Readiness:   75%   (Ready for Controlled Testing)
```

#### Confusion Matrix Results (10,000 patients)
```
         Predicted
Actual    TB   Non-TB
TB        122      6     (95.3% recall)
Non-TB      4   9868    (99.96% specificity)
```

**Key Metrics:**
- **True Positives:** 122 (correctly identified TB cases)
- **False Negatives:** 6 (missed TB cases - 4.7%)
- **False Positives:** 4 (false alarms - 0.04%)
- **True Negatives:** 9,868 (correctly identified healthy)

---

### Dependencies Added

#### New Requirement
- `seaborn` - Required by `data_sampler.py` for statistical visualizations

**Installation:**
```bash
pip install seaborn
```

---

### Technical Details

#### Root Cause Analysis

The primary failure mode was a **cascading type mismatch**:

1. State space defined with integers but data used strings
2. Evidence conversion attempted to reconcile types but failed
3. TB column inadvertently included in evidence
4. Bayesian inference threw exception
5. Exception caught silently, defaulted to 0.5 probability
6. All predictions became 0 (Non-TB) with 0.5 threshold

This created a **silent failure** where:
- No obvious errors were displayed
- Model appeared to run successfully
- Metrics showed high accuracy (98.7%) due to class imbalance
- But precision/recall were 0%, revealing complete failure

#### Verification Steps

To verify the fixes:
```bash
# Delete old model to force retraining
rm -f models/tb_detector.pkl

# Run with UTF-8 encoding support
export PYTHONIOENCODING=utf-8
python run.py
```

Expected output should show:
- TB cases with probabilities near 1.0
- Healthy cases with probabilities near 0.0
- Precision and recall both > 90%
- Readiness score ≥ 75%

---

### Migration Notes

#### For Existing Deployments

1. **Delete saved models:** Old models were trained with incorrect state space
   ```bash
   rm -f models/tb_detector.pkl
   ```

2. **Retrain:** Next run will automatically retrain with correct configuration

3. **Verify:** Check that precision/recall are non-zero in output

#### For New Deployments

No special steps required. The system will work correctly out of the box.

---

### Testing Recommendations

#### Unit Tests Needed
- [ ] Test state name consistency across data and model
- [ ] Test evidence preprocessing with various input types
- [ ] Test that TB column is never in evidence
- [ ] Test confusion matrix calculation and display

#### Integration Tests Needed
- [ ] End-to-end prediction pipeline
- [ ] Model save/load with correct state space
- [ ] Clinical case evaluation accuracy

#### Validation Tests Needed
- [ ] Cross-validation with stratified folds
- [ ] Performance on edge cases (very few symptoms, many symptoms)
- [ ] Probability calibration analysis

---

### Known Limitations

1. **Clinical Accuracy:** Only 40% on clinical test cases (4/10 correct)
   - Issue: Rule-based classifications may be too aggressive
   - Recommendation: Tune thresholds or add more sophisticated rules

2. **False Negatives:** 6 TB cases missed (4.7%)
   - Issue: Patients with atypical presentations
   - Recommendation: Add demographic features to model

3. **Class Imbalance:** Only 1.3% TB prevalence in test data
   - Issue: Real-world performance may vary with different prevalence
   - Recommendation: Test with various prevalence rates

---

### Future Improvements

#### Planned Enhancements
- [ ] Add cross-validation to training pipeline
- [ ] Implement probability calibration
- [ ] Add demographic features to Bayesian network
- [ ] Create automated test suite
- [ ] Add ROC curve and AUC metrics
- [ ] Implement uncertainty quantification

#### Under Consideration
- [ ] Switch to more complex network structure (symptom dependencies)
- [ ] Add temporal modeling for symptom progression
- [ ] Integrate with real clinical data
- [ ] Build REST API for model serving

---

### Contributors

**Bug Fixes:** Claude Code (AI Assistant)
**Original System:** Previous development team
**Date:** December 12, 2025

---

### References

- WHO TB Clinical Guidelines
- pgmpy Documentation: https://pgmpy.org/
- Bayesian Networks for Medical Diagnosis
