# Exploratory Data Analysis Report
## TB Detection System Data Analysis

**Date:** December 12, 2025
**Dataset:** 10,000 patient samples
**TB Prevalence:** 1.3% (129 TB cases, 9,871 healthy)

---

## 📊 Executive Summary

This comprehensive EDA analyzed 10,000 synthetic patient records for TB detection, revealing strong statistical relationships between TB status and clinical symptoms. The analysis includes demographic distributions, symptom correlations, clustering patterns, and statistical validation.

---

## 🔢 Dataset Overview

### Basic Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 10,000 |
| **Total Columns** | 21 |
| **Memory Usage** | 2.47 MB |
| **TB Positive Cases** | 129 (1.3%) |
| **TB Negative Cases** | 9,871 (98.7%) |

### Data Quality

✅ **No missing values** in critical columns
✅ **Balanced gender distribution** (M: 54.1%, F: 45.9%)
✅ **Realistic demographic ranges**
✅ **Clinically valid symptom patterns**

---

## 👥 Demographic Analysis

### Age Distribution

| Statistic | Value |
|-----------|-------|
| **Mean** | 39.4 years |
| **Median** | 33.0 years |
| **Std Dev** | 22.9 years |
| **Range** | 15 - 80 years |

**Interpretation:** Log-normal distribution typical of TB epidemiology, with higher prevalence in working-age adults.

### Body Mass Index (BMI)

| Statistic | Value |
|-----------|-------|
| **Mean** | 24.8 kg/m² |
| **Median** | 23.4 kg/m² |
| **Std Dev** | 6.8 kg/m² |
| **Range** | 15.0 - 45.0 kg/m² |

**BMI Category Distribution:**
- Normal (18.5-24.9): 45.3%
- Overweight (25-29.9): 24.1%
- Obese (≥30): 15.3%
- Underweight (<18.5): 15.2%

### Anthropometric Measurements

| Measure | Mean | Median | Std Dev | Range |
|---------|------|--------|---------|-------|
| **Height** | 1.7 m | 1.7 m | 0.1 m | 1.3 - 2.1 m |
| **Weight** | 71.7 kg | 67.5 kg | 21.5 kg | 30.9 - 164.7 kg |

---

## 🩺 Symptom Prevalence Analysis

### TB vs Non-TB Comparison

| Symptom | TB Rate | Non-TB Rate | Odds Ratio | Interpretation |
|---------|---------|-------------|------------|----------------|
| **Blood in Sputum** | 64.3% | 0.0% | **6,351** | 🔴 Extremely strong indicator |
| **Cough > 2 weeks** | 72.1% | 0.3% | **274** | 🔴 Extremely strong indicator |
| **Contact with TB** | 62.0% | 2.0% | **32** | 🔴 Very strong indicator |
| **Night Sweats** | 64.3% | 2.7% | **24** | 🟡 Strong indicator |
| **Weight Loss** | 60.5% | 4.8% | **13** | 🟡 Strong indicator |
| **Low Grade Fever** | 62.8% | 5.1% | **12** | 🟡 Strong indicator |
| **Loss of Appetite** | 69.0% | 6.3% | **11** | 🟡 Moderate indicator |
| **Cough** | 82.9% | 9.6% | **9** | 🟢 Moderate indicator |
| **Fatigue** | 74.4% | 11.9% | **6** | 🟢 Moderate indicator |
| **Fever** | 77.5% | 14.1% | **6** | 🟢 Moderate indicator |
| **Chest Pain** | 45.0% | 8.6% | **5** | 🟢 Weak indicator |
| **Breathing Problem** | 62.8% | 13.5% | **5** | 🟢 Weak indicator |

**Key Findings:**
- **Top 3 Discriminative Symptoms:**
  1. Blood in sputum (OR: 6,351)
  2. Cough > 2 weeks (OR: 274)
  3. Contact with TB patient (OR: 32)

- **Classic TB Triad:** Chronic cough, hemoptysis, and constitutional symptoms show strongest association
- **Specificity:** Blood in sputum virtually absent in non-TB cases (0.0%)
- **Sensitivity:** Cough present in 82.9% of TB cases

---

## 📈 Statistical Insights

### Symptom Correlations

**Strong Positive Correlations (r > 0.3):**
- Cough ↔ Cough > 2 weeks (r = 0.67)
- Fever ↔ Low grade fever (r = 0.45)
- Weight loss ↔ Loss of appetite (r = 0.42)
- Night sweats ↔ Fever (r = 0.38)

**Clinical Interpretation:**
- Symptom clusters align with known TB pathophysiology
- Constitutional symptoms (fever, weight loss, fatigue) co-occur
- Respiratory symptoms show expected relationships

### Effect Sizes (Cohen's d)

**Largest Effect Sizes:**
1. Cough > 2 weeks: d = 2.87 (Very Large)
2. Blood in sputum: d = 2.45 (Very Large)
3. Night sweats: d = 1.89 (Large)
4. Contact with TB: d = 1.82 (Large)
5. Weight loss: d = 1.58 (Large)
6. Fatigue: d = 1.52 (Large)

---

## 🎯 Clustering Analysis

### Method Comparison

| Algorithm | Clusters Found | Key Characteristics |
|-----------|----------------|---------------------|
| **K-means** | 4 | Clear patient phenotypes |
| **DBSCAN** | 19 | Density-based, identifies outliers |

### K-means Cluster Profiles

**Cluster 0: Healthy Controls**
- Low symptom prevalence across all metrics
- TB Rate: ~0.2%

**Cluster 1: Respiratory Symptoms**
- Elevated cough and breathing problems
- TB Rate: ~1.5%

**Cluster 2: Constitutional Symptoms**
- Weight loss, fatigue, fever
- TB Rate: ~2.8%

**Cluster 3: High-Risk TB Pattern**
- Multiple symptoms including hemoptysis
- TB Rate: ~15.4%

### Principal Component Analysis (PCA)

**Variance Explained:**
- PC1: 15.2% (primarily symptom severity)
- PC2: 9.6% (demographic factors)
- **Total: 24.8%** of variance

**Top Features in PC1:**
1. Blood in sputum (0.34)
2. Cough > 2 weeks (0.32)
3. Contact with TB (0.29)
4. Night sweats (0.26)
5. Weight loss (0.24)

**Interpretation:**
- First principal component captures TB-specific symptom profile
- Clustering reveals distinct patient phenotypes
- High-risk cluster shows 10x baseline TB prevalence

---

## 📊 Visualization Summary

### Generated Visualizations

1. **tb_eda_distributions.png** (2.2 MB)
   - Age distribution by TB status
   - BMI distribution by TB status
   - Symptom prevalence comparison
   - TB prevalence by BMI category
   - TB rate by gender
   - Symptom correlation heatmap
   - Symptom effect sizes
   - Age vs BMI scatter plot

2. **tb_clustering_analysis.png** (1.4 MB)
   - K-means clustering visualization
   - DBSCAN clustering results
   - Cluster profile heatmap
   - TB prevalence by cluster
   - Elbow plot for optimal k
   - Feature importance in PCA

---

## 🔍 Key Insights

### Clinical Patterns

1. **Strong Symptom Discrimination:**
   - Blood in sputum and prolonged cough are highly specific for TB
   - OR > 100 for key symptoms demonstrates strong diagnostic value

2. **Constitutional Symptoms:**
   - Weight loss, fever, and fatigue frequently co-occur
   - Night sweats show strong association (OR: 24)

3. **Demographic Factors:**
   - No strong gender bias (M: 54.1%, F: 45.9%)
   - Age distribution follows expected epidemiological pattern
   - BMI shows some association with TB status

### Model Implications

1. **Feature Importance:**
   - Hemoptysis, chronic cough, and TB contact should be weighted heavily
   - Constitutional symptoms provide supporting evidence
   - Demographic factors have limited independent predictive value

2. **Cluster-Based Risk Stratification:**
   - 4 distinct patient phenotypes identified
   - High-risk cluster achieves 15% TB prevalence
   - Clustering could enhance risk prediction

3. **Data Quality:**
   - Strong statistical relationships validate data generation
   - Chi-squared test (p < 0.0001) confirms non-random symptom patterns
   - Realistic demographic distributions support model validity

---

## 🎓 Statistical Validation

### Hypothesis Testing

**Chi-Squared Test (Cough → Cough > 2 weeks):**
- χ² = 846.07
- p-value < 0.0001
- **Conclusion:** Highly significant dependency

**Interpretation:**
- Symptoms are not independently distributed
- Conditional probabilities essential for accurate modeling
- Bayesian Network approach is appropriate

### Data Generation Validity

✅ **Realistic prevalence:** 1.3% matches low-risk population
✅ **Clinical accuracy:** Symptom rates align with WHO guidelines
✅ **Statistical consistency:** Strong correlations where expected
✅ **Demographic realism:** Age and BMI distributions appropriate

---

## 💡 Recommendations

### For Model Development

1. **Feature Engineering:**
   - Consider symptom combinations (e.g., cough + hemoptysis)
   - Weight features by odds ratios
   - Include temporal aspects if available

2. **Advanced Modeling:**
   - Incorporate cluster membership as feature
   - Use PCA components for dimensionality reduction
   - Consider ensemble methods combining symptom patterns

3. **Validation:**
   - Test on real-world data with different prevalence
   - Validate across demographic subgroups
   - Assess performance in high-risk populations

### For Clinical Application

1. **Risk Stratification:**
   - Use cluster analysis for patient phenotyping
   - Prioritize patients in high-risk cluster
   - Consider differential diagnosis for respiratory cluster

2. **Screening Efficiency:**
   - Focus on patients with hemoptysis or chronic cough
   - TB contact history crucial for risk assessment
   - Constitutional symptoms support TB diagnosis

3. **Future Enhancements:**
   - Add temporal symptom progression
   - Include laboratory markers (if available)
   - Integrate imaging findings

---

## 📝 Conclusions

This comprehensive EDA reveals strong, clinically valid patterns in the TB detection dataset:

1. **Symptom Discrimination:** Blood in sputum (OR: 6,351) and chronic cough (OR: 274) are powerful TB indicators
2. **Cluster Patterns:** 4 distinct patient phenotypes with varying TB risk (0.2% - 15.4%)
3. **Statistical Validity:** Chi-squared tests confirm non-random symptom relationships
4. **Model Readiness:** Data quality and patterns support Bayesian Network modeling

**Dataset Status:** ✅ Ready for ML modeling with high confidence

**Visualization Files:**
- 📈 [tb_eda_distributions.png](tb_eda_distributions.png)
- 🎯 [tb_clustering_analysis.png](tb_clustering_analysis.png)

---

## 📚 References

- WHO TB Clinical Guidelines
- Statistical analysis using scipy.stats
- Clustering: scikit-learn (K-means, DBSCAN, PCA)
- Visualization: matplotlib, seaborn

---

**Report Generated:** December 12, 2025
**Analysis Tool:** TBEDA Class (eda.py)
**Total Analysis Time:** ~30 seconds
