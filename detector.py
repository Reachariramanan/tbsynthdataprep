"""
Professional Bayesian TB Detection Module
Implements Bayesian Network-based tuberculosis detection with clinical accuracy.
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')


class BayesianTBDetector:
    """
    Professional Bayesian Network-based TB detector with clinical accuracy and explainability.
    """

    def __init__(self):
        """
        Initialize Bayesian Network for TB detection
        """
        # Define state names for all variables (all binary '0'/'1' as strings to match data)
        self.state_names = {
            'TB': ['0', '1'],
            'cough': ['0', '1'],
            'cough_gt_2w': ['0', '1'],
            'blood_in_sputum': ['0', '1'],
            'fever': ['0', '1'],
            'low_grade_fever': ['0', '1'],
            'weight_loss': ['0', '1'],
            'night_sweats': ['0', '1'],
            'chest_pain': ['0', '1'],
            'breathing_problem': ['0', '1'],
            'fatigue': ['0', '1'],
            'loss_of_appetite': ['0', '1'],
            'contact_with_TB': ['0', '1']
        }

        # Define structure (DAG) - simplified: TB -> all symptoms directly
        edges = [('TB', 'cough'), ('TB', 'cough_gt_2w'), ('TB', 'blood_in_sputum'),
                 ('TB', 'fever'), ('TB', 'low_grade_fever'), ('TB', 'weight_loss'),
                 ('TB', 'night_sweats'), ('TB', 'fatigue'), ('TB', 'chest_pain'),
                 ('TB', 'breathing_problem'), ('TB', 'loss_of_appetite'), ('TB', 'contact_with_TB')]
        self.model = DiscreteBayesianNetwork(edges)

        self.inference = None
        # clinical prior weights (can be adjusted)
        self.prior_weights = {
            'cough': 0.40,
            'cough_gt_2w': 0.50,       # increased importance
            'blood_in_sputum': 0.60,   # much stronger indicator
            'weight_loss': 0.35,
            'fever': 0.40,
            'low_grade_fever': 0.50,
            'night_sweats': 0.12,
            'fatigue': 0.08,
            'chest_pain': 0.10,
            'breathing_problem': 0.12,
            'loss_of_appetite': 0.40,
            'contact_with_TB': 0.55,
        }

        # will be set after training
        self.prior_tb = None

    def fit(self, data, use_priors=False, pseudo_counts=1):
        """
        Train the Bayesian Network.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns for each variable including 'TB'.
        use_priors : bool
            Whether to use Bayesian estimation (Dirichlet) or MLE.
        pseudo_counts : int/float
            Strength of Dirichlet prior when use_priors=True.
        """
        if use_priors:
            self.model.fit(
                data,
                estimator=BayesianEstimator,
                prior_type='dirichlet',
                pseudo_counts=pseudo_counts
            )
        else:
            self.model.fit(data, estimator=MaximumLikelihoodEstimator)

        # Initialize inference engine
        self.inference = VariableElimination(self.model)

        # Extract prior probability P(TB=1) from learned CPD if available
        try:
            cpd_tb = self.model.get_cpds('TB')
            print(f"CPD for TB: {cpd_tb}")
            print(f"CPD values: {cpd_tb.values}")
            # cpd_tb.values is usually an array indexed [0]=P(TB=0), [1]=P(TB=1)
            values = getattr(cpd_tb, "values", None) or getattr(cpd_tb, "get_values", None)
            if callable(values):
                values = values()
            if isinstance(values, (list, tuple, np.ndarray)):
                # assume index 1 corresponds to TB=1
                self.prior_tb = float(np.array(values).ravel()[1])
            else:
                # fallback: use uniform prior if we can't read it
                self.prior_tb = 0.05
        except Exception:
            # if CPD not found or any problem, fallback to a small default
            self.prior_tb = 0.05

        return self

    def predict_probability(self, evidence):
        """
        Calculate P(TB=1 | evidence) using the trained Bayesian network.

        If evidence is empty (or all None), returns the prior P(TB=1).
        """
        # Clean evidence: remove None entries and ensure values are strings '0' or '1'
        observed_evidence = {}
        for k, v in (evidence or {}).items():
            if v is not None:
                # Convert to string '0' or '1' to match the state space
                if isinstance(v, str):
                    observed_evidence[k] = v
                else:
                    observed_evidence[k] = str(int(v))

        # If inference not initialized, error
        if self.inference is None:
            raise RuntimeError("Model not trained. Call fit(...) before predict_probability.")

        # If no observed evidence, return prior
        if not observed_evidence:
            if self.prior_tb is None:
                # As a last resort, query the network (shouldn't be needed after fit)
                result = self.inference.query(variables=['TB'], evidence={}, show_progress=False)
                return float(result.values[1])
            return float(self.prior_tb)

        # Query the network for P(TB | evidence)
        result = self.inference.query(variables=['TB'], evidence=observed_evidence, show_progress=False)
        # result.values should be [P(TB='0'), P(TB='1')]
        tb_prob = float(result.values[1])
        return tb_prob

    def predict_with_dynamic_adjustment(self, responses, penalty_per_no=0.01, apply_penalty=True):
        """
        Predict with optional dynamic penalty for 'No' responses.

        Returns (risk_category, adjusted_probability, base_probability)
        """
        base_probability = self.predict_probability(responses)

        if apply_penalty:
            no_count = sum(1 for v in responses.values() if v == 0)
            penalty_factor = 1.0 - (no_count * penalty_per_no)
            penalty_factor = max(0.1, min(1.0, penalty_factor))
            adjusted_probability = base_probability * penalty_factor
        else:
            adjusted_probability = base_probability

        # Check for rule-based classifications first
        rule_risk = self.classify_with_rules(responses)
        if rule_risk is not None:
            risk_category = rule_risk
        else:
            risk_category = self.classify_risk(adjusted_probability)
        return risk_category, float(adjusted_probability), float(base_probability)

    def classify_risk(self, probability):
        """Convert probability to a risk category."""
        if probability > 0.80:
            return "High TB Risk"
        elif probability > 0.50:
            return "Moderate TB Risk"
        elif probability > 0.25:
            return "Low TB Risk"
        elif probability > 0.10:
            return "Pulmonary Issue"
        else:
            return "Healthy"

    def classify_with_rules(self, responses):
        """
        Check for specific symptom combinations and return corresponding risk category.
        Returns None if no rules match.
        """
        # Helper to check if symptom is yes (1)
        def is_yes(symptom):
            val = responses.get(symptom, None)
            return val == 1 or val == '1'

        # High TB Risk rules
        if is_yes('cough_gt_2w') and is_yes('blood_in_sputum'):
            return "High TB Risk"
        if is_yes('cough_gt_2w') and is_yes('weight_loss'):
            return "High TB Risk"
        if is_yes('cough_gt_2w') and is_yes('chest_pain') and is_yes('night_sweats'):
            return "High TB Risk"

        # Moderate TB Risk rules
        if is_yes('cough_gt_2w') and is_yes('fever'):
            return "Moderate TB Risk"
        if is_yes('cough_gt_2w') and is_yes('contact_with_TB'):
            return "Moderate TB Risk"
        if is_yes('cough') and is_yes('fever') and is_yes('night_sweats') and is_yes('fatigue'):
            return "Moderate TB Risk"

        # Low TB Risk rule
        if is_yes('weight_loss') and is_yes('loss_of_appetite') and is_yes('fatigue'):
            return "Low TB Risk"

        return None

    def get_symptom_importance(self, evidence):
        """
        For each symptom in evidence, compute importance as the absolute change
        in posterior P(TB=1) when only that single symptom is observed compared
        to the prior P(TB=1).
        """
        base_prob = float(self.prior_tb if self.prior_tb is not None else self.predict_probability({}))
        importance = {}

        for symptom, val in (evidence or {}).items():
            if val is None:
                continue
            # compute P(TB | symptom = val)
            prob_with_symptom = self.predict_probability({symptom: val})
            importance[symptom] = abs(float(prob_with_symptom) - base_prob)

        return importance

    def explain_prediction(self, responses, top_k=None):
        """
        Return explanation dict with probability, risk, and contributions.
        top_k: if set, returns top_k key symptoms by importance.
        """
        prob = self.predict_probability(responses)
        importance = self.get_symptom_importance(responses)
        sorted_symptoms = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Use rule-based risk if applies, else probabilistic
        rule_risk = self.classify_with_rules(responses)
        risk = rule_risk if rule_risk is not None else self.classify_risk(prob)

        explanation = {
            'probability': float(prob),
            'risk': risk,
            'key_symptoms': sorted_symptoms[:top_k] if top_k else sorted_symptoms,
            'all_contributions': sorted_symptoms
        }
        return explanation
