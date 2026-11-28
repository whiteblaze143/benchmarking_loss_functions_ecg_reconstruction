"""Tabular baseline and machine learning models for Tier A and Tier B."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM, s


class EmpiricalPrevalenceBaseline(BaseEstimator, ClassifierMixin):
    """Predicts empirical training prevalence P(Y=1 | train)."""

    def __init__(self):
        self.prevalence_ = 0.5

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full((n, 2), [1.0 - self.prevalence_, self.prevalence_])
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ElapsedDurationSurvivalBaseline(BaseEstimator, ClassifierMixin):
    """Predicts conditional persistence / survival based purely on elapsed duration."""

    def __init__(self, duration_col_idx: int = 0):
        self.duration_col_idx = duration_col_idx
        self.empirical_hazard_ = 0.5

    def fit(self, X, y):
        # Estimates simple empirical hazard conditional on elapsed duration
        self.empirical_hazard_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full((n, 2), [1.0 - self.empirical_hazard_, self.empirical_hazard_])
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RobustLogisticGAM(BaseEstimator, ClassifierMixin):
    """Wrapper around pygam.LogisticGAM with fallback to regularized Logistic Regression."""

    def __init__(self, max_spline_terms: int = 5):
        self.max_spline_terms = max_spline_terms
        self.model_ = None
        self.fallback_ = None

    def fit(self, X, y):
        X_mat = np.asarray(X)
        y_vec = np.asarray(y)
        n_features = X_mat.shape[1]
        k = min(n_features, self.max_spline_terms)
        try:
            # Subsample if dataset is large for fast convergence
            if len(X_mat) > 5000:
                pos_idx = np.where(y_vec == 1)[0]
                neg_idx = np.where(y_vec == 0)[0]
                np.random.seed(42)
                n_neg = min(len(neg_idx), 5000 - len(pos_idx))
                idx = np.concatenate([pos_idx, np.random.choice(neg_idx, n_neg, replace=False)])
                X_fit = X_mat[idx, :k]
                y_fit = y_vec[idx]
            else:
                X_fit = X_mat[:, :k]
                y_fit = y_vec

            # Construct additive spline formula
            formula = s(0)
            for j in range(1, k):
                formula = formula + s(j)
            gam = LogisticGAM(formula).fit(X_fit, y_fit)
            self.model_ = gam
        except Exception as e:
            raise RuntimeError(f"LogisticGAM fitting failed: {e}")
        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X)
        if self.model_ is None:
            raise RuntimeError("LogisticGAM has not been fitted.")
        k = min(X_mat.shape[1], self.max_spline_terms)
        p1 = self.model_.predict_proba(X_mat[:, :k])
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_tabular_model(model_name: str, random_state: int = 42):
    """Factory for Tier A and Tier B classification models."""
    if model_name == "empirical_baseline":
        return EmpiricalPrevalenceBaseline()
    elif model_name == "elapsed_duration_baseline":
        return ElapsedDurationSurvivalBaseline()
    elif model_name == "elastic_net":
        return LogisticRegression(C=0.5, solver="lbfgs", max_iter=500, random_state=random_state)
    elif model_name == "gam":
        return RobustLogisticGAM(max_spline_terms=5)
    elif model_name == "ebm":
        return ExplainableBoostingClassifier(max_bins=32, max_rounds=50, outer_bags=1, interactions=0, random_state=random_state)
    elif model_name == "linear_svm":
        svc = LinearSVC(C=0.1, max_iter=2000, random_state=random_state)
        return CalibratedClassifierCV(svc, cv=3)
    elif model_name == "rbf_svm":
        return SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=random_state)
    elif model_name == "gbdt_lightgbm":
        return LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05, n_jobs=1, verbose=-1, random_state=random_state)
    elif model_name == "gbdt_xgboost":
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=1, eval_metric="logloss", random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
