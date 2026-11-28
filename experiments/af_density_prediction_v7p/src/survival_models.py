"""Survival and dynamic time-to-event models (Tier C)."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from lifelines import CoxPHFitter, WeibullAFTFitter


class SurvivalBenchmarkSuite:
    """Wraps Cox Proportional Hazards and Weibull AFT models."""

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.cph_ = CoxPHFitter(penalizer=penalizer)
        self.waft_ = WeibullAFTFitter(penalizer=penalizer)

    def fit_and_evaluate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        duration_col: str = "time_to_event_s",
        event_col: str = "event",
        feature_cols: list = None,
    ) -> Dict[str, Any]:
        """Fits Cox PH and Weibull AFT on train_df and evaluates concordance index on val_df."""
        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c not in [duration_col, event_col, "patient_id", "record_id", "dataset", "fold"]]

        # Clean non-finite
        cols = [duration_col, event_col] + feature_cols
        train_clean = train_df[cols].dropna().copy()
        val_clean = val_df[cols].dropna().copy()

        # Ensure positive duration for AFT
        train_clean[duration_col] = np.clip(train_clean[duration_col], 1.0, None)
        val_clean[duration_col] = np.clip(val_clean[duration_col], 1.0, None)

        results = {}

        # 1. Cox PH
        try:
            self.cph_.fit(train_clean, duration_col=duration_col, event_col=event_col)
            c_train_cph = float(self.cph_.concordance_index_)
            # Predict partial hazard on val
            val_scores = -self.cph_.predict_partial_hazard(val_clean[feature_cols])
            from lifelines.utils import concordance_index
            c_val_cph = float(concordance_index(val_clean[duration_col], val_scores, val_clean[event_col]))
            results["cox_c_index_train"] = c_train_cph
            results["cox_c_index_val"] = c_val_cph
        except Exception as e:
            results["cox_c_index_train"] = np.nan
            results["cox_c_index_val"] = np.nan

        # 2. Weibull AFT
        try:
            self.waft_.fit(train_clean, duration_col=duration_col, event_col=event_col)
            c_train_waft = float(self.waft_.concordance_index_)
            val_expectations = self.waft_.predict_expectation(val_clean[feature_cols])
            from lifelines.utils import concordance_index
            c_val_waft = float(concordance_index(val_clean[duration_col], val_expectations, val_clean[event_col]))
            results["weibull_c_index_train"] = c_train_waft
            results["weibull_c_index_val"] = c_val_waft
        except Exception as e:
            results["weibull_c_index_train"] = np.nan
            results["weibull_c_index_val"] = np.nan

        return results
