"""Local Kaplan-Meier implementation using lifelines.

This module provides a pandas-based Kaplan-Meier estimator that wraps
the lifelines library for local/small-scale survival analysis.
"""

from typing import Optional, Dict, Any, Union, List

import pandas as pd
import numpy as np

from kairix.statistics.rmst import RMSTEngine


class KaplanMeierFitter:
    """Local Kaplan-Meier survival curve estimator using lifelines.
    
    Attributes:
        survival_df_: DataFrame containing the survival curve.
        is_fitted: Whether the model has been fitted to data.
    """
    
    def __init__(self):
        """Initialize the KaplanMeierFitter."""
        try:
            from lifelines import KaplanMeierFitter as LKMF
            self._lifelines_class = LKMF # Delayed instantiation
            self._lifelines_kmf = None
        except ImportError:
            raise ImportError(
                "lifelines is required for local Kaplan-Meier estimation. "
                "Install it with: pip install lifelines"
            )
        self.survival_df_: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._stats: Dict[str, Any] = {}
        self._fitted_alpha: float = 0.05
    
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        **kwargs,
    ) -> "KaplanMeierFitter":
        """Fit the Kaplan-Meier estimator to survival data."""
        
        # Instantiate fresh for every fit to clear state
        self._lifelines_kmf = self._lifelines_class()

        # Capture confidence level if passed, else default to 0.95 (alpha=0.05)
        # Lifelines uses alpha (1 - confidence), not confidence_level
        alpha = kwargs.pop("alpha", 0.05)
        self._fitted_alpha = alpha
        
        # Handle label: prefer passed kwarg (from compare_rmst), default to generic
        # This prevents "multiple values for keyword argument 'label'" error
        label = kwargs.pop("label", "survival_probability")
        
        # Fit the lifelines estimator
        self._lifelines_kmf.fit(
            df[duration_col],
            df[event_col],
            label=label, 
            alpha=alpha,
            **kwargs,
        )
        
        # 1. PARITY FIX: Capture exact same stats as Spark implementation
        n_samples = len(df)
        total_events = df[event_col].sum()
        
        self._stats = {
            "n_samples": n_samples,
            "total_events": int(total_events),
            "event_rate": float(total_events / n_samples) if n_samples > 0 else 0.0,
            "min_duration": float(df[duration_col].min()),
            "max_duration": float(df[duration_col].max()),
            "duration_col": duration_col,
            "event_col": event_col,
        }
        
        self._is_fitted = True
        
        # Create survival DataFrame
        # Lifelines returns index as time, column as label
        self.survival_df_ = self._lifelines_kmf.survival_function_.reset_index()
        # We enforce a standard schema regardless of the internal label used
        self.survival_df_.columns = ["duration", "survival_probability"]
        
        # Standardize CIs into the main df immediately
        ci_df = self._lifelines_kmf.confidence_interval_
        if ci_df is not None and not ci_df.empty:
            # Lifelines CI columns are like 'survival_probability_lower_0.95'
            # We blindly grab the first and second columns to map to lower/upper
            self.survival_df_["ci_lower"] = ci_df.iloc[:, 0].values
            self.survival_df_["ci_upper"] = ci_df.iloc[:, 1].values
        
        return self
    
    def predict_survival(self, duration: float) -> float:
        """Predict survival probability at a given duration."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Handle edge cases that Lifelines might warn about
        try:
            return float(self._lifelines_kmf.predict(duration))
        except:
            return 0.0 if duration > self._stats["max_duration"] else 1.0
    
    def median_survival(self) -> Optional[float]:
        """Calculate the median survival time."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        median = self._lifelines_kmf.median_survival_time_
        
        # Lifelines returns infinity if median not reached
        if median == float("inf") or np.isinf(median):
            return None
        return float(median)
    
    def survival_df(self) -> pd.DataFrame:
        """Get the survival curve DataFrame."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.survival_df_
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Merge calculated stats with median
        return {
            **self._stats,
            "median_survival": self.median_survival(),
            "is_fitted": self._is_fitted,
        }
    
    def get_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """Get survival curve with confidence intervals.
        
        Note:
            If the requested confidence_level differs from the fitted one,
            this method currently returns the FITTED intervals and logs a warning.
            Re-fitting for dynamic CIs is expensive and not implemented here.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # Check for mismatch
        requested_alpha = 1.0 - confidence_level
        if abs(requested_alpha - self._fitted_alpha) > 0.0001:
            # In a strict environment, we might want to error or refit.
            # For now, we return existing but notify.
            import warnings
            warnings.warn(
                f"Requested confidence level {confidence_level} differs from "
                f"fitted level {1 - self._fitted_alpha}. Returning fitted intervals."
            )
        
        return self.survival_df_.copy()
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted
    
    def compute_rmst(
        self,
        time_horizon: float,
    ) -> Dict[str, float]:
        """Compute Restricted Mean Survival Time at a given time horizon.
        
        Args:
            time_horizon: The truncation time point for RMST calculation.
        
        Returns:
            Dictionary containing RMST and variance calculations:
                - rmst: Restricted Mean Survival Time
                - variance: Variance of the RMST estimate
                - std_error: Standard error
                - ci_lower: 95% CI lower bound
                - ci_upper: 95% CI upper bound
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        engine = RMSTEngine(time_horizon)
        return engine._compute_from_lifelines(self._lifelines_kmf)
    
    @staticmethod
    def compare_rmst(
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        time_horizon: float,
        group_1_name: Optional[str] = None,
        group_2_name: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare RMST between two groups using log-rank style interface."""
        
        # Get unique groups
        unique_groups = df[group_col].unique().tolist()
        if len(unique_groups) != 2:
            raise ValueError(
                f"Group column must have exactly 2 unique values, "
                f"found {len(unique_groups)}: {unique_groups}"
            )
        
        if group_1_name is None:
            group_1_name = str(unique_groups[0])
        if group_2_name is None:
            group_2_name = str(unique_groups[1])
        
        # Split data
        group_1_data = df[df[group_col] == group_1_name]
        group_2_data = df[df[group_col] == group_2_name]
        
        # Fit KMF for each group
        # Passes df and column names to match the fit() signature
        kmf_1 = KaplanMeierFitter()
        kmf_1.fit(
            group_1_data,    # df
            duration_col,    # col name string
            event_col,       # col name string
            label=group_1_name,
        )
        
        kmf_2 = KaplanMeierFitter()
        kmf_2.fit(
            group_2_data,    # df
            duration_col,    # col name string
            event_col,       # col name string
            label=group_2_name,
        )
        
        # Compute RMST comparison
        engine = RMSTEngine(time_horizon)
        result = engine.calculate_diff_test(kmf_2, kmf_1, alpha=alpha)
        
        # Store group names
        result['group_1_name'] = group_1_name
        result['group_2_name'] = group_2_name
        
        # Store summary stats
        result['summary'] = {
            'n_group_1': len(group_1_data),
            'n_group_2': len(group_2_data),
            'events_group_1': int(group_1_data[event_col].sum()),
            'events_group_2': int(group_2_data[event_col].sum()),
        }
        
        return result