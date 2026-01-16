"""Restricted Mean Survival Time (RMST) Engine.

This module implements pure mathematical RMST calculations following the Tian et al. method.
It is designed to be decoupled from any specific survival analysis backend (lifelines, Spark, etc.),
accepting raw numpy arrays for maximum flexibility.
"""

from typing import Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats


class RMSTEngine:
    """Pure mathematical implementation of Restricted Mean Survival Time.
    
    This class computes RMST and its variance using vectorized numpy operations.
    It does not depend on any specific survival analysis library, making it
    reusable across local (lifelines) and distributed (Spark) backends.
    
    RMST at time tau is the area under the survival curve from 0 to tau:
    RMST(tau) = integral_0^tau S(t) dt
    
    The variance is computed using the reverse-cumsum method which is
    computationally efficient for large datasets.
    
    Attributes:
        time_horizon: The truncation time point for RMST calculation.
    """
    
    def __init__(self, time_horizon: float):
        """Initialize the RMST engine with a time horizon.
        
        Args:
            time_horizon: The truncation time point (tau) for RMST calculation.
                          The RMST will be computed as the area under the survival
                          curve from 0 to this time point.
        """
        if time_horizon <= 0:
            raise ValueError(f"time_horizon must be positive, got {time_horizon}")
        self.time_horizon = time_horizon
    
    def _compute_from_arrays(
        self,
        times: np.ndarray,
        probs: np.ndarray,
        event_counts: np.ndarray,
        at_risk_counts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute RMST and variance from raw survival data arrays.
        
        This is the core computational method that accepts raw numpy arrays,
        making it compatible with data from any backend (lifelines, Spark, etc.).
        
        Args:
            times: Array of unique event times from the Kaplan-Meier estimator.
            probs: Array of survival probabilities corresponding to each time.
            event_counts: Array of observed events at each time point.
            at_risk_counts: Array of subjects at risk at each time point.
        
        Returns:
            Dictionary containing:
                - rmst: Restricted Mean Survival Time
                - variance: Variance of the RMST estimate
                - std_error: Standard error of the RMST estimate
                - ci_lower: 95% CI lower bound
                - ci_upper: 95% CI upper bound
        """
        if len(times) == 0:
            warnings.warn("Empty times array, returning zeros")
            return {
                "rmst": 0.0,
                "variance": 0.0,
                "std_error": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }
        
        # Ensure arrays are numpy arrays
        times = np.asarray(times, dtype=np.float64)
        probs = np.asarray(probs, dtype=np.float64)
        event_counts = np.asarray(event_counts, dtype=np.float64)
        at_risk_counts = np.asarray(at_risk_counts, dtype=np.float64)
        
        # Clip at time_horizon
        # Find indices where times <= time_horizon
        valid_mask = times <= self.time_horizon
        if not np.any(valid_mask):
            warnings.warn(
                f"No events observed before time_horizon={self.time_horizon}. "
                "Consider increasing the time horizon."
            )
            return {
                "rmst": 0.0,
                "variance": 0.0,
                "std_error": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }
        
        times = times[valid_mask]
        probs = probs[valid_mask]
        event_counts = event_counts[valid_mask]
        at_risk_counts = at_risk_counts[valid_mask]
        
        # Handle the last interval: if time_horizon falls between time points
        # We need to extend the last survival probability to time_horizon
        if len(times) > 0 and times[-1] < self.time_horizon:
            # Add the time_horizon point with the last survival probability
            times = np.append(times, self.time_horizon)
            probs = np.append(probs, probs[-1])
            
            # FIX: Also extend counts to match length, preventing IndexError.
            # We append 0 since the horizon cutoff is not an event time.
            event_counts = np.append(event_counts, 0.0)
            at_risk_counts = np.append(at_risk_counts, 0.0)
        
        # Compute RMST using the trapezoidal rule (rectangle summation for step functions)
        # For Kaplan-Meier, we use left-endpoint rectangle summation
        rmst = 0.0
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            rmst += probs[i] * dt
        
        # Add the final rectangle from last time to time_horizon if applicable
        if times[-1] == self.time_horizon and len(times) > 1:
            dt = times[-1] - times[-2]
            rmst += probs[-2] * dt
        
        # Compute variance using the reverse-cumsum method (Tian et al.)
        # Var(RMST) = sum over all unique times t of [S(tau)^2 * Var(dN(t)) / Y(t)^2]
        # where dN(t) is the event count and Y(t) is the at-risk count
        
        # Compute survival probability at tau
        if len(probs) > 0:
            s_tau = probs[-1]
        else:
            s_tau = 1.0
        
        # Variance contribution from each time point
        variance = 0.0
        for i in range(len(times)):
            # Check > 0 to avoid division by zero
            # The appended horizon point has counts=0, so it safely skips this block
            if at_risk_counts[i] > 0 and event_counts[i] > 0:
                # Contribution from this time point to variance
                # The factor (time_horizon - t_i) accounts for the remaining area
                remaining_time = self.time_horizon - times[i]
                if remaining_time > 0:
                    var_contribution = (s_tau**2) * (event_counts[i] / (at_risk_counts[i] ** 2))
                    variance += var_contribution * (remaining_time ** 2)
        
        std_error = np.sqrt(variance)
        
        # 95% CI using normal approximation
        z_95 = stats.norm.ppf(0.975)
        ci_lower = max(0.0, rmst - z_95 * std_error)
        ci_upper = rmst + z_95 * std_error
        
        return {
            "rmst": float(rmst),
            "variance": float(variance),
            "std_error": float(std_error),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
    
    def _compute_from_lifelines(
        self,
        kmf: Any,
    ) -> Dict[str, float]:
        """Compute RMST from a fitted lifelines KaplanMeierFitter object.
        
        Args:
            kmf: A fitted lifelines.KaplanMeierFitter object.
        
        Returns:
            Dictionary containing RMST and variance calculations.
        """
        # Extract data from lifelines KMF
        # Get the survival function DataFrame
        sf = kmf.survival_function_
        if sf is None or sf.empty:
            raise ValueError("KaplanMeierFitter has no survival function. Call fit() first.")
        
        times = sf.index.values
        probs = sf.iloc[:, 0].values
        
        # Get event table from lifelines
        event_table = kmf.event_table
        if event_table is None:
            raise ValueError("KaplanMeierFitter has no event table. Call fit() first.")
        
        event_counts = event_table['observed'].values
        at_risk_counts = event_table['at_risk'].values
        
        return self._compute_from_arrays(times, probs, event_counts, at_risk_counts)
    
    def calculate_diff_test(
        self,
        data_treatment: Dict[str, Any],
        data_control: Dict[str, Any],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare RMST between treatment and control groups.
        
        Performs a Z-test for the difference in RMST between two groups:
        Z = (RMST_treatment - RMST_control) / sqrt(Var_treatment + Var_control)
        
        Args:
            data_treatment: Dictionary containing RMST data for treatment group.
                Expected keys: 'times', 'probs', 'event_counts', 'at_risk_counts'
                Alternatively, can be a fitted lifelines.KaplanMeierFitter object.
            data_control: Dictionary containing RMST data for control group.
                Same format as data_treatment.
            alpha: Significance level for the test. Default is 0.05.
        
        Returns:
            Dictionary containing:
                - diff: RMST treatment - RMST control
                - p_value: Two-sided p-value from Z-test
                - z_score: Z-statistic for the test
                - horizon: The time horizon used
                - significant: Whether the difference is statistically significant
                - treatment_rmst: RMST of treatment group
                - control_rmst: RMST of control group
                - ci_lower: 95% CI lower bound for difference
                - ci_upper: 95% CI upper bound for difference
        """
        # Compute RMST for each group
        if hasattr(data_treatment, 'survival_function_'):  # lifelines KMF object
            treatment_result = self._compute_from_lifelines(data_treatment)
        else:
            treatment_result = self._compute_from_arrays(
                data_treatment['times'],
                data_treatment['probs'],
                data_treatment['event_counts'],
                data_treatment['at_risk_counts'],
            )
        
        if hasattr(data_control, 'survival_function_'):  # lifelines KMF object
            control_result = self._compute_from_lifelines(data_control)
        else:
            control_result = self._compute_from_arrays(
                data_control['times'],
                data_control['probs'],
                data_control['event_counts'],
                data_control['at_risk_counts'],
            )
        
        # Compute difference
        diff = treatment_result['rmst'] - control_result['rmst']
        
        # Pooled variance
        pooled_var = treatment_result['variance'] + control_result['variance']
        pooled_se = np.sqrt(pooled_var)
        
        # Z-score
        if pooled_se > 0:
            z_score = diff / pooled_se
        else:
            z_score = 0.0
        
        # Two-sided p-value
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
        
        # Significance
        significant = p_value < alpha
        
        # 95% CI for difference
        z_975 = stats.norm.ppf(1.0 - alpha / 2)
        ci_lower = diff - z_975 * pooled_se
        ci_upper = diff + z_975 * pooled_se
        
        return {
            "diff": float(diff),
            "p_value": float(p_value),
            "z_score": float(z_score),
            "horizon": self.time_horizon,
            "significant": bool(significant),
            "treatment_rmst": treatment_result['rmst'],
            "control_rmst": control_result['rmst'],
            "treatment_variance": treatment_result['variance'],
            "control_variance": control_result['variance'],
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
    
    def compute_from_survival_df(
        self,
        survival_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute RMST from a survival DataFrame.
        
        This method accepts a survival DataFrame with columns:
        - duration: Time points
        - survival_probability: Survival probabilities
        - n_events: (Optional) Event counts at each time
        - n_at_risk: (Optional) Number at risk at each time
        
        Args:
            survival_df: DataFrame with survival curve data.
        
        Returns:
            Dictionary containing RMST and variance calculations.
        """
        if 'duration' not in survival_df.columns:
            raise ValueError("survival_df must have 'duration' column")
        if 'survival_probability' not in survival_df.columns:
            raise ValueError("survival_df must have 'survival_probability' column")
        
        times = survival_df['duration'].values
        probs = survival_df['survival_probability'].values
        
        # If event counts and at-risk counts are available, use them
        if 'n_events' in survival_df.columns and 'n_at_risk' in survival_df.columns:
            event_counts = survival_df['n_events'].values
            at_risk_counts = survival_df['n_at_risk'].values
        else:
            # Estimate from survival curve changes
            # When survival drops, it indicates events
            event_counts = np.zeros(len(probs))
            at_risk_counts = np.zeros(len(probs))
            
            # Approximate: assume constant at-risk until events
            n_total = len(times)  # Approximation
            at_risk_counts = np.full(len(probs), n_total)
            
            # Events proportional to survival drop
            survival_drops = np.zeros(len(probs))
            survival_drops[0] = 1.0 - probs[0]
            for i in range(1, len(probs)):
                survival_drops[i] = probs[i - 1] - probs[i]
            
            event_counts = survival_drops * n_total
        
        return self._compute_from_arrays(times, probs, event_counts, at_risk_counts)


def compare_rmst_groups(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    time_horizon: float,
    group_1_name: Optional[str] = None,
    group_2_name: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Convenience function to compare RMST between two groups.
    
    Args:
        df: DataFrame containing survival data.
        duration_col: Column name for duration/time-to-event.
        event_col: Column name for event indicator (0=censored, 1=event).
        group_col: Column name for group assignment.
        time_horizon: Time horizon for RMST calculation.
        group_1_name: Name of the first group (control). If None, uses first unique value.
        group_2_name: Name of the second group (treatment). If None, uses second unique value.
        alpha: Significance level for the test.
    
    Returns:
        Dictionary with RMST comparison results.
    """
    # Get unique groups
    unique_groups = df[group_col].unique().tolist()
    if len(unique_groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {len(unique_groups)}")
    
    if group_1_name is None:
        group_1_name = str(unique_groups[0])
    if group_2_name is None:
        group_2_name = str(unique_groups[1])
    
    # Split data
    group_1_data = df[df[group_col] == group_1_name]
    group_2_data = df[df[group_col] == group_2_name]
    
    # Import lifelines locally to avoid dependency issues
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise ImportError(
            "lifelines is required for RMST comparison. "
            "Install it with: pip install lifelines"
        )
    
    # Fit KMF for each group
    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(
        group_1_data[duration_col],
        group_1_data[event_col],
        label="control",
    )
    
    kmf_2 = KaplanMeierFitter()
    kmf_2.fit(
        group_2_data[duration_col],
        group_2_data[event_col],
        label="treatment",
    )
    
    # Compute RMST comparison
    engine = RMSTEngine(time_horizon)
    result = engine.calculate_diff_test(kmf_2, kmf_1, alpha=alpha)
    
    result['group_1_name'] = group_1_name
    result['group_2_name'] = group_2_name
    
    return result