"""Local Kaplan-Meier implementation using lifelines.

This module provides a pandas-based Kaplan-Meier estimator that wraps
the lifelines library for local/small-scale survival analysis.
"""

from typing import Optional, Dict, Any, Union

import pandas as pd


class KaplanMeierFitter:
    """Local Kaplan-Meier survival curve estimator using lifelines.
    
    This class wraps the lifelines KaplanMeierFitter to provide a consistent
    interface with the distributed Spark implementation.
    
    Attributes:
        kmf: The underlying lifelines KaplanMeierFitter instance.
        survival_df_: DataFrame containing the survival curve.
        is_fitted: Whether the model has been fitted to data.
        
    Examples:
        >>> from kairix.survival import KaplanMeierFitter
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'duration': [5, 6, 7, 8, 10],
        ...     'event': [1, 0, 1, 0, 1]
        ... })
        >>> kmf = KaplanMeierFitter()
        >>> kmf.fit(df, duration_col='duration', event_col='event')
        >>> print(kmf.median_survival())
    """
    
    def __init__(self):
        """Initialize the KaplanMeierFitter."""
        try:
            from lifelines import KaplanMeierFitter as LKMF
            self._lifelines_kmf = LKMF()
        except ImportError:
            raise ImportError(
                "lifelines is required for local Kaplan-Meier estimation. "
                "Install it with: pip install lifelines"
            )
        self.survival_df_: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._stats: Dict[str, Any] = {}
    
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        **kwargs,
    ) -> "KaplanMeierFitter":
        """Fit the Kaplan-Meier estimator to survival data.
        
        Args:
            df: Input pandas DataFrame containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            **kwargs: Additional keyword arguments passed to lifelines.
            
        Returns:
            Self: The fitted estimator instance for method chaining.
        """
        # Fit the lifelines estimator
        self._lifelines_kmf.fit(
            df[duration_col],
            df[event_col],
            label=kwargs.pop("label", "survival"),
            **kwargs,
        )
        
        # Store statistics
        self._stats = {
            "n_samples": len(df),
            "duration_col": duration_col,
            "event_col": event_col,
            "event_rate": df[event_col].mean(),
        }
        
        self._is_fitted = True
        
        # Create survival DataFrame
        self.survival_df_ = self._lifelines_kmf.survival_function_.reset_index()
        self.survival_df_.columns = ["duration", "survival_probability"]
        
        # Add confidence intervals if available
        if hasattr(self._lifelines_kmf, "confidence_interval_"):
            ci = self._lifelines_kmf.confidence_interval_
            if ci is not None and not ci.empty:
                self.survival_df_["ci_lower"] = ci.iloc[:, 0].values
                self.survival_df_["ci_upper"] = ci.iloc[:, 1].values
        
        return self
    
    def predict_survival(self, duration: float) -> float:
        """Predict survival probability at a given duration.
        
        Args:
            duration: Time point at which to estimate survival probability.
            
        Returns:
            Estimated survival probability at the given duration.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return float(self._lifelines_kmf.predict(duration))
    
    def median_survival(self) -> Optional[float]:
        """Calculate the median survival time.
        
        Returns:
            Duration at which survival probability drops to 0.5 or below.
            Returns None if median is not reached in the observed data.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        median = self._lifelines_kmf.median_survival_time_
        if median == float("inf"):
            return None
        return float(median)
    
    def survival_df(self) -> pd.DataFrame:
        """Get the survival curve DataFrame.
        
        Returns:
            DataFrame with columns: duration, survival_probability, and confidence intervals.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.survival_df_
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary statistics.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        summary = self._lifelines_kmf.summary
        return {
            **self._stats,
            "median_survival": self.median_survival(),
            "is_fitted": self._is_fitted,
        }
    
    def get_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """Get survival curve with confidence intervals.
        
        Args:
            confidence_level: Confidence level for intervals. Default is 0.95 (95% CI).
            
        Returns:
            DataFrame with survival probabilities and confidence bounds.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        ci = self._lifelines_kmf.confidence_interval_
        if ci is None or ci.empty:
            raise ValueError("Confidence intervals not available")
        
        result = self.survival_df_.copy()
        result["ci_lower"] = ci.iloc[:, 0].values
        result["ci_upper"] = ci.iloc[:, 1].values
        return result
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted
