"""
Unified Survival Analysis Estimator.

This module provides a unified interface for survival analysis that automatically
dispatches to the appropriate implementation (local/lifelines or distributed/Spark)
based on the input DataFrame type.
"""

from typing import Optional, Dict, Any, Union

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import desc  # Required for efficient Spark tail extraction

from kairix.core.engine import get_backend
from kairix.core.reporting import interpret_kairix_metrics
from kairix.survival.local_impl import KaplanMeierFitter as LocalKMFitter
from kairix.survival.spark_impl import KaplanMeier as SparkKMFitter


class SurvivalEstimator:
    """Unified interface for survival analysis estimators.
    
    This class automatically detects whether the input is a pandas DataFrame
    or a PySpark DataFrame and dispatches to the appropriate implementation.
    
    Attributes:
        backend: The detected backend ("pandas" or "spark").
        estimator: The underlying estimator instance.
        
    Examples:
        >>> import pandas as pd
        >>> from kairix.survival import SurvivalEstimator
        >>> df = pd.DataFrame({
        ...     'duration': [5, 6, 7, 8, 10],
        ...     'event': [1, 0, 1, 0, 1]
        ... })
        >>> estimator = SurvivalEstimator()
        >>> estimator.fit(df, duration_col='duration', event_col='event')
        >>> print(estimator.median_survival())
    """
    
    def __init__(self):
        """Initialize the SurvivalEstimator."""
        self.backend: Optional[str] = None
        self._estimator: Optional[Any] = None
        self._is_fitted: bool = False
        self._stats: Dict[str, Any] = {}
    
    def fit(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        duration_col: str,
        event_col: str,
        **kwargs,
    ) -> "SurvivalEstimator":
        """Fit the survival estimator to data.
        
        Args:
            df: Input DataFrame (pandas or PySpark).
            duration_col: Name of the duration/time column.
            event_col: Name of the event indicator column (0=censored, 1=event).
            **kwargs: Additional keyword arguments passed to the backend estimator.
            
        Returns:
            Self for method chaining.
            
        Raises:
            TypeError: If df is not a pandas or PySpark DataFrame.
        """
        # Detect backend
        self.backend = get_backend(df)
        
        # Dispatch to appropriate implementation
        if self.backend == "pandas":
            self._estimator = LocalKMFitter()
            self._estimator.fit(df, duration_col=duration_col, event_col=event_col, **kwargs)
        elif self.backend == "spark":
            self._estimator = SparkKMFitter()
            self._estimator.fit(df, duration_col=duration_col, event_col=event_col, **kwargs)
        
        self._is_fitted = True
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
        return self._estimator.predict_survival(duration)
    
    def median_survival(self) -> Optional[float]:
        """Calculate the median survival time.
        
        Returns:
            Duration at which survival probability drops to 0.5 or below.
            Returns None if median is not reached in the observed data.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._estimator.median_survival()
    
    def survival_df(self) -> Union[pd.DataFrame, SparkDataFrame]:
        """Get the survival curve DataFrame.
        
        Returns:
            DataFrame with columns: duration, survival_probability, and confidence intervals.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._estimator.survival_df()
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary statistics (n_samples, median_survival, etc).
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._estimator.summary()
    
    def get_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """Get survival curve with confidence intervals.
        
        Args:
            confidence_level: Confidence level for intervals. Default is 0.95 (95% CI).
            
        Returns:
            DataFrame with survival probabilities and confidence bounds.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if hasattr(self._estimator, "get_confidence_intervals"):
            return self._estimator.get_confidence_intervals(confidence_level)
        else:
            raise NotImplementedError(
                f"Confidence intervals not supported for {self.backend} backend"
            )
    
    def interpret(self, mode: str = 'retention', time_unit: str = 'days') -> Dict[str, Any]:
        """Generate a business-aligned interpretation of the results.
        
        This method extracts key metrics (Median, Max Observed Time, Terminal Probability)
        using backend-specific optimizations to ensure performance on large datasets.

        Args:
            mode: 'retention' (event = bad/churn) or 'conversion' (event = good/purchase).
            time_unit: Label for the time axis (e.g., 'days', 'hours').

        Returns:
            Dict containing:
                'summary': A human-readable narrative string.
                'structured_metrics': Raw metrics for downstream APIs/Dashboards.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # 1. Get Base Stats
        stats = self.summary()
        n_samples = stats.get("n_samples", 0)
        median_survival = stats.get("median_survival")
        event_rate = stats.get("event_rate", 0.0)
        
        # 2. Extract "Tail" Metrics (Required for Business Logic)
        # We need the probability at the very last observed time point.
        max_time = 0.0
        prob_at_max = 0.0
        
        surv_df = self.survival_df()
        
        if self.backend == "pandas":
            # Pandas: sort by duration and take the last row
            # Assuming columns: 'duration', 'survival_probability'
            if not surv_df.empty:
                tail = surv_df.sort_values(by="duration").iloc[-1]
                max_time = float(tail["duration"])
                prob_at_max = float(tail["survival_probability"])
            
        elif self.backend == "spark":
            # Spark: Optimize by ordering and taking limit 1 (Lazy Evaluation)
            # This avoids collecting the whole curve to the driver.
            # Assuming columns: 'duration', 'survival_probability'
            row = surv_df.orderBy(desc("duration")).select("duration", "survival_probability").first()
            if row:
                max_time = float(row["duration"])
                prob_at_max = float(row["survival_probability"])
        
        # 3. Generate Interpretation using Kairix Business Logic
        return interpret_kairix_metrics(
            n_samples=n_samples,
            median_time=median_survival,
            cum_event_rate=event_rate,
            max_time_observed=max_time,
            prob_at_max_time=prob_at_max,
            mode=mode,
            time_unit=time_unit
        )
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted