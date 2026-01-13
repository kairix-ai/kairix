"""Kairix - Operationalizing Modern Statistics.

Kairix provides a unified interface for survival analysis that automatically
dispatches to local (pandas/lifelines) or distributed (Spark) implementations.

Example:
    >>> import kairix
    >>> from kairix import SurvivalEstimator, SurvivalTester
    >>> import pandas as pd
    >>> 
    >>> # Kaplan-Meier survival analysis
    >>> df = pd.DataFrame({'duration': [5, 6, 7], 'event': [1, 0, 1]})
    >>> estimator = SurvivalEstimator()
    >>> estimator.fit(df, duration_col='duration', event_col='event')
    >>> print(estimator.median_survival())
    >>> 
    >>> # A/B Testing with log-rank test
    >>> df_ab = pd.DataFrame({
    ...     'tenure': [5, 6, 7, 8, 10, 12, 15, 16],
    ...     'churned': [1, 0, 1, 0, 1, 1, 1, 0],
    ...     'variant': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    ... })
    >>> tester = SurvivalTester()
    >>> result = tester.run_test(df_ab, 'tenure', 'churned', group_col='variant')
    >>> print(f"P-Value: {result['p_value']:.4f}")
"""

__version__ = "0.1.0"

# Unified API exports
from kairix.survival import (
    SurvivalEstimator,
    KaplanMeierFitter,
    KaplanMeier,
    SurvivalTester,
    LogRankResult,
    BayesianResult,
)

__all__ = [
    "SurvivalEstimator",  # Unified interface (recommended)
    "KaplanMeierFitter",  # Local (pandas/lifelines)
    "KaplanMeier",        # Distributed (Spark)
    "SurvivalTester",     # A/B Testing for survival analysis
    "LogRankResult",      # Log-rank test result dataclass
    "BayesianResult",     # Bayesian comparison result dataclass
]
