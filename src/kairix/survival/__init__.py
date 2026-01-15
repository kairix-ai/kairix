"""Survival analysis module for Kairix.

This module provides unified survival analysis tools that automatically
dispatch to local (pandas/lifelines) or distributed (Spark) implementations
based on the input DataFrame type.

Example:
    >>> from kairix.survival import SurvivalTester, KaplanMeierFitter, KaplanMeier
    >>> import pandas as pd
    >>> 
    >>> # Kaplan-Meier survival analysis
    >>> df = pd.DataFrame({'duration': [5, 6, 7], 'event': [1, 0, 1]})
    >>> km = KaplanMeierFitter()
    >>> km.fit(df, duration_col='duration', event_col='event')
    >>> print(km.median_survival())
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
    >>> 
    >>> # Bayesian comparison
    >>> bayes = tester.run_bayesian_comparison(df_ab, 'tenure', 'churned', group_col='variant')
    >>> print(f"Prob B > A: {bayes['prob_superiority']:.2%}")
"""

from kairix.survival.estimator import SurvivalEstimator
from kairix.survival.local_impl import KaplanMeierFitter
from kairix.survival.spark_impl import KaplanMeier
from kairix.survival.ab_testing import (
    SurvivalTester, LogRankResult, BayesianResult, CriticalTimeResult,
    RMSTResult, HorizonAnalysisResult
)

__all__ = [
    "SurvivalEstimator",  # Unified interface
    "KaplanMeierFitter",  # Local (pandas/lifelines)
    "KaplanMeier",        # Distributed (Spark)
    "SurvivalTester",     # A/B Testing for survival analysis
    "LogRankResult",      # Log-rank test result dataclass
    "BayesianResult",     # Bayesian comparison result dataclass
    "CriticalTimeResult", # Critical time analysis result dataclass
    "RMSTResult",         # RMST comparison result dataclass
    "HorizonAnalysisResult", # Horizon analysis result dataclass
]
