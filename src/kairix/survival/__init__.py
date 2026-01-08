"""Survival analysis module for Kairix.

This module provides unified survival analysis tools that automatically
dispatch to local (pandas/lifelines) or distributed (Spark) implementations
based on the input DataFrame type.

Example:
    >>> from kairix.survival import SurvivalEstimator, KaplanMeierFitter, KaplanMeier
    >>> import pandas as pd
    >>> 
    >>> # Unified interface (auto-detects pandas vs spark)
    >>> df = pd.DataFrame({'duration': [5, 6, 7], 'event': [1, 0, 1]})
    >>> estimator = SurvivalEstimator()
    >>> estimator.fit(df, duration_col='duration', event_col='event')
    >>> print(estimator.median_survival())
    >>> print(estimator.interpret())
"""

from kairix.survival.estimator import SurvivalEstimator
from kairix.survival.local_impl import KaplanMeierFitter
from kairix.survival.spark_impl import KaplanMeier
from kairix.survival.utils import generate_synthetic_survival_data

__all__ = [
    "SurvivalEstimator",  # Unified interface
    "KaplanMeierFitter",  # Local (pandas/lifelines)
    "KaplanMeier",        # Distributed (Spark)
    "generate_synthetic_survival_data",
]
