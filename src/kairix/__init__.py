"""Kairix - Operationalizing Modern Statistics.

Kairix provides a unified interface for survival analysis that automatically
dispatches to local (pandas/lifelines) or distributed (Spark) implementations.

Example:
    >>> import kairix
    >>> from kairix import SurvivalEstimator
    >>> import pandas as pd
    >>> 
    >>> # Unified API - works with pandas
    >>> df = pd.DataFrame({'duration': [5, 6, 7], 'event': [1, 0, 1]})
    >>> estimator = SurvivalEstimator()
    >>> estimator.fit(df, duration_col='duration', event_col='event')
    >>> print(estimator.median_survival())
"""

__version__ = "0.1.0"

# Unified API exports
from kairix.survival import (
    SurvivalEstimator,
    KaplanMeierFitter,
    KaplanMeier,
)

__all__ = [
    "SurvivalEstimator",  # Unified interface (recommended)
    "KaplanMeierFitter",  # Local (pandas/lifelines)
    "KaplanMeier",        # Distributed (Spark)
]
