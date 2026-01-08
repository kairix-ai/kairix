"""Core utilities for Kairix.

This module provides shared infrastructure used across all analysis modules:
- engine: Dispatcher for detecting and handling Pandas vs Spark DataFrames
- validation: Rigorous data checks (data types, sparsity, etc.)
- reporting: Standardized English interpretation layer
"""

from kairix.core.engine import get_backend, is_pandas, is_spark
from kairix.core.validation import validate_input_schema
from kairix.core.reporting import (
    interpret_kaplan_meier,
    interpret_cox_model,
    generate_summary_report,
)

__all__ = [
    # Engine
    "get_backend",
    "is_pandas",
    "is_spark",
    # Validation
    "validate_input_schema",
    # Reporting
    "interpret_kairix_metrics",
]
