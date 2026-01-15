"""Statistics module for Kairix.

This module provides pure mathematical and statistical implementations
that are decoupled from specific backend implementations.
"""

from kairix.statistics.rmst import RMSTEngine, compare_rmst_groups

__all__ = [
    "RMSTEngine",
    "compare_rmst_groups",
]
