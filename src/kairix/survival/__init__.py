"""Survival analysis module for Kairix."""

from kairix.survival.univariate import DistributedKaplanMeier
from kairix.survival.utils import generate_synthetic_survival_data

__all__ = ["DistributedKaplanMeier", "generate_synthetic_survival_data"]
