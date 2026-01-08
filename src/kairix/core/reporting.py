"""Interpretation Layer for standardized English outputs.

This module provides human-readable interpretations of survival analysis results.
"""

from typing import Dict, Any, Optional


def interpret_kaplan_meier(
    n_samples: int,
    median_survival: Optional[float],
    event_rate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> str:
    """Generate a standardized English interpretation of Kaplan-Meier results.
    
    Args:
        n_samples: Number of samples in the analysis.
        median_survival: Median survival time, or None if not reached.
        event_rate: Proportion of events (non-censored observations).
        ci_lower: Lower bound of median survival confidence interval.
        ci_upper: Upper bound of median survival confidence interval.
        
    Returns:
        Human-readable interpretation string.
    """
    parts = []
    
    # Sample size description
    if n_samples < 100:
        size_desc = "small sample"
    elif n_samples < 1000:
        size_desc = "moderate sample"
    elif n_samples < 10000:
        size_desc = "large sample"
    else:
        size_desc = "very large sample"
    
    parts.append(
        f"Based on a {size_desc} of {n_samples:,} observations "
        f"with an event rate of {event_rate:.1%}, "
    )
    
    # Median survival interpretation
    if median_survival is not None:
        median_str = f"{median_survival:.2f}"
        if ci_lower is not None and ci_upper is not None:
            median_str = f"{median_str} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
        parts.append(f"the median survival time is {median_str}. ")
    else:
        parts.append(
            "the median survival time was not reached during the observation period. "
            "This suggests that more than 50% of subjects survived beyond "
            "the maximum follow-up time. "
        )
    
    return "".join(parts)


def interpret_cox_model(
    n_samples: int,
    n_events: int,
    concordance_index: float,
    coefficients: Dict[str, float],
) -> str:
    """Generate a standardized English interpretation of Cox model results.
    
    Args:
        n_samples: Number of samples in the analysis.
        n_events: Number of events observed.
        concordance_index: C-index (model discrimination ability).
        coefficients: Dictionary of feature coefficients.
        
    Returns:
        Human-readable interpretation string.
    """
    parts = []
    
    parts.append(
        f"Cox proportional hazards model fitted on {n_samples:,} observations "
        f"with {n_events:,} events. "
    )
    
    # C-index interpretation
    if concordance_index < 0.5:
        c_desc = "poor"
    elif concordance_index < 0.7:
        c_desc = "fair"
    elif concordance_index < 0.8:
        c_desc = "good"
    else:
        c_desc = "excellent"
    
    parts.append(
        f"The model has a concordance index of {concordance_index:.3f}, "
        f"indicating {c_desc} discriminatory ability. "
    )
    
    # Coefficient interpretation (top features by absolute value)
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    if sorted_coefs:
        parts.append("\n\nKey findings:\n")
        for name, coef in sorted_coefs[:5]:  # Top 5 features
            hr = round(float(coef), 2)
            direction = "increases" if coef > 0 else "decreases"
            parts.append(
                f"- {name}: Each unit increase {direction} the hazard by {abs(hr):.2f} "
                f"(HR = {hr:.2f})\n"
            )
    
    return "".join(parts)


def generate_summary_report(
    model_type: str,
    stats: Dict[str, Any],
    interpretation: str,
) -> str:
    """Generate a complete summary report for a survival model.
    
    Args:
        model_type: Type of survival model (e.g., "Kaplan-Meier", "Cox").
        stats: Dictionary of model statistics.
        interpretation: Human-readable interpretation string.
        
    Returns:
        Complete summary report string.
    """
    report = f"=== {model_type} Analysis Summary ===\n\n"
    
    for key, value in stats.items():
        if isinstance(value, float):
            formatted = f"{value:.4f}" if value < 0.01 else f"{value:.4f}"
        else:
            formatted = str(value)
        report += f"{key.replace('_', ' ').title()}: {formatted}\n"
    
    report += f"\n=== Interpretation ===\n{interpretation}\n"
    
    return report
