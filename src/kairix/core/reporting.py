from typing import Dict, Any, Optional

def interpret_kairix_metrics(
    n_samples: int,
    median_time: Optional[float],
    cum_event_rate: float,
    max_time_observed: float,
    prob_at_max_time: float,
    mode: str = 'retention', 
    time_unit: str = 'days'
) -> Dict[str, Any]:
    """
    Generates business-aligned interpretation of survival analysis results, 
    handling the semantic inversion between Retention (staying) and Conversion (acting).

    Args:
        n_samples: Total cohort size.
        median_time: Time at which S(t) = 0.5. None if S(t) never drops below 0.5.
        cum_event_rate: Total % of events observed in the dataset (e.g., 0.05 for 5%).
        max_time_observed: The furthest point in time (t) we have data for.
        prob_at_max_time: The Kaplan-Meier probability S(t) at max_time_observed.
        mode: 'retention' (Event = Churn/Death) or 'conversion' (Event = Purchase/Click).
        time_unit: Label for time axis (e.g., 'days', 'hours').

    Returns:
        Dict containing:
        - 'summary': A polished, executive-summary style string.
        - 'structured_metrics': Raw numbers with business-appropriate keys for dashboards.
    """
    
    narrative = ""
    metrics_payload = {
        "n": n_samples,
        "max_event_horizon": max_time_observed,
        "mode": mode
    }

    # ---------------------------------------------------------
    # MODE 1: RETENTION (Churn Analysis)
    # Goal: Keep S(t) high. Event (Churn) is BAD.
    # ---------------------------------------------------------
    if mode == 'retention':
        # Metric 1: Half-Life (The "Bleed" Metric)
        if median_time:
            kpi_text = f"The cohort **Half-Life** is {median_time:.1f} {time_unit}."
            implication = "50% of users have churned by this point."
            metrics_payload["half_life"] = median_time
            metrics_payload["retention_status"] = "High Churn (Median reached)"
        else:
            kpi_text = (f"The cohort Half-Life (>50% retention) exceeds the "
                        f"observation window of {max_time_observed} {time_unit}.")
            implication = "Retention is stable."
            metrics_payload["half_life"] = None
            metrics_payload["retention_status"] = "Strong (Median not reached)"

        # Metric 2: Terminal Retention
        metrics_payload["retention_at_max"] = prob_at_max_time
        
        narrative = (
            f"**Retention Analysis:** {kpi_text} {implication} "
            f"At the end of {max_time_observed} {time_unit}, "
            f"{prob_at_max_time:.1%} of users remain active."
        )

    # ---------------------------------------------------------
    # MODE 2: CONVERSION (Funnel Analysis)
    # Goal: Drive S(t) to 0. Event (Purchase) is GOOD.
    # Metric focus: 1 - S(t)
    # ---------------------------------------------------------
    elif mode == 'conversion':
        # Invert S(t) to get Cumulative Conversion Rate
        total_conversion = 1.0 - prob_at_max_time
        metrics_payload["total_conversion_rate"] = total_conversion
        
        # Metric 1: Velocity (How fast do they act?)
        if median_time:
            # High Velocity Case (e.g., Micro-funnels, OTP entry)
            kpi_text = (f"**High Velocity:** The majority (50%) of users convert "
                        f"within {median_time:.1f} {time_unit}.")
            metrics_payload["median_time_to_convert"] = median_time
            metrics_payload["velocity_status"] = "High (Median reached)"
        else:
            # Low Velocity Case (Standard SaaS Sales / E-commerce)
            kpi_text = (f"**Conversion Ceiling:** The conversion curve has not crossed 50%. "
                        f"Median time-to-convert is undefined for this window.")
            metrics_payload["median_time_to_convert"] = None
            metrics_payload["velocity_status"] = "Standard (Median not reached)"

        narrative = (
            f"**Funnel Efficiency:** At {max_time_observed} {time_unit}, the cumulative conversion "
            f"reaches {total_conversion:.1%}. {kpi_text}"
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'retention' or 'conversion'.")

    return {
        "summary": narrative,
        "structured_metrics": metrics_payload
    }