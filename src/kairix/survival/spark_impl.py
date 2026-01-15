"""Distributed Kaplan-Meier estimator for large-scale survival analysis using PySpark."""

from typing import Optional, Dict, Any, Union

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, Row

# Ensure kairix.core.validation is available or mock this for standalone testing
from kairix.core.validation import validate_input_schema
from kairix.statistics.rmst import RMSTEngine


class KaplanMeier:
    """Distributed Kaplan-Meier survival curve estimator using PySpark.

    This class implements a scalable Kaplan-Meier estimator that can handle
    massive datasets (100M+ rows) without relying on driver-side memory.
    It uses the Map-Window-Reduce pattern for distributed computation.

    The algorithm follows these steps:
    1. Adaptive Discretization: Transform continuous durations into bins
    2. Aggregation: Compute event counts and risk set sizes per bin
    3. Risk Set Calculation: Use window functions for cumulative sums
    4. Survival Probability: Apply log-sum-exp trick for numerical stability

    Attributes:
        survival_df: DataFrame containing the survival curve with columns:
            - duration: The discretized time point
            - n_at_risk: Number of subjects at risk at this time
            - n_events: Number of events at this time
            - survival_probability: Estimated survival probability
            - variance: Variance of the survival probability (Greenwood)
        is_fitted: Whether the model has been fitted to data
    """

    def __init__(self) -> None:
        """Initialize the DistributedKaplanMeier estimator."""
        self.survival_df: Optional[DataFrame] = None
        self.is_fitted: bool = False
        self._stats: Dict[str, Any] = {}

    def fit(
        self,
        df: DataFrame,
        duration_col: str,
        event_col: str,
        bins: int = 10000,
    ) -> "KaplanMeier":
        """Fit the Kaplan-Meier estimator to survival data.

        Args:
            df: Input PySpark DataFrame containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            bins: Number of discretization bins. Default is 10000.

        Returns:
            Self: The fitted estimator instance for method chaining.
        """
        # Validate input
        validate_input_schema(df, duration_col, event_col)

        # 1. OPTIMIZATION: Get all global stats in a SINGLE pass
        # We calculate total_events here to support Conversion Rate logic downstream
        stats_row: Row = df.select(
            F.count(F.lit(1)).alias("n_samples"),
            F.sum(event_col).alias("total_events"),
            F.min(duration_col).alias("min_duration"),
            F.max(duration_col).alias("max_duration"),
        ).first()

        n_samples = stats_row["n_samples"]
        total_events = stats_row["total_events"]
        min_duration = stats_row["min_duration"]
        max_duration = stats_row["max_duration"]

        if n_samples == 0:
            raise ValueError("DataFrame is empty")

        if min_duration is None or max_duration is None:
            raise ValueError("Duration column contains only NULL values")

        # Handle edge case where all durations are the same
        duration_range = max_duration - min_duration
        if duration_range == 0:
            resolution = 1.0
        else:
            resolution = duration_range / bins

        # Store fitted parameters
        self._stats = {
            "n_samples": n_samples,
            "total_events": total_events,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "resolution": resolution,
            "duration_col": duration_col,
            "event_col": event_col,
        }

        # Step 1: Adaptive Discretization (Map)
        # Transform continuous durations into discrete bins
        discretized_df = df.withColumn(
            "duration_bin",
            (F.floor(F.col(duration_col) / resolution) * resolution).cast(DoubleType()),
        )

        # Step 2: Aggregation (Reduce)
        # Group by discretized duration
        aggregated_df = discretized_df.groupBy("duration_bin").agg(
            F.sum(event_col).alias("d_t"),  # Number of events at this time
            F.count(F.lit(1)).alias("total_observed"),  # Number at risk at this time
        )

        # Step 3: Risk Set Calculation (Window)
        # Calculate risk set using window functions
        window_spec = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )

        # Calculate cumulative sum of total_observed
        aggregated_df = aggregated_df.withColumn(
            "cumulative_observed",
            F.sum("total_observed").over(window_spec),
        )

        # Risk set: N - cumulative_observed + total_observed (at current time)
        aggregated_df = aggregated_df.withColumn(
            "n_at_risk",
            F.lit(n_samples) - F.col("cumulative_observed") + F.col("total_observed"),
        )

        # Step 4: Survival Probability (Log-Sum-Exp Trick)
        # Using log(1 - d_t / n_t) to avoid underflow
        aggregated_df = aggregated_df.withColumn(
            "log_survival_conditional",
            F.log(
                F.when(F.col("n_at_risk") > 0, 1 - F.col("d_t") / F.col("n_at_risk"))
                .otherwise(1.0)
            ),
        )

        log_survival_window = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )

        aggregated_df = aggregated_df.withColumn(
            "cumulative_log_survival",
            F.sum("log_survival_conditional").over(log_survival_window),
        )

        # Convert back to survival probability using exp
        aggregated_df = aggregated_df.withColumn(
            "survival_probability",
            F.exp(F.col("cumulative_log_survival")),
        )

        # Ensure survival probability is bounded [0, 1]
        aggregated_df = aggregated_df.withColumn(
            "survival_probability",
            F.when(F.col("survival_probability") < 0, 0.0)
            .when(F.col("survival_probability") > 1, 1.0)
            .otherwise(F.col("survival_probability")),
        )

        # Step 5: Variance Calculation (Greenwood's Formula)
        aggregated_df = aggregated_df.withColumn(
            "greenwood_term",
            F.when(
                (F.col("n_at_risk") > 1) & (F.col("d_t") > 0),
                F.col("d_t") / (F.col("n_at_risk") * (F.col("n_at_risk") - F.col("d_t")))
            ).otherwise(0.0)
        )

        aggregated_df = aggregated_df.withColumn(
            "cumulative_greenwood",
            F.sum("greenwood_term").over(log_survival_window),
        )

        aggregated_df = aggregated_df.withColumn(
            "variance",
            F.pow(F.col("survival_probability"), 2) * F.col("cumulative_greenwood"),
        )

        # Ensure variance is non-negative
        aggregated_df = aggregated_df.withColumn(
            "variance",
            F.when(F.col("variance") < 0, 0.0).otherwise(F.col("variance"))
        )

        # Select and rename columns for final output
        self.survival_df = aggregated_df.select(
            F.col("duration_bin").alias("duration"),
            F.col("n_at_risk"),
            F.col("d_t").alias("n_events"),
            F.col("survival_probability"),
            F.col("variance"),
        ).orderBy("duration")

        # PERFORMANCE CRITICAL: Cache the small aggregated result
        # This prevents re-scanning the raw rows for every subsequent query.
        self.survival_df.cache()

        self.is_fitted = True

        return self

    def predict_survival(self, duration: float) -> float:
        """Predict survival probability at a given duration.

        Args:
            duration: Time point at which to estimate survival probability.

        Returns:
            Estimated survival probability. Returns 1.0 if duration < min,
            or the last observed value if duration > max.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")

        # Find the survival probability at or just before the given duration
        result = self.survival_df.filter(F.col("duration") <= duration).orderBy(
            F.col("duration").desc()
        ).first()

        if result is None:
            # Duration is smaller than the smallest observed duration
            return 1.0

        return float(result["survival_probability"])

    def median_survival(self) -> Optional[float]:
        """Calculate the median survival time.

        Returns:
            Duration at which survival probability drops to 0.5 or below.
            Returns None if median is not reached in the observed data.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")

        # Spark Lazy Evaluation Optimization:
        # We only need the FIRST row where prob <= 0.5.
        # This scans the small cached dataframe, not the original data.
        result = self.survival_df.filter(
            F.col("survival_probability") <= 0.5
        ).first()

        if result is None:
            return None  # Median not reached

        return float(result["duration"])

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model including business metrics.

        Returns:
            Dictionary containing model summary statistics.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        n_samples = self._stats.get("n_samples", 0)
        total_events = self._stats.get("total_events", 0)

        # Calculate Median Survival (this triggers a small Spark job on cached DF)
        median = self.median_survival()

        return {
            "n_samples": n_samples,
            "total_events": total_events,
            # This allows the Interpreter to calculate global conversion rate:
            "event_rate": total_events / n_samples if n_samples > 0 else 0.0,
            "median_survival": median,
            "min_duration": self._stats.get("min_duration"),
            "max_duration": self._stats.get("max_duration"),
            "duration_col": self._stats.get("duration_col"),
            "event_col": self._stats.get("event_col"),
            "is_fitted": self.is_fitted,
        }

    def get_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> DataFrame:
        """Get survival curve with confidence intervals.

        Uses the log-log transformation for confidence intervals,
        which provides better coverage for extreme values (Petos/Greenwood).

        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95).

        Returns:
            DataFrame with 'ci_lower' and 'ci_upper' columns appended.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")

        # Calculate standard error from variance
        se = F.sqrt(F.col("variance"))

        # Calculate log-log transformation for CI
        # log(-log(S(t)))
        log_log_survival = F.log(-F.log(F.col("survival_probability")))
        se_log_log = se / (F.col("survival_probability") * F.log(F.col("survival_probability")))

        # Z-score for confidence level
        from scipy import stats  # Lazy import to avoid heavy dependency at module level
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        # Calculate log-log CI bounds
        log_log_lower = log_log_survival - z_score * se_log_log
        log_log_upper = log_log_survival + z_score * se_log_log

        # Transform back to survival scale
        lower_bound = F.exp(-F.exp(log_log_lower))
        upper_bound = F.exp(-F.exp(log_log_upper))

        # Apply boundary constraints
        lower_bound = F.when(lower_bound < 0, 0.0).otherwise(lower_bound)
        lower_bound = F.when(lower_bound > 1, 1.0).otherwise(lower_bound)
        upper_bound = F.when(upper_bound < 0, 0.0).otherwise(upper_bound)
        upper_bound = F.when(upper_bound > 1, 1.0).otherwise(upper_bound)

        return self.survival_df.withColumn(
            "ci_lower", lower_bound
        ).withColumn(
            "ci_upper", upper_bound
        )
    
    # to make api compatible with Lifelines
    @property
    def timeline_(self):
        """Return timeline as numpy array for compatibility with Lifelines."""
        if self.survival_df is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        # Collect the duration column and convert to numpy array
        pdf = self.survival_df.select("duration").toPandas()
        return pdf["duration"].values

    @property
    def survival_function_(self):
        """Return survival function as pandas Series for compatibility with Lifelines."""
        if self.survival_df is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        # Collect the survival probability column and return as pandas Series
        pdf = self.survival_df.select("survival_probability").toPandas()
        return pdf["survival_probability"]

    @property
    def median_survival_time_(self) -> float:
        """Return median survival time for compatibility with Lifelines.
        
        Returns:
            Median survival time. Returns inf if median is not reached.
        """
        median = self.median_survival()
        if median is None:
            return float('inf')
        return median

    # Aliases for compatibility with user's script
    @property
    def timeline(self):
        """Alias for timeline_ for compatibility."""
        return self.timeline_

    @property
    def survival_prob(self):
        """Alias for survival_function_ for compatibility."""
        return self.survival_function_.values
    
    def compute_rmst(
        self,
        time_horizon: float,
    ) -> Dict[str, float]:
        """Compute Restricted Mean Survival Time at a given time horizon.
        
        Args:
            time_horizon: The truncation time point for RMST calculation.
        
        Returns:
            Dictionary containing RMST and variance calculations:
                - rmst: Restricted Mean Survival Time
                - variance: Variance of the RMST estimate
                - std_error: Standard error
                - ci_lower: 95% CI lower bound
                - ci_upper: 95% CI upper bound
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")
        
        # Collect the survival DataFrame to driver (it's small - aggregated data)
        pdf = self.survival_df.toPandas()
        
        # Prepare arrays for RMSTEngine
        times = pdf["duration"].values
        probs = pdf["survival_probability"].values
        event_counts = pdf["n_events"].values
        at_risk_counts = pdf["n_at_risk"].values
        
        engine = RMSTEngine(time_horizon)
        return engine._compute_from_arrays(times, probs, event_counts, at_risk_counts)
    
    @staticmethod
    def compare_rmst(
        df: DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        time_horizon: float,
        group_1_name: Optional[str] = None,
        group_2_name: Optional[str] = None,
        alpha: float = 0.05,
        bins: int = 10000,
    ) -> Dict[str, Any]:
        """Compare RMST between two groups using distributed computation.
        
        This is a static method that fits Kaplan-Meier estimators for both groups
        using distributed Spark computation and computes the RMST difference.
        
        Args:
            df: Input PySpark DataFrame containing survival data.
            duration_col: Column name for duration/time-to-event.
            event_col: Column name for event indicator (0=censored, 1=event).
            group_col: Column name for group assignment.
            time_horizon: Time horizon for RMST calculation.
            group_1_name: Name of the control group. If None, uses first unique value.
            group_2_name: Name of the treatment group. If None, uses second unique value.
            alpha: Significance level for the test.
            bins: Number of discretization bins for KM estimation.
        
        Returns:
            Dictionary containing:
                - diff: RMST treatment - RMST control
                - p_value: Two-sided p-value from Z-test
                - z_score: Z-statistic for the test
                - horizon: The time horizon used
                - significant: Whether the difference is statistically significant
                - treatment_rmst: RMST of treatment group
                - control_rmst: RMST of control group
                - group_1_name: Name of control group
                - group_2_name: Name of treatment group
        """
        # Get unique groups
        unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        if len(unique_groups) != 2:
            raise ValueError(
                f"Group column must have exactly 2 unique values, "
                f"found {len(unique_groups)}: {unique_groups}"
            )
        
        if group_1_name is None:
            group_1_name = str(unique_groups[0])
        if group_2_name is None:
            group_2_name = str(unique_groups[1])
        
        # Filter and fit KMF for each group
        group_1_data = df.filter(F.col(group_col) == group_1_name)
        group_2_data = df.filter(F.col(group_col) == group_2_name)
        
        kmf_1 = KaplanMeier()
        kmf_1.fit(
            group_1_data,
            duration_col=duration_col,
            event_col=event_col,
            bins=bins,
        )
        
        kmf_2 = KaplanMeier()
        kmf_2.fit(
            group_2_data,
            duration_col=duration_col,
            event_col=event_col,
            bins=bins,
        )
        
        # Compute RMST for each group
        result_1 = kmf_1.compute_rmst(time_horizon)
        result_2 = kmf_2.compute_rmst(time_horizon)
        
        # Compute difference with Z-test
        diff = result_2['rmst'] - result_1['rmst']
        pooled_var = result_2['variance'] + result_1['variance']
        pooled_se = (pooled_var ** 0.5) if pooled_var > 0 else 0.0
        
        if pooled_se > 0:
            z_score = diff / pooled_se
        else:
            z_score = 0.0
        
        from scipy import stats
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
        significant = p_value < alpha
        
        # 95% CI for difference
        z_975 = stats.norm.ppf(1.0 - alpha / 2)
        ci_lower = diff - z_975 * pooled_se
        ci_upper = diff + z_975 * pooled_se
        
        # Get summary stats
        summary_1 = kmf_1.summary()
        summary_2 = kmf_2.summary()
        
        return {
            "diff": float(diff),
            "p_value": float(p_value),
            "z_score": float(z_score),
            "horizon": time_horizon,
            "significant": bool(significant),
            "treatment_rmst": result_2['rmst'],
            "control_rmst": result_1['rmst'],
            "treatment_variance": result_2['variance'],
            "control_variance": result_1['variance'],
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "group_1_name": group_1_name,
            "group_2_name": group_2_name,
            "summary": {
                'n_group_1': summary_1['n_samples'],
                'n_group_2': summary_2['n_samples'],
                'events_group_1': summary_1['total_events'],
                'events_group_2': summary_2['total_events'],
            },
        }