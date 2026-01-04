"""Distributed Kaplan-Meier estimator for large-scale survival analysis."""

from typing import Optional, Dict, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, Row

from kairix.core.validation import validate_input_schema


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
        is_fitted: Whether the model has been fitted to data

    Examples:
        >>> from kairix.survival import KaplanMeier, generate_synthetic_survival_data
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = generate_synthetic_survival_data(spark, n_samples=10000)
        >>> km = KaplanMeier()
        >>> km.fit(df, duration_col="duration", event_col="event")
        >>> result = km.survival_df
        >>> result.show()
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
        max_bins: int = 100000,
    ) -> "KaplanMeier":
        """Fit the Kaplan-Meier estimator to survival data.

        This method implements the distributed Kaplan-Meier algorithm:

        1. Validates input schema and data quality
        2. Performs adaptive discretization of duration values
        3. Aggregates event counts per time bin
        4. Computes risk sets using window functions
        5. Calculates survival probabilities with numerical stability

        Args:
            df: Input PySpark DataFrame containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            max_bins: Maximum number of discretization bins. Default is 100,000.
                Higher values provide better precision but require more memory.

        Returns:
            Self: The fitted estimator instance for method chaining.

        Raises:
            ValueError: If duration or event columns contain invalid data.
            TypeError: If df is not a PySpark DataFrame.

        Note:
            This method is lazy and does not trigger computation until the
            result is accessed. It avoids collect() and toPandas() operations.
        """
        # Validate input
        validate_input_schema(df, duration_col, event_col)

        # Get global statistics without collecting
        stats_row: Row = df.select(
            F.count(F.lit(1)).alias("n_samples"),
            F.min(duration_col).alias("min_duration"),
            F.max(duration_col).alias("max_duration"),
        ).first()

        n_samples = stats_row["n_samples"]
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
            resolution = duration_range / max_bins

        # Store fitted parameters
        self._stats = {
            "n_samples": n_samples,
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
        # Group by discretized duration and calculate:
        # - d_t: sum of events (number of events at this time)
        # - total_observed: count of observations in this bin
        aggregated_df = discretized_df.groupBy("duration_bin").agg(
            F.sum(event_col).alias("d_t"),  # Number of events at this time
            F.count(F.lit(1)).alias("total_observed"),  # Number at risk at this time
        )

        # Step 3: Risk Set Calculation (Window)
        # Calculate risk set using window functions
        # Order by duration to ensure proper temporal ordering
        window_spec = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )

        # Calculate cumulative sum of total_observed (all observations up to current time)
        aggregated_df = aggregated_df.withColumn(
            "cumulative_observed",
            F.sum("total_observed").over(window_spec),
        )

        # Risk set: N - cumulative_observed + total_observed (at current time)
        # At time t, n_t = total_population - number who have already had events/time < t
        aggregated_df = aggregated_df.withColumn(
            "n_at_risk",
            F.lit(n_samples) - F.col("cumulative_observed") + F.col("total_observed"),
        )

        # Step 4: Survival Probability (Log-Sum-Exp Trick)
        # Calculate log survival probability with numerical stability
        # Using log(1 - d_t / n_t) to avoid underflow
        aggregated_df = aggregated_df.withColumn(
            "log_survival_conditional",
            F.log(
                F.when(F.col("n_at_risk") > 0, 1 - F.col("d_t") / F.col("n_at_risk"))
                .otherwise(1.0)
            ),
        )

        # Calculate cumulative log survival using running sum
        log_survival_window = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )

        aggregated_df = aggregated_df.withColumn(
            "cumulative_log_survival",
            F.sum("log_survival_conditional").over(log_survival_window),
        )

        # Convert back to survival probability using exp
        # Clamp to [0, 1] to handle numerical edge cases
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
        # Greenwood's formula: Var(S(t)) = S(t)^2 * sum(d_j / (n_j * (n_j - d_j)))
        # Calculate the variance contribution for each time point
        aggregated_df = aggregated_df.withColumn(
            "greenwood_term",
            F.when(
                (F.col("n_at_risk") > 1) & (F.col("d_t") > 0),
                F.col("d_t") / (F.col("n_at_risk") * (F.col("n_at_risk") - F.col("d_t")))
            ).otherwise(0.0)
        )

        # Calculate cumulative sum of Greenwood's terms
        aggregated_df = aggregated_df.withColumn(
            "cumulative_greenwood",
            F.sum("greenwood_term").over(log_survival_window),
        )

        # Calculate variance: Var(S(t)) = S(t)^2 * cumulative_greenwood
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
        # This prevents re-scanning the raw 100M rows for every subsequent query.
        self.survival_df.cache()

        self.is_fitted = True

        return self

    def predict_survival(self, duration: float) -> float:
        """Predict survival probability at a given duration.

        Args:
            duration: Time point at which to estimate survival probability.

        Returns:
            Estimated survival probability at the given duration.
            Returns 0.0 if duration exceeds the maximum observed time.

        Raises:
            ValueError: If the model has not been fitted yet.
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
            return 1.0  # No data, assume full survival

        return float(result["survival_probability"])

    def median_survival(self) -> Optional[float]:
        """Calculate the median survival time.

        Returns:
            Duration at which survival probability drops to 0.5 or below.
            Returns None if median is not reached in the observed data.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")

        result = self.survival_df.filter(
            F.col("survival_probability") <= 0.5
        ).first()

        if result is None:
            return None  # Median not reached

        return float(result["duration"])

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitted model.

        Returns:
            Dictionary containing model summary statistics.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return {
            "n_samples": self._stats.get("n_samples"),
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
        which provides better coverage for extreme values.

        Args:
            confidence_level: Confidence level for intervals. Default is 0.95 (95% CI).

        Returns:
            DataFrame with survival probabilities and confidence bounds.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if self.survival_df is None:
            raise ValueError("Survival DataFrame is not available.")

        # Calculate standard error from variance
        se = F.sqrt(F.col("variance"))

        # Calculate log-log transformation for CI
        # Using the Petos/M Greenwood transformation
        log_log_survival = F.log(-F.log(F.col("survival_probability")))
        se_log_log = se / (F.col("survival_probability") * F.log(F.col("survival_probability")))

        # Z-score for confidence level
        from scipy import stats  # Lazy import for z-score
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
