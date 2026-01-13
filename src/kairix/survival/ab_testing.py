"""A/B Testing for Survival Analysis using Log-Rank Test.

This module provides functionality to compare survival curves between two groups
using the log-rank test and Bayesian inference. It follows the Kairix
"Inference Matrix" pattern that decouples statistical methods from compute backends.

The Dispatcher Pattern:
    SurvivalTester detects input DataFrame type (pandas or Spark) and routes
    execution to the appropriate implementation strategy.
"""

from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass
import math

import pandas as pd
import numpy as np
from scipy import stats
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from kairix.core.validation import validate_input_schema


@dataclass
class LogRankResult:
    """Result of a log-rank test comparing two survival curves.
    
    Attributes:
        test_statistic: The chi-square test statistic.
        p_value: The p-value for the test (two-sided).
        degrees_of_freedom: Degrees of freedom (always 1 for comparing 2 groups).
        significant: Whether the difference is statistically significant at alpha=0.05.
        group_1_name: Name of the first group.
        group_2_name: Name of the second group.
    """
    test_statistic: float
    p_value: float
    degrees_of_freedom: int = 1
    significant: bool = False
    group_1_name: str = "control"
    group_2_name: str = "treatment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "significant": self.significant,
            "group_1_name": self.group_1_name,
            "group_2_name": self.group_2_name,
        }


@dataclass
class BayesianResult:
    """Result of Bayesian survival comparison.
    
    Attributes:
        prob_superiority: Probability that group 2 has better survival than group 1.
        credible_intervals: Dictionary containing credible intervals for survival at t_end.
        posterior_samples: Number of posterior samples used.
    """
    prob_superiority: float
    credible_intervals: Dict[str, Tuple[float, float]]
    posterior_samples: int = 2000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "prob_superiority": self.prob_superiority,
            "credible_intervals": {k: list(v) for k, v in self.credible_intervals.items()},
            "posterior_samples": self.posterior_samples,
        }


@dataclass
class CriticalTimeResult:
    """Result of critical time analysis for survival curve comparison.
    
    This identifies when the largest difference between two survival curves occurs,
    which is useful for understanding when treatment effects are most pronounced.
    
    Attributes:
        cumulative_max_time: Time when cumulative difference between groups is maximized.
        cumulative_max_difference: Value of maximum cumulative difference (O - E).
        cumulative_max_direction: Direction of maximum difference ("treatment_better" or "control_better").
        per_timepoint_max_time: Time when per-timepoint difference is maximized.
        per_timepoint_max_difference: Value of maximum per-timepoint difference.
        per_timepoint_max_direction: Direction of max per-timepoint difference.
        timeline: List of all time points analyzed.
        cumulative_differences: Cumulative (O - E) at each time point.
        per_timepoint_differences: Per-timepoint (O - E) at each time point.
        group_1_name: Name of the first group (control).
        group_2_name: Name of the second group (treatment).
    """
    cumulative_max_time: float
    cumulative_max_difference: float
    cumulative_max_direction: str
    per_timepoint_max_time: float
    per_timepoint_max_difference: float
    per_timepoint_max_direction: str
    timeline: List[float]
    cumulative_differences: List[float]
    per_timepoint_differences: List[float]
    group_1_name: str = "control"
    group_2_name: str = "treatment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "cumulative_max_time": self.cumulative_max_time,
            "cumulative_max_difference": self.cumulative_max_difference,
            "cumulative_max_direction": self.cumulative_max_direction,
            "per_timepoint_max_time": self.per_timepoint_max_time,
            "per_timepoint_max_difference": self.per_timepoint_max_difference,
            "per_timepoint_max_direction": self.per_timepoint_max_direction,
            "timeline": self.timeline,
            "cumulative_differences": self.cumulative_differences,
            "per_timepoint_differences": self.per_timepoint_differences,
            "group_1_name": self.group_1_name,
            "group_2_name": self.group_2_name,
        }


class SurvivalTester:
    """A/B Testing for Survival Analysis.
    
    This class implements the log-rank test and Bayesian comparison to evaluate
    differences between two survival curves. It automatically dispatches to the
    appropriate implementation based on the input DataFrame type.
    
    The Log-Rank Test:
        Tests the null hypothesis that there is no difference between the
        survival curves of two groups. Uses the Z-score formulation:
        
        Z = (sum(O1j - E1j)) / sqrt(sum(V1j))
        
        Where:
        - O1j = Observed events in group 1 at time j
        - E1j = Expected events in group 1 at time j
        - V1j = Variance at time j
        
    The Bayesian Comparison:
        Uses a Beta-Binomial conjugate model to compute the probability that
        one group has better survival than another.
        
    Example:
        >>> from kairix.survival import SurvivalTester
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'tenure': [5, 6, 7, 8, 10, 12, 15, 16],
        ...     'churned': [1, 0, 1, 0, 1, 1, 1, 0],
        ...     'variant': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        ... })
        >>> tester = SurvivalTester()
        >>> result = tester.run_test(df, 'tenure', 'churned', group_col='variant')
        >>> print(f"P-Value: {result['p_value']:.4f}")
    """
    
    def __init__(self) -> None:
        """Initialize the SurvivalTester."""
        self._is_fitted: bool = False
        self._last_result: Optional[Union[LogRankResult, BayesianResult]] = None
        self._stats: Dict[str, Any] = {}
        self._group_1_name: str = "control"
        self._group_2_name: str = "treatment"
    
    def run_test(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        duration_col: str,
        event_col: str,
        group_col: str = "variant",
        test_type: str = "log_rank",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Run a statistical test comparing survival curves between two groups.
        
        Args:
            df: Input DataFrame (pandas or PySpark) containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            group_col: Name of the column containing group/variant labels.
            test_type: Type of test to run ('log_rank' or 'wilcoxon'). Default is 'log_rank'.
            alpha: Significance level. Default is 0.05.
            
        Returns:
            Dictionary containing:
                - test_statistic: The chi-square test statistic.
                - p_value: The p-value for the test.
                - significant: Whether the difference is statistically significant.
                - summary: Group-level statistics (n_samples, events, etc.).
                
        Raises:
            ValueError: If the group column does not have exactly 2 unique values.
            TypeError: If df is not a pandas or PySpark DataFrame.
        """
        # Get unique groups
        if isinstance(df, SparkDataFrame):
            unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        else:
            unique_groups = df[group_col].unique().tolist()
        
        if len(unique_groups) != 2:
            raise ValueError(
                f"Group column must have exactly 2 unique values, "
                f"found {len(unique_groups)}: {unique_groups}"
            )
        
        # Set group names (first is control, second is treatment)
        self._group_1_name = str(unique_groups[0])
        self._group_2_name = str(unique_groups[1])
        
        # Dispatch to appropriate implementation
        if isinstance(df, SparkDataFrame):
            result = self._run_test_spark(
                df, duration_col, event_col, group_col, test_type, alpha
            )
        else:
            result = self._run_test_pandas(
                df, duration_col, event_col, group_col, test_type, alpha
            )
        
        self._last_result = result
        self._is_fitted = True
        
        return result.to_dict()
    
    def _run_test_pandas(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        test_type: str,
        alpha: float,
    ) -> LogRankResult:
        """Run log-rank test using pandas (local implementation)."""
        group_1_data = df[df[group_col] == self._group_1_name]
        group_2_data = df[df[group_col] == self._group_2_name]
        
        # Extract arrays
        d1 = group_1_data[duration_col].values
        e1 = group_1_data[event_col].values
        d2 = group_2_data[duration_col].values
        e2 = group_2_data[event_col].values
        
        # Get unique time points across both groups
        all_times = np.concatenate([d1, d2])
        unique_times = np.unique(all_times)
        
        # Calculate observed and expected events, and variance
        O1_sum = 0.0  # Sum of observed events in group 1
        E1_sum = 0.0  # Sum of expected events in group 1
        V1_sum = 0.0  # Sum of variance
        
        for t in unique_times:
            # Events at time t for each group
            o1 = e1[d1 == t].sum()  # Observed in group 1
            o2 = e2[d2 == t].sum()  # Observed in group 2
            Oj = o1 + o2  # Total observed events at time t
            
            # Risk sets at time t
            n1 = (d1 >= t).sum()  # At risk in group 1
            n2 = (d2 >= t).sum()  # At risk in group 2
            Nj = n1 + n2  # Total at risk
            
            if Nj > 0 and Oj > 0:
                # Expected events in group 1 under null hypothesis
                E1j = Oj * (n1 / Nj)
                
                # Variance ( Greenwood's formula variant for two-sample test)
                if Nj > 1:
                    V1j = (n1 * n2 * Oj * (Nj - Oj)) / (Nj * Nj * (Nj - 1))
                else:
                    V1j = 0.0
                
                O1_sum += o1
                E1_sum += E1j
                V1_sum += V1j
        
        # Calculate Z-score
        if V1_sum > 0:
            Z = (O1_sum - E1_sum) / math.sqrt(V1_sum)
            chi_square = Z * Z  # Chi-square with 1 df
        else:
            chi_square = 0.0
        
        # Calculate p-value (two-sided)
        p_value = 1.0 - stats.chi2.cdf(chi_square, df=1)
        
        # Store stats
        self._stats = {
            "n_group_1": len(group_1_data),
            "n_group_2": len(group_2_data),
            "events_group_1": e1.sum(),
            "events_group_2": e2.sum(),
            "observed_1": O1_sum,
            "expected_1": E1_sum,
            "variance": V1_sum,
        }
        
        return LogRankResult(
            test_statistic=chi_square,
            p_value=p_value,
            degrees_of_freedom=1,
            significant=p_value < alpha,
            group_1_name=self._group_1_name,
            group_2_name=self._group_2_name,
        )
    
    def _run_test_spark(
        self,
        df: SparkDataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        test_type: str,
        alpha: float,
    ) -> LogRankResult:
        """Run log-rank test using PySpark (distributed implementation)."""
        # Validate input schema
        validate_input_schema(df, duration_col, event_col)
        
        # Get global statistics in a single pass
        stats_row = df.groupBy(group_col).agg(
            F.count(F.lit(1)).alias("n_samples"),
            F.sum(event_col).alias("total_events"),
            F.min(duration_col).alias("min_duration"),
            F.max(duration_col).alias("max_duration"),
        ).collect()
        
        # Extract stats for each group
        group_1_stats = None
        group_2_stats = None
        for row in stats_row:
            if row[group_col] == self._group_1_name:
                group_1_stats = row
            else:
                group_2_stats = row
        
        n_1 = group_1_stats["n_samples"]
        n_2 = group_2_stats["n_samples"]
        events_1 = group_1_stats["total_events"]
        events_2 = group_2_stats["total_events"]
        min_dur = min(group_1_stats["min_duration"], group_2_stats["min_duration"])
        max_dur = max(group_1_stats["max_duration"], group_2_stats["max_duration"])
        
        # Calculate resolution for discretization
        duration_range = max_dur - min_dur
        resolution = 1.0 if duration_range == 0 else duration_range / 10000
        
        # Step 1: Global Discretization (Map)
        discretized_df = df.withColumn(
            "duration_bin",
            (F.floor(F.col(duration_col) / resolution) * resolution).cast("double"),
        )
        
        # Step 2: Aggregation (Reduce) - Group by bin AND group
        agg_df = discretized_df.groupBy("duration_bin", group_col).agg(
            F.sum(event_col).alias("d_t"),  # Events at this time
            F.count(F.lit(1)).alias("total_observed"),  # Total at risk at this time
        )
        
        # Step 3: Risk Set Calculation (Window)
        window_spec = Window.partitionBy(group_col).orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        
        agg_df = agg_df.withColumn(
            "cumulative_observed",
            F.sum("total_observed").over(window_spec),
        )
        
        # Calculate risk set for each group
        if row[group_col] == self._group_1_name:
            agg_df = agg_df.withColumn(
                "n_at_risk",
                F.lit(n_1) - F.col("cumulative_observed") + F.col("total_observed")
            )
        else:
            agg_df = agg_df.withColumn(
                "n_at_risk",
                F.lit(n_2) - F.col("cumulative_observed") + F.col("total_observed")
            )
        
        # Step 4: Pivot & Statistic (Align groups on same time bin)
        pivot_df = agg_df.groupBy("duration_bin").pivot(
            group_col, values=[self._group_1_name, self._group_2_name]
        ).agg(
            F.first("d_t").alias("d_t"),
            F.first("n_at_risk").alias("n_at_risk"),
        )
        
        pivot_df = pivot_df.fillna(0)
        
        # Calculate observed and expected events, and variance
        pivot_df = pivot_df.withColumn(
            "Oj",
            F.col(f"d_t_{self._group_1_name}") + F.col(f"d_t_{self._group_2_name}")
        )
        pivot_df = pivot_df.withColumn(
            "Nj",
            F.col(f"n_at_risk_{self._group_1_name}") + F.col(f"n_at_risk_{self._group_2_name}")
        )
        
        # Expected events for group 1
        pivot_df = pivot_df.withColumn(
            "E1j",
            F.when(F.col("Nj") > 0,
                   F.col("Oj") * F.col(f"n_at_risk_{self._group_1_name}") / F.col("Nj"))
            .otherwise(0)
        )
        
        # Variance for group 1
        pivot_df = pivot_df.withColumn(
            "V1j",
            F.when(F.col("Nj") > 1,
                   (F.col(f"n_at_risk_{self._group_1_name}") * 
                    F.col(f"n_at_risk_{self._group_2_name}") * 
                    F.col("Oj") * 
                    (F.col("Nj") - F.col("Oj"))) / 
                   (F.col("Nj") * F.col("Nj") * (F.col("Nj") - 1)))
            .otherwise(0)
        )
        
        # Collect results (small dataset)
        results = pivot_df.select(
            f"d_t_{self._group_1_name}",
            "E1j", "V1j"
        ).collect()
        
        O1_sum = 0.0
        E1_sum = 0.0
        V1_sum = 0.0
        
        for row in results:
            O1_sum += row[f"d_t_{self._group_1_name}"] or 0
            E1_sum += row["E1j"] or 0
            V1_sum += row["V1j"] or 0
        
        # Calculate Z-score and chi-square
        if V1_sum > 0:
            Z = (O1_sum - E1_sum) / math.sqrt(V1_sum)
            chi_square = Z * Z
        else:
            chi_square = 0.0
        
        p_value = 1.0 - stats.chi2.cdf(chi_square, df=1)
        
        # Store stats
        self._stats = {
            "n_group_1": n_1,
            "n_group_2": n_2,
            "events_group_1": events_1,
            "events_group_2": events_2,
            "observed_1": O1_sum,
            "expected_1": E1_sum,
            "variance": V1_sum,
        }
        
        return LogRankResult(
            test_statistic=chi_square,
            p_value=p_value,
            degrees_of_freedom=1,
            significant=p_value < alpha,
            group_1_name=self._group_1_name,
            group_2_name=self._group_2_name,
        )
    
    def run_bayesian_comparison(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        duration_col: str,
        event_col: str,
        group_col: str = "variant",
        n_samples: int = 2000,
    ) -> Dict[str, Any]:
        """Run Bayesian survival comparison using Beta-Binomial conjugate model.
        
        This method computes the probability that one group has better survival
        than another by simulating survival curves from posterior distributions.
        
        Args:
            df: Input DataFrame (pandas or PySpark) containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            group_col: Name of the column containing group/variant labels.
            n_samples: Number of posterior samples for Monte Carlo estimation.
            
        Returns:
            Dictionary containing:
                - prob_superiority: Probability that group 2 > group 1.
                - credible_intervals: 95% CI for survival at final time.
                - summary: Group-level statistics.
        """
        # Get unique groups
        if isinstance(df, SparkDataFrame):
            unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        else:
            unique_groups = df[group_col].unique().tolist()
        
        if len(unique_groups) != 2:
            raise ValueError(
                f"Group column must have exactly 2 unique values, "
                f"found {len(unique_groups)}: {unique_groups}"
            )
        
        group_1_label = str(unique_groups[0])
        group_2_label = str(unique_groups[1])
        
        # Split data by group
        if isinstance(df, SparkDataFrame):
            group_1_data = df.filter(F.col(group_col) == group_1_label).toPandas()
            group_2_data = df.filter(F.col(group_col) == group_2_label).toPandas()
        else:
            group_1_data = df[df[group_col] == group_1_label]
            group_2_data = df[df[group_col] == group_2_label]
        
        # Calculate hazard rates per time bin using KM-like approach
        # For simplicity, we use the overall event rate per group
        n1 = len(group_1_data)
        n2 = len(group_2_data)
        events1 = group_1_data[event_col].sum()
        events2 = group_2_data[event_col].sum()
        
        # Posterior hazard rates (Beta-Binomial conjugate)
        # Prior: Beta(1, 1) - uninformative
        # Posterior: Beta(1 + events, 1 + (n - events))
        alpha1_prior, beta1_prior = 1, 1
        alpha2_prior, beta2_prior = 1, 1
        
        alpha1_post = alpha1_prior + events1
        beta1_post = beta1_prior + (n1 - events1)
        alpha2_post = alpha2_prior + events2
        beta2_post = beta2_prior + (n2 - events2)
        
        # Sample from posterior hazard distributions
        np.random.seed(42)
        hazard_samples1 = np.random.beta(alpha1_post, beta1_post, n_samples)
        hazard_samples2 = np.random.beta(alpha2_post, beta2_post, n_samples)
        
        # Survival at final time (simplified: S = (1 - h)^t where t is max duration)
        max_duration = max(group_1_data[duration_col].max(), group_2_data[duration_col].max())
        
        survival1 = (1 - hazard_samples1) ** max_duration
        survival2 = (1 - hazard_samples2) ** max_duration
        
        # Calculate probability that group 2 has better survival
        prob_superiority = (survival2 > survival1).mean()
        
        # Calculate credible intervals (95%)
        ci1 = np.percentile(survival1, [2.5, 97.5])
        ci2 = np.percentile(survival2, [2.5, 97.5])
        
        result = BayesianResult(
            prob_superiority=prob_superiority,
            credible_intervals={
                f"{group_1_label}_survival": (float(ci1[0]), float(ci1[1])),
                f"{group_2_label}_survival": (float(ci2[0]), float(ci2[1])),
            },
            posterior_samples=n_samples,
        )
        
        # Store summary
        self._stats = {
            "n_group_1": n1,
            "n_group_2": n2,
            "events_group_1": events1,
            "events_group_2": events2,
            "max_duration": max_duration,
        }
        
        self._last_result = result
        self._is_fitted = True
        
        return result.to_dict()
    
    def find_critical_time(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        duration_col: str,
        event_col: str,
        group_col: str = "variant",
    ) -> CriticalTimeResult:
        """Find the critical time when the largest difference between groups occurs.
        
        This method identifies two key time points:
        1. Cumulative max: When the running cumulative difference (O - E) is maximized
        2. Per-timepoint max: When the individual time point contribution is maximized
        
        Args:
            df: Input DataFrame (pandas or PySpark) containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            group_col: Name of the column containing group/variant labels.
            
        Returns:
            CriticalTimeResult containing:
                - cumulative_max_time: Time when groups diverge most
                - per_timepoint_max_time: Time of largest single difference
                - timeline and difference arrays for visualization
        
        Raises:
            ValueError: If the group column does not have exactly 2 unique values.
        """
        # Get unique groups
        if isinstance(df, SparkDataFrame):
            unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        else:
            unique_groups = df[group_col].unique().tolist()
        
        if len(unique_groups) != 2:
            raise ValueError(
                f"Group column must have exactly 2 unique values, "
                f"found {len(unique_groups)}: {unique_groups}"
            )
        
        # Set group names
        group_1_name = str(unique_groups[0])
        group_2_name = str(unique_groups[1])
        
        # Dispatch to appropriate implementation
        if isinstance(df, SparkDataFrame):
            return self._find_critical_time_spark(
                df, duration_col, event_col, group_col, group_1_name, group_2_name
            )
        else:
            return self._find_critical_time_pandas(
                df, duration_col, event_col, group_col, group_1_name, group_2_name
            )
    
    def _find_critical_time_pandas(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        group_1_name: str,
        group_2_name: str,
    ) -> CriticalTimeResult:
        """Find critical time using pandas (local implementation)."""
        group_1_data = df[df[group_col] == group_1_name]
        group_2_data = df[df[group_col] == group_2_name]
        
        # Extract arrays
        d1 = group_1_data[duration_col].values
        e1 = group_1_data[event_col].values
        d2 = group_2_data[duration_col].values
        e2 = group_2_data[event_col].values
        
        # Get unique time points across both groups
        all_times = np.concatenate([d1, d2])
        unique_times = np.sort(np.unique(all_times))
        
        # Track per-timepoint and cumulative differences
        per_timepoint_diffs = []
        cumulative_diffs = []
        timeline = []
        cumulative_sum = 0.0
        
        for t in unique_times:
            # Events at time t for each group
            o1 = e1[d1 == t].sum()  # Observed in group 1
            o2 = e2[d2 == t].sum()  # Observed in group 2
            Oj = o1 + o2  # Total observed events at time t
            
            # Risk sets at time t
            n1 = (d1 >= t).sum()  # At risk in group 1
            n2 = (d2 >= t).sum()  # At risk in group 2
            Nj = n1 + n2  # Total at risk
            
            if Nj > 0 and Oj > 0:
                # Expected events in group 1 under null hypothesis
                E1j = Oj * (n1 / Nj)
                
                # Per-timepoint difference (O1 - E1)
                per_timepoint_diff = o1 - E1j
                
                timeline.append(float(t))
                per_timepoint_diffs.append(float(per_timepoint_diff))
                cumulative_sum += per_timepoint_diff
                cumulative_diffs.append(float(cumulative_sum))
        
        # Find cumulative max
        cum_max_idx = np.argmax(np.abs(cumulative_diffs))
        cumulative_max_time = timeline[cum_max_idx] if timeline else 0.0
        cumulative_max_diff = cumulative_diffs[cum_max_idx] if cumulative_diffs else 0.0
        cumulative_max_direction = (
            f"{group_2_name}_better" if cumulative_max_diff < 0 else f"{group_1_name}_better"
        )
        
        # Find per-timepoint max
        pt_max_idx = np.argmax(np.abs(per_timepoint_diffs))
        per_timepoint_max_time = timeline[pt_max_idx] if timeline else 0.0
        per_timepoint_max_diff = per_timepoint_diffs[pt_max_idx] if per_timepoint_diffs else 0.0
        per_timepoint_max_direction = (
            f"{group_2_name}_better" if per_timepoint_max_diff < 0 else f"{group_1_name}_better"
        )
        
        # Store stats
        self._stats = {
            "n_group_1": len(group_1_data),
            "n_group_2": len(group_2_data),
            "events_group_1": int(e1.sum()),
            "events_group_2": int(e2.sum()),
        }
        
        return CriticalTimeResult(
            cumulative_max_time=cumulative_max_time,
            cumulative_max_difference=cumulative_max_diff,
            cumulative_max_direction=cumulative_max_direction,
            per_timepoint_max_time=per_timepoint_max_time,
            per_timepoint_max_difference=per_timepoint_max_diff,
            per_timepoint_max_direction=per_timepoint_max_direction,
            timeline=timeline,
            cumulative_differences=cumulative_diffs,
            per_timepoint_differences=per_timepoint_diffs,
            group_1_name=group_1_name,
            group_2_name=group_2_name,
        )
    
    def _find_critical_time_spark(
        self,
        df: SparkDataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
        group_1_name: str,
        group_2_name: str,
    ) -> CriticalTimeResult:
        """Find critical time using PySpark (distributed implementation)."""
        # Get global statistics
        stats_row = df.groupBy(group_col).agg(
            F.count(F.lit(1)).alias("n_samples"),
            F.sum(event_col).alias("total_events"),
            F.min(duration_col).alias("min_duration"),
            F.max(duration_col).alias("max_duration"),
        ).collect()
        
        # Extract stats for each group
        group_1_stats = None
        group_2_stats = None
        for row in stats_row:
            if row[group_col] == group_1_name:
                group_1_stats = row
            else:
                group_2_stats = row
        
        n_1 = group_1_stats["n_samples"]
        n_2 = group_2_stats["n_samples"]
        events_1 = group_1_stats["total_events"]
        events_2 = group_2_stats["total_events"]
        min_dur = min(group_1_stats["min_duration"], group_2_stats["min_duration"])
        max_dur = max(group_1_stats["max_duration"], group_2_stats["max_duration"])
        
        # Calculate resolution for discretization
        duration_range = max_dur - min_dur
        resolution = 1.0 if duration_range == 0 else duration_range / 10000
        
        # Step 1: Global Discretization
        discretized_df = df.withColumn(
            "duration_bin",
            (F.floor(F.col(duration_col) / resolution) * resolution).cast("double"),
        )
        
        # Step 2: Aggregation - Group by bin AND group
        agg_df = discretized_df.groupBy("duration_bin", group_col).agg(
            F.sum(event_col).alias("d_t"),
            F.count(F.lit(1)).alias("total_observed"),
        )
        
        # Step 3: Risk Set Calculation
        window_spec = Window.partitionBy(group_col).orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        
        agg_df = agg_df.withColumn(
            "cumulative_observed",
            F.sum("total_observed").over(window_spec),
        )
        
        # Calculate risk set for each group
        agg_df = agg_df.withColumn(
            "n_at_risk",
            F.when(F.col(group_col) == group_1_name,
                   F.lit(n_1) - F.col("cumulative_observed") + F.col("total_observed"))
            .otherwise(F.lit(n_2) - F.col("cumulative_observed") + F.col("total_observed"))
        )
        
        # Step 4: Pivot & Calculate per-timepoint difference
        pivot_df = agg_df.groupBy("duration_bin").pivot(
            group_col, values=[group_1_name, group_2_name]
        ).agg(
            F.first("d_t").alias("d_t"),
            F.first("n_at_risk").alias("n_at_risk"),
        )
        
        pivot_df = pivot_df.fillna(0)
        
        # Calculate Oj, Nj, and per-timepoint difference (O1 - E1)
        pivot_df = pivot_df.withColumn(
            "Oj",
            F.col(f"d_t_{group_1_name}") + F.col(f"d_t_{group_2_name}")
        )
        pivot_df = pivot_df.withColumn(
            "Nj",
            F.col(f"n_at_risk_{group_1_name}") + F.col(f"n_at_risk_{group_2_name}")
        )
        pivot_df = pivot_df.withColumn(
            "E1j",
            F.when(F.col("Nj") > 0,
                   F.col("Oj") * F.col(f"n_at_risk_{group_1_name}") / F.col("Nj"))
            .otherwise(0)
        )
        pivot_df = pivot_df.withColumn(
            "per_timepoint_diff",
            F.col(f"d_t_{group_1_name}") - F.col("E1j")
        )
        
        # Collect results
        results = pivot_df.select(
            "duration_bin",
            "per_timepoint_diff"
        ).orderBy("duration_bin").collect()
        
        # Calculate cumulative differences and find max
        timeline = []
        per_timepoint_diffs = []
        cumulative_diffs = []
        cumulative_sum = 0.0
        
        for row in results:
            timeline.append(float(row["duration_bin"]))
            diff = float(row["per_timepoint_diff"]) if row["per_timepoint_diff"] else 0.0
            per_timepoint_diffs.append(diff)
            cumulative_sum += diff
            cumulative_diffs.append(cumulative_sum)
        
        # Find cumulative max
        cum_max_idx = np.argmax(np.abs(cumulative_diffs)) if cumulative_diffs else 0
        cumulative_max_time = timeline[cum_max_idx] if timeline else 0.0
        cumulative_max_diff = cumulative_diffs[cum_max_idx] if cumulative_diffs else 0.0
        cumulative_max_direction = (
            f"{group_2_name}_better" if cumulative_max_diff < 0 else f"{group_1_name}_better"
        )
        
        # Find per-timepoint max
        pt_max_idx = np.argmax(np.abs(per_timepoint_diffs)) if per_timepoint_diffs else 0
        per_timepoint_max_time = timeline[pt_max_idx] if timeline else 0.0
        per_timepoint_max_diff = per_timepoint_diffs[pt_max_idx] if per_timepoint_diffs else 0.0
        per_timepoint_max_direction = (
            f"{group_2_name}_better" if per_timepoint_max_diff < 0 else f"{group_1_name}_better"
        )
        
        # Store stats
        self._stats = {
            "n_group_1": n_1,
            "n_group_2": n_2,
            "events_group_1": int(events_1),
            "events_group_2": int(events_2),
        }
        
        return CriticalTimeResult(
            cumulative_max_time=cumulative_max_time,
            cumulative_max_difference=cumulative_max_diff,
            cumulative_max_direction=cumulative_max_direction,
            per_timepoint_max_time=per_timepoint_max_time,
            per_timepoint_max_difference=per_timepoint_max_diff,
            per_timepoint_max_direction=per_timepoint_max_direction,
            timeline=timeline,
            cumulative_differences=cumulative_diffs,
            per_timepoint_differences=per_timepoint_diffs,
            group_1_name=group_1_name,
            group_2_name=group_2_name,
        )
    
    def run_test_with_critical_time(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        duration_col: str,
        event_col: str,
        group_col: str = "variant",
        test_type: str = "log_rank",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Run log-rank test AND critical time analysis in a single call.
        
        This is a convenience method that combines run_test() and find_critical_time()
        to provide both statistical significance and temporal insight in one call.
        
        Args:
            df: Input DataFrame (pandas or PySpark) containing survival data.
            duration_col: Name of the column containing duration/time-to-event.
            event_col: Name of the column containing event indicator (0 or 1).
            group_col: Name of the column containing group/variant labels.
            test_type: Type of test to run ('log_rank' or 'wilcoxon'). Default is 'log_rank'.
            alpha: Significance level. Default is 0.05.
            
        Returns:
            Dictionary containing:
                - log_rank_result: Standard log-rank test results (test_statistic, p_value, significant)
                - critical_time: CriticalTimeResult with both metrics
                - summary: Group-level statistics (n_samples, events, etc.)
        """
        # Run log-rank test first
        log_rank_result = self.run_test(
            df, duration_col, event_col, group_col, test_type, alpha
        )
        
        # Then find critical time
        critical_time_result = self.find_critical_time(
            df, duration_col, event_col, group_col
        )
        
        return {
            "log_rank_result": {
                "test_statistic": log_rank_result["test_statistic"],
                "p_value": log_rank_result["p_value"],
                "significant": log_rank_result["significant"],
                "degrees_of_freedom": log_rank_result["degrees_of_freedom"],
            },
            "critical_time": critical_time_result.to_dict(),
            "summary": self.summary(),
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the test results.
        
        Returns:
            Dictionary containing test statistics and group metrics.
        """
        if not self._is_fitted:
            raise ValueError("Test has not been run. Call run_test() first.")
        
        if isinstance(self._last_result, LogRankResult):
            return {
                **self._stats,
                "test_statistic": self._last_result.test_statistic,
                "p_value": self._last_result.p_value,
                "significant": self._last_result.significant,
            }
        else:
            return {
                **self._stats,
                "prob_superiority": self._last_result.prob_superiority,
            }
    
    @property
    def is_fitted(self) -> bool:
        """Whether a test has been run."""
        return self._is_fitted
    
    @property
    def group_names(self) -> Tuple[str, str]:
        """Get the names of the two groups being compared."""
        return (self._group_1_name, self._group_2_name)
