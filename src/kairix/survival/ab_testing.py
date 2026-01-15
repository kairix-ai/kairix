"""A/B Testing for Survival Analysis (Fixed Directionality & Scale).

This module implements the "Inference Matrix" pattern:
1. Dispatcher: Routes to Pandas (Local) or Spark (Distributed).
2. Logic: Uses "Excess Events" (Observed - Expected) to track treatment effect.
    - Positive Value: Treatment is WORSE (dying faster than expected).
    - Negative Value: Treatment is BETTER (surviving longer than expected).
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
        group_1_name: Name of the first group (Control).
        group_2_name: Name of the second group (Treatment).
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
    """Result of critical time analysis tracking Treatment Deviation.
    
    Metric: (Observed_Group2 - Expected_Group2)
    - Positive Value (> 0): Group 2 has EXCESS events (Performing Worse).
    - Negative Value (< 0): Group 2 has FEWER events (Performing Better).
    """
    cumulative_max_time: float
    cumulative_max_excess: float  
    crossover_time: Optional[float] 
    timeline: List[float]
    cumulative_excess_events: List[float]
    per_timepoint_excess: List[float]
    group_1_name: str = "control"
    group_2_name: str = "treatment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "cumulative_max_time": self.cumulative_max_time,
            "cumulative_max_excess": self.cumulative_max_excess,
            "crossover_time": self.crossover_time,
            "timeline": self.timeline,
            "cumulative_excess_events": self.cumulative_excess_events,
            "per_timepoint_excess": self.per_timepoint_excess,
            "group_1_name": self.group_1_name,
            "group_2_name": self.group_2_name,
        }


class SurvivalTester:
    """A/B Testing for Survival Analysis.
    
    This class implements the log-rank test and Bayesian comparison to evaluate
    differences between two survival curves. It automatically dispatches to the
    appropriate implementation based on the input DataFrame type.
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
        bins: int = 100000, 
    ) -> Dict[str, Any]:
        """Run a statistical test comparing survival curves between two groups."""
        # Get unique groups
        if isinstance(df, SparkDataFrame):
            unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        else:
            unique_groups = df[group_col].unique().tolist()
        
        unique_groups.sort() # Ensure consistency
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
                df, duration_col, event_col, group_col, test_type, alpha, bins
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
        O1_sum = 0.0
        E1_sum = 0.0
        V1_sum = 0.0
        
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
                
                # Variance (Greenwood's formula variant for two-sample test)
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
            chi_square = Z * Z
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
        bins: int = 100000,
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
        group_1_stats = next((r for r in stats_row if r[group_col] == self._group_1_name), None)
        group_2_stats = next((r for r in stats_row if r[group_col] == self._group_2_name), None)
        
        if not group_1_stats or not group_2_stats:
             raise ValueError("One or both groups have no data.")

        n_1 = group_1_stats["n_samples"]
        n_2 = group_2_stats["n_samples"]
        events_1 = group_1_stats["total_events"]
        events_2 = group_2_stats["total_events"]
        min_dur = min(group_1_stats["min_duration"], group_2_stats["min_duration"])
        max_dur = max(group_1_stats["max_duration"], group_2_stats["max_duration"])
        
        # Calculate resolution for discretization (Scale Fix)
        duration_range = max_dur - min_dur
        resolution = 1.0 if duration_range == 0 else duration_range / bins
        
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
        
        # === EARLY PIVOT ===
        # Pivot counts first to align time bins. Fill missing activity with 0.
        pivot_df = agg_df.groupBy("duration_bin").pivot(
            group_col, values=[self._group_1_name, self._group_2_name]
        ).agg(
            F.first("d_t").alias("d_t"),
            F.first("total_observed").alias("total_observed")
        ).fillna(0)
        
        # Step 3: Risk Set Calculation (Window) on ALIGNED data
        window_spec = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        
        # Group 1 Risk Set
        pivot_df = pivot_df.withColumn(
            f"{self._group_1_name}_cumulative",
            F.sum(f"{self._group_1_name}_total_observed").over(window_spec)
        )
        pivot_df = pivot_df.withColumn(
            f"{self._group_1_name}_n_at_risk",
            F.lit(n_1) - F.col(f"{self._group_1_name}_cumulative") + F.col(f"{self._group_1_name}_total_observed")
        )
        
        # Group 2 Risk Set
        pivot_df = pivot_df.withColumn(
            f"{self._group_2_name}_cumulative",
            F.sum(f"{self._group_2_name}_total_observed").over(window_spec)
        )
        pivot_df = pivot_df.withColumn(
            f"{self._group_2_name}_n_at_risk",
            F.lit(n_2) - F.col(f"{self._group_2_name}_cumulative") + F.col(f"{self._group_2_name}_total_observed")
        )
        
        # Step 4: Statistic Calculation
        col_d_t_1 = F.col(f"{self._group_1_name}_d_t")
        col_d_t_2 = F.col(f"{self._group_2_name}_d_t")
        col_n_1 = F.col(f"{self._group_1_name}_n_at_risk")
        col_n_2 = F.col(f"{self._group_2_name}_n_at_risk")

        pivot_df = pivot_df.withColumn("Oj", col_d_t_1 + col_d_t_2)
        pivot_df = pivot_df.withColumn("Nj", col_n_1 + col_n_2)
        
        # Expected events for group 1
        pivot_df = pivot_df.withColumn(
            "E1j",
            F.when(F.col("Nj") > 0,
                   F.col("Oj") * col_n_1 / F.col("Nj"))
            .otherwise(0)
        )
        
        # Variance for group 1
        pivot_df = pivot_df.withColumn(
            "V1j",
            F.when(F.col("Nj") > 1,
                   (col_n_1 * col_n_2 * F.col("Oj") * (F.col("Nj") - F.col("Oj"))) / 
                   (F.col("Nj") * F.col("Nj") * (F.col("Nj") - 1)))
            .otherwise(0)
        )
        
        # Collect results
        results = pivot_df.select(
            f"{self._group_1_name}_d_t",
            "E1j", "V1j"
        ).collect()
        
        O1_sum = 0.0
        E1_sum = 0.0
        V1_sum = 0.0
        
        for row in results:
            O1_sum += row[f"{self._group_1_name}_d_t"] or 0
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
        """Run Bayesian survival comparison using Beta-Binomial conjugate model."""
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
        n1 = len(group_1_data)
        n2 = len(group_2_data)
        events1 = group_1_data[event_col].sum()
        events2 = group_2_data[event_col].sum()
        
        # Posterior hazard rates (Beta-Binomial conjugate)
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
        
        # Survival at final time (simplified)
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
        bins: int = 100000,
    ) -> CriticalTimeResult:
        """Find the critical time when the largest difference between groups occurs."""
        # Get unique groups
        if isinstance(df, SparkDataFrame):
            unique_groups = [row[0] for row in df.select(group_col).distinct().collect()]
        else:
            unique_groups = df[group_col].unique().tolist()
        
        unique_groups.sort()
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
                df, duration_col, event_col, group_col, group_1_name, group_2_name, bins
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
        
        timeline = []
        excess_events = []
        cum_excess = []
        curr_cum = 0.0
        
        for t in unique_times:
            # Events at time t for each group
            o1 = e1[d1 == t].sum()  # Observed in group 1
            o2 = e2[d2 == t].sum()  # Observed in group 2
            
            # Risk sets at time t
            n1 = (d1 >= t).sum()  # At risk in group 1
            n2 = (d2 >= t).sum()  # At risk in group 2
            
            if (n1+n2) > 0 and (o1+o2) > 0:
                # Expected events in group 2 (Treatment) under null hypothesis
                # E2 = O_total * (N2 / N_total)
                E2 = (o1 + o2) * (n2 / (n1 + n2))
                
                # Excess = Observed Treatment - Expected Treatment
                # Pos > 0: Treatment dying FASTER (Bad)
                # Neg < 0: Treatment dying SLOWER (Good)
                diff = o2 - E2
                
                curr_cum += diff
                timeline.append(float(t))
                excess_events.append(float(diff))
                cum_excess.append(float(curr_cum))
        
        # Crossover Detection (When trend flips from Bad to Good or vice versa)
        crossover = None
        for i in range(1, len(cum_excess)):
            if cum_excess[i-1] > 0 and cum_excess[i] < 0:
                crossover = timeline[i]
                break
        
        max_excess = max(cum_excess) if cum_excess else 0.0
        max_time_idx = np.argmax(cum_excess) if cum_excess else 0
        
        return CriticalTimeResult(
            cumulative_max_time=timeline[max_time_idx] if timeline else 0.0,
            cumulative_max_excess=max_excess,
            crossover_time=crossover,
            timeline=timeline,
            cumulative_excess_events=cum_excess,
            per_timepoint_excess=excess_events,
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
        bins: int = 100000,
    ) -> CriticalTimeResult:
        """Find critical time using PySpark with EARLY PIVOT stability."""
        
        # --- 1. Global Stats ---
        stats_row = df.groupBy(group_col).agg(
            F.count(F.lit(1)).alias("n_samples"),
            F.min(duration_col).alias("min_duration"),
            F.max(duration_col).alias("max_duration"),
        ).collect()
        
        group_1_stats = next(r for r in stats_row if r[group_col] == group_1_name)
        group_2_stats = next(r for r in stats_row if r[group_col] == group_2_name)
        
        n_1 = group_1_stats["n_samples"]
        n_2 = group_2_stats["n_samples"]
        min_dur = min(group_1_stats["min_duration"], group_2_stats["min_duration"])
        max_dur = max(group_1_stats["max_duration"], group_2_stats["max_duration"])
        
        # Resolution
        duration_range = max_dur - min_dur
        resolution = 1.0 if duration_range == 0 else duration_range / bins
        
        # --- 2. Discretize & Aggregate ---
        discretized_df = df.withColumn(
            "duration_bin",
            (F.floor(F.col(duration_col) / resolution) * resolution).cast("double"),
        )
        
        agg_df = discretized_df.groupBy("duration_bin", group_col).agg(
            F.sum(event_col).alias("d_t"),
            F.count(F.lit(1)).alias("total_observed"),
        )
        
        # --- 3. EARLY PIVOT (Required for Risk Set Alignment) ---
        pivot_df = agg_df.groupBy("duration_bin").pivot(
            group_col, values=[group_1_name, group_2_name]
        ).agg(
            F.first("d_t").alias("d_t"),
            F.first("total_observed").alias("total_observed")
        ).fillna(0)
        
        # --- 4. Calculate Risk Sets on ALIGNED Data ---
        window_spec = Window.orderBy("duration_bin").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        
        # Group 1 (Control) Risk Set
        pivot_df = pivot_df.withColumn(
            f"{group_1_name}_cum_obs", 
            F.sum(f"{group_1_name}_total_observed").over(window_spec)
        )
        pivot_df = pivot_df.withColumn(
            f"{group_1_name}_n_at_risk",
            F.lit(n_1) - F.col(f"{group_1_name}_cum_obs") + F.col(f"{group_1_name}_total_observed")
        )

        # Group 2 (Treatment) Risk Set
        pivot_df = pivot_df.withColumn(
            f"{group_2_name}_cum_obs", 
            F.sum(f"{group_2_name}_total_observed").over(window_spec)
        )
        pivot_df = pivot_df.withColumn(
            f"{group_2_name}_n_at_risk",
            F.lit(n_2) - F.col(f"{group_2_name}_cum_obs") + F.col(f"{group_2_name}_total_observed")
        )
        
        # --- 5. Calculate Divergence (Excess Events) ---
        col_d_t_1 = F.col(f"{group_1_name}_d_t")
        col_d_t_2 = F.col(f"{group_2_name}_d_t")
        col_n_1 = F.col(f"{group_1_name}_n_at_risk")
        col_n_2 = F.col(f"{group_2_name}_n_at_risk")

        pivot_df = pivot_df.withColumn("Oj", col_d_t_1 + col_d_t_2)
        pivot_df = pivot_df.withColumn("Nj", col_n_1 + col_n_2)
        
        # Expected events for Group 2 (Treatment)
        pivot_df = pivot_df.withColumn(
            "E2",
            F.when(F.col("Nj") > 0, F.col("Oj") * col_n_2 / F.col("Nj")).otherwise(0)
        )
        
        # Excess = Observed_Treatment - Expected_Treatment
        pivot_df = pivot_df.withColumn("excess_events", col_d_t_2 - F.col("E2"))
        
        # Collect results sorted by time
        results = pivot_df.select("duration_bin", "excess_events").orderBy("duration_bin").collect()
        
        # --- 6. Python-side Trajectory Analysis ---
        timeline = []
        excess_events = []
        cum_excess = []
        curr_cum = 0.0
        
        for row in results:
            t = float(row["duration_bin"])
            diff = float(row["excess_events"] or 0.0)
            
            curr_cum += diff
            timeline.append(t)
            excess_events.append(diff)
            cum_excess.append(curr_cum)
            
        # Crossover
        crossover = None
        for i in range(1, len(cum_excess)):
            if cum_excess[i-1] > 0 and cum_excess[i] < 0:
                crossover = timeline[i]
                break

        max_excess = max(cum_excess) if cum_excess else 0.0
        max_time_idx = np.argmax(cum_excess) if cum_excess else 0

        return CriticalTimeResult(
            cumulative_max_time=timeline[max_time_idx] if timeline else 0.0,
            cumulative_max_excess=max_excess,
            crossover_time=crossover,
            timeline=timeline,
            cumulative_excess_events=cum_excess,
            per_timepoint_excess=excess_events,
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
        bins: int = 100000,
    ) -> Dict[str, Any]:
        """Run log-rank test AND critical time analysis in a single call."""
        # Run log-rank test first
        log_rank_result = self.run_test(
            df, duration_col, event_col, group_col, test_type, alpha, bins
        )
        
        # Then find critical time
        critical_time_result = self.find_critical_time(
            df, duration_col, event_col, group_col, bins
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
        """Get a summary of the test results."""
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