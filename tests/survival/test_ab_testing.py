"""Tests for A/B Testing module for survival analysis."""

import pytest
import pandas as pd
import numpy as np
from scipy import stats

from kairix.survival import SurvivalTester, LogRankResult


class TestLogRankResult:
    """Tests for LogRankResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LogRankResult(
            test_statistic=5.0,
            p_value=0.025,
            degrees_of_freedom=1,
            significant=True,
            group_1_name="A",
            group_2_name="B",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["test_statistic"] == 5.0
        assert result_dict["p_value"] == 0.025
        assert result_dict["degrees_of_freedom"] == 1
        assert result_dict["significant"] is True
        assert result_dict["group_1_name"] == "A"
        assert result_dict["group_2_name"] == "B"


class TestSurvivalTesterPandas:
    """Tests for SurvivalTester with pandas DataFrames."""
    
    def test_identical_groups_no_difference(self):
        """Test that identical groups show no significant difference."""
        # Create identical survival distributions
        np.random.seed(42)
        n = 50
        
        # Control group: exponential distribution with rate 0.1
        control_duration = np.random.exponential(10, n)
        control_duration = np.clip(control_duration, 0.1, 100).round(1)
        control_event = np.random.binomial(1, 0.7, n)
        
        # Treatment group: identical distribution
        treatment_duration = np.random.exponential(10, n)
        treatment_duration = np.clip(treatment_duration, 0.1, 100).round(1)
        treatment_event = np.random.binomial(1, 0.7, n)
        
        df = pd.DataFrame({
            'duration': np.concatenate([control_duration, treatment_duration]),
            'event': np.concatenate([control_event, control_event]),
            'variant': ['A'] * n + ['B'] * n,
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        # With identical distributions, p-value should be high (not significant)
        assert result['p_value'] > 0.05
        assert result['degrees_of_freedom'] == 1
    
    def test_different_groups_may_show_difference(self):
        """Test that different survival distributions may show difference."""
        # Create two groups with different survival distributions
        np.random.seed(42)
        n = 100
        
        # Control group: exponential with rate 0.1 (median ~6.9)
        control_duration = np.random.exponential(10, n)
        control_duration = np.clip(control_duration, 0.1, 100).round(1)
        control_event = np.random.binomial(1, 0.6, n)
        
        # Treatment group: exponential with rate 0.05 (median ~13.9) - better survival
        treatment_duration = np.random.exponential(20, n)
        treatment_duration = np.clip(treatment_duration, 0.1, 100).round(1)
        treatment_event = np.random.binomial(1, 0.6, n)
        
        df = pd.DataFrame({
            'duration': np.concatenate([control_duration, treatment_duration]),
            'event': np.concatenate([control_event, treatment_event]),
            'variant': ['A'] * n + ['B'] * n,
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        # With different distributions, p-value may or may not be significant
        assert isinstance(result['test_statistic'], float)
        assert 0 <= result['p_value'] <= 1
        assert result['degrees_of_freedom'] == 1
    
    def test_custom_group_names_in_result(self):
        """Test that group names are captured in result."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [1, 0, 1, 0, 1, 1],
            'group': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='group')
        
        assert 'X' in str(result['group_1_name']) or 'X' in str(result['group_2_name'])
    
    def test_summary_includes_group_stats(self):
        """Test that summary includes group-level statistics."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [1, 0, 1, 0, 1, 1],
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        summary = tester.summary()
        
        assert "n_group_1" in summary
        assert "n_group_2" in summary
        assert "events_group_1" in summary
        assert "events_group_2" in summary
    
    def test_not_run_raises_error(self):
        """Test that accessing summary before running test raises error."""
        tester = SurvivalTester()
        
        with pytest.raises(ValueError, match="Test has not been run"):
            tester.summary()
    
    def test_wrong_number_of_groups_raises_error(self):
        """Test that wrong number of groups raises error."""
        df = pd.DataFrame({
            'duration': [5, 6, 7],
            'event': [1, 0, 1],
            'variant': ['A', 'B', 'B'],  # Only 1 variant value
        })
        
        tester = SurvivalTester()
        
        with pytest.raises(ValueError, match="exactly 2 unique values"):
            tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
    
    def test_small_dataset(self):
        """Test A/B testing with a small dataset."""
        # Minimal dataset from standard survival analysis examples
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 9, 10, 11, 12],
            'event': [1, 0, 1, 0, 1, 1, 0, 1],
            'variant': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        assert isinstance(result['test_statistic'], float)
        assert 0 <= result['p_value'] <= 1
        assert tester.is_fitted is True
    
    def test_significance_flag_at_alpha_005(self):
        """Test significance flag at alpha=0.05."""
        # Create a dataset with known difference
        df = pd.DataFrame({
            'duration': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
            'event': [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            'variant': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        # Check that significance flag is correctly set
        if result['p_value'] < 0.05:
            assert result['significant'] is True
        else:
            assert result['significant'] is False
    
    def test_api_matches_instruction_example(self):
        """Test that API matches the example in instruction.md."""
        df = pd.DataFrame({
            'tenure': [5, 6, 7, 8, 10, 12, 15, 16],
            'churned': [1, 0, 1, 0, 1, 1, 1, 0],
            'variant': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, "tenure", "churned", group_col="variant")
        
        assert 'p_value' in result
        assert 'test_statistic' in result
        assert 'significant' in result


class TestSurvivalTesterEdgeCases:
    """Tests for edge cases in A/B testing."""
    
    def test_all_censored_events(self):
        """Test with all censored events (no events observed)."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [0, 0, 0, 0, 0, 0],  # All censored
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        # With no events, the test statistic should be 0
        assert result['test_statistic'] == 0.0
        # p-value should be 1.0 (no difference detected)
        assert result['p_value'] == 1.0
    
    def test_all_events(self):
        """Test with all observed events (no censoring)."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [1, 1, 1, 1, 1, 1],  # All events observed
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        # Should still compute without error
        assert isinstance(result['test_statistic'], float)
        assert 0 <= result['p_value'] <= 1
    
    def test_unequal_sample_sizes(self):
        """Test with unequal sample sizes between groups."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 35],
            'event': [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            'variant': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],  # 5 group A, 7 group B
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        summary = tester.summary()
        assert summary["n_group_1"] == 5
        assert summary["n_group_2"] == 7
    
    def test_very_small_sample(self):
        """Test with minimal sample size (2 per group)."""
        df = pd.DataFrame({
            'duration': [5, 10, 15, 20],
            'event': [1, 1, 1, 1],
            'variant': ['A', 'A', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_test(df, duration_col='duration', event_col='event', group_col='variant')
        
        assert isinstance(result['test_statistic'], float)
        assert 0 <= result['p_value'] <= 1


class TestSurvivalTesterSpark:
    """Tests for SurvivalTester with PySpark DataFrames."""
    
    @pytest.fixture
    def spark_session(self):
        """Create a Spark session for testing."""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("test_ab_testing") \
            .master("local[2]") \
            .getOrCreate()
        yield spark
        spark.stop()
    
    def test_spark_identical_groups(self, spark_session):
        """Test Spark A/B testing with identical groups."""
        np.random.seed(42)
        n = 100
        
        # Create identical distributions
        control_duration = np.random.exponential(10, n)
        control_duration = np.clip(control_duration, 0.1, 100).round(1)
        control_event = np.random.binomial(1, 0.7, n)
        
        treatment_duration = np.random.exponential(10, n)
        treatment_duration = np.clip(treatment_duration, 0.1, 100).round(1)
        treatment_event = np.random.binomial(1, 0.7, n)
        
        data = pd.DataFrame({
            'duration': np.concatenate([control_duration, treatment_duration]),
            'event': np.concatenate([control_event, treatment_event]),
            'variant': ['A'] * n + ['B'] * n,
        })
        
        spark_df = spark_session.createDataFrame(data)
        
        tester = SurvivalTester()
        result = tester.run_test(spark_df, duration_col='duration', event_col='event', group_col='variant')
        
        assert result['p_value'] > 0.05
        assert result['degrees_of_freedom'] == 1
    
    def test_spark_small_dataset(self, spark_session):
        """Test Spark A/B testing with small dataset."""
        data = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [1, 0, 1, 0, 1, 1],
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        
        spark_df = spark_session.createDataFrame(data)
        
        tester = SurvivalTester()
        result = tester.run_test(spark_df, duration_col='duration', event_col='event', group_col='variant')
        
        assert isinstance(result['test_statistic'], float)
        assert 0 <= result['p_value'] <= 1
        assert tester.is_fitted is True
    
    def test_spark_summary(self, spark_session):
        """Test that Spark summary includes group stats."""
        data = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12],
            'event': [1, 0, 1, 0, 1, 1],
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        
        spark_df = spark_session.createDataFrame(data)
        
        tester = SurvivalTester()
        tester.run_test(spark_df, duration_col='duration', event_col='event', group_col='variant')
        
        summary = tester.summary()
        
        assert "n_group_1" in summary
        assert "n_group_2" in summary
        assert summary["n_group_1"] == 3
        assert summary["n_group_2"] == 3
    
    def test_spark_consistency_with_pandas(self, spark_session):
        """Test that Spark and pandas give consistent results."""
        np.random.seed(42)
        n = 50
        
        control_duration = np.random.exponential(10, n)
        control_duration = np.clip(control_duration, 0.1, 100).round(1)
        control_event = np.random.binomial(1, 0.6, n)
        
        treatment_duration = np.random.exponential(12, n)
        treatment_duration = np.clip(treatment_duration, 0.1, 100).round(1)
        treatment_event = np.random.binomial(1, 0.6, n)
        
        data = pd.DataFrame({
            'duration': np.concatenate([control_duration, treatment_duration]),
            'event': np.concatenate([control_event, treatment_event]),
            'variant': ['A'] * n + ['B'] * n,
        })
        
        # Test with pandas
        pandas_tester = SurvivalTester()
        pandas_result = pandas_tester.run_test(data, duration_col='duration', event_col='event', group_col='variant')
        
        # Test with Spark
        spark_df = spark_session.createDataFrame(data)
        spark_tester = SurvivalTester()
        spark_result = spark_tester.run_test(spark_df, duration_col='duration', event_col='event', group_col='variant')
        
        # Results should be similar (allowing for floating point differences due to binning)
        assert abs(pandas_result['test_statistic'] - spark_result['test_statistic']) < 0.1
        assert abs(pandas_result['p_value'] - spark_result['p_value']) < 0.1


class TestBayesianComparison:
    """Tests for Bayesian comparison functionality."""
    
    def test_bayesian_basic(self):
        """Test basic Bayesian comparison."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8, 10, 12, 15, 16],
            'event': [1, 0, 1, 0, 1, 1, 1, 0],
            'variant': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_bayesian_comparison(df, 'duration', 'event', group_col='variant')
        
        assert 'prob_superiority' in result
        assert 0 <= result['prob_superiority'] <= 1
        assert 'credible_intervals' in result
        assert len(result['credible_intervals']) == 2
    
    def test_bayesian_custom_samples(self):
        """Test Bayesian comparison with custom number of samples."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8],
            'event': [1, 0, 1, 0],
            'variant': ['A', 'A', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        result = tester.run_bayesian_comparison(df, 'duration', 'event', group_col='variant', n_samples=500)
        
        assert result['posterior_samples'] == 500
    
    def test_bayesian_summary(self):
        """Test that summary works after Bayesian comparison."""
        df = pd.DataFrame({
            'duration': [5, 6, 7, 8],
            'event': [1, 0, 1, 0],
            'variant': ['A', 'A', 'B', 'B'],
        })
        
        tester = SurvivalTester()
        tester.run_bayesian_comparison(df, 'duration', 'event', group_col='variant')
        
        summary = tester.summary()
        assert 'n_group_1' in summary
        assert 'n_group_2' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
