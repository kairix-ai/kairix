"""Pytest configuration and fixtures for Kairix tests."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Create a SparkSession for testing.

    This fixture creates a local SparkSession with minimal configuration
    suitable for unit testing. The session is shared across all tests
    in a test session for efficiency.

    Yields:
        SparkSession: Active SparkSession instance.
    """
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("kairix-test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    # Set log level to reduce noise during tests
    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    # Cleanup after tests
    spark.stop()


@pytest.fixture(scope="function")
def small_survival_data(spark_session):
    """Create a small survival dataset for testing.

    This fixture provides a small DataFrame suitable for comparing
    results against lifelines.KaplanMeierFitter.

    Returns:
        DataFrame with columns: id (string), duration (float), event (int).
    """
    from kairix.survival.utils import generate_synthetic_survival_data

    return generate_synthetic_survival_data(
        spark_session, n_samples=100, censoring_rate=0.3, seed=42
    )


@pytest.fixture(scope="function")
def medium_survival_data(spark_session):
    """Create a medium-sized survival dataset for scale testing.

    Returns:
        DataFrame with 10,000 samples.
    """
    from kairix.survival.utils import generate_synthetic_survival_data

    return generate_synthetic_survival_data(
        spark_session, n_samples=10000, censoring_rate=0.3, seed=42
    )
