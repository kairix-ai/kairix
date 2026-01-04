"""Synthetic data generators for survival analysis."""

from typing import Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, floor, lit, rand, when
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


def generate_synthetic_survival_data(
    spark: SparkSession,
    n_samples: int = 100000,
    distribution: str = "exponential",
    scale: float = 100.0,
    censoring_rate: float = 0.3,
    seed: Optional[int] = None,
) -> DataFrame:
    """Generate synthetic survival data for testing and benchmarking.

    This function creates a DataFrame with survival data following either an
    Exponential or Weibull distribution. The data includes an ID column,
    duration (time-to-event or censoring), and an event indicator.

    Args:
        spark: Active SparkSession.
        n_samples: Number of samples to generate. Default is 100,000.
        distribution: Distribution for survival times. Options are 'exponential'
            or 'weibull'. Default is 'exponential'.
        scale: Scale parameter for the distribution. For exponential, this is
            the mean. For Weibull, this is the scale. Default is 100.0.
        censoring_rate: Proportion of uncensored observations (event=1).
            Default is 0.3 (30% observed events).
        seed: Random seed for reproducibility. Default is None.

    Returns:
        DataFrame with columns:
            - id (string): Unique identifier
            - duration (float): Time-to-event or censoring time
            - event (int): 1 if event observed, 0 if censored

    Raises:
        ValueError: If distribution is not 'exponential' or 'weibull'.
        ValueError: If censoring_rate is not in [0, 1].

    Examples:
        >>> df = generate_synthetic_survival_data(spark, n_samples=1000)
        >>> df.printSchema()
        root
         |-- id: string (nullable = false)
         |-- duration: double (nullable = false)
         |-- event: integer (nullable = false)
    """
    if distribution not in ("exponential", "weibull"):
        raise ValueError("Distribution must be 'exponential' or 'weibull'")

    if not 0 <= censoring_rate <= 1:
        raise ValueError("censoring_rate must be in [0, 1]")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate random numbers for survival times and censoring
    # Using numpy for efficient generation, then creating DataFrame
    uniform_random = np.random.random(n_samples)

    if distribution == "exponential":
        # Exponential distribution: T = -scale * log(U) where U ~ Uniform(0,1)
        survival_times = -scale * np.log(1 - uniform_random)
    else:  # weibull
        # Weibull distribution with shape parameter k=1.5
        shape = 1.5
        survival_times = scale * (-np.log(1 - uniform_random)) ** (1 / shape)

    # Generate censoring indicator
    # Observing an event with probability (1 - censoring_rate)
    events = (uniform_random > censoring_rate).astype(np.int32)

    # Create local DataFrame with pandas/numpy
    local_data = {
        "id": [f"sample_{i:08d}" for i in range(n_samples)],
        "duration": survival_times.astype(np.float64),
        "event": events,
    }

    # Create DataFrame from local data
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("duration", DoubleType(), False),
            StructField("event", IntegerType(), False),
        ]
    )

    df = spark.createDataFrame(
        spark.sparkContext.parallelize(
            list(zip(local_data["id"], local_data["duration"], local_data["event"]))
        ),
        schema=schema,
    )

    return df
