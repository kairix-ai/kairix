"""Engine dispatcher for detecting and handling Pandas vs Spark DataFrames."""

from typing import Union, Type

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame


BackendType = Union[pd.DataFrame, SparkDataFrame]


def get_backend(df: BackendType) -> str:
    """Detect the backend type of a DataFrame.
    
    Args:
        df: A pandas or PySpark DataFrame.
        
    Returns:
        String identifier: "pandas" or "spark".
        
    Raises:
        TypeError: If df is not a recognized DataFrame type.
    """
    if isinstance(df, pd.DataFrame):
        return "pandas"
    elif isinstance(df, SparkDataFrame):
        return "spark"
    else:
        raise TypeError(
            f"Unsupported DataFrame type: {type(df)}. "
            "Expected pandas.DataFrame or pyspark.sql.DataFrame."
        )


def is_spark(df: BackendType) -> bool:
    """Check if the DataFrame is a PySpark DataFrame."""
    return isinstance(df, SparkDataFrame)


def is_pandas(df: BackendType) -> bool:
    """Check if the DataFrame is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)
