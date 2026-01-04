"""Validation utilities for Kairix."""

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType


def validate_input_schema(
    df: DataFrame, duration_col: str, event_col: str
) -> None:
    """Validate that the input DataFrame has the required schema.

    Args:
        df: Input PySpark DataFrame.
        duration_col: Name of the duration column.
        event_col: Name of the event column.

    Raises:
        ValueError: If required columns are missing or have wrong types.
        TypeError: If df is not a PySpark DataFrame.
    """
    if not isinstance(df, DataFrame):
        raise TypeError("df must be a PySpark DataFrame")

    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in DataFrame")

    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found in DataFrame")

    # Validate duration column type (numeric)
    duration_dtype = df.schema[duration_col].dataType
    if not isinstance(duration_dtype, (DoubleType, IntegerType)):
        raise ValueError(
            f"Duration column '{duration_col}' must be numeric (DoubleType or IntegerType), "
            f"found {duration_dtype}"
        )

    # Validate event column type (integer)
    event_dtype = df.schema[event_col].dataType
    if not isinstance(event_dtype, IntegerType):
        raise ValueError(
            f"Event column '{event_col}' must be IntegerType, found {event_dtype}"
        )

    # Validate no negative durations
    min_duration = df.selectExpr(f"MIN({duration_col})").collect()[0][0]
    if min_duration is not None and min_duration < 0:
        raise ValueError(f"Duration values must be non-negative, found min={min_duration}")
