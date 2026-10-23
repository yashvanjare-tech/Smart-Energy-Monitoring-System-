"""
bda/data_cleaning.py

PySpark data-cleaning stage of the BDA pipeline.

Takes the raw appliance-level electricity readings (data/raw/electricity_data.csv)
and produces a cleaned, typed, de-duplicated Spark DataFrame ready for
aggregation / feature engineering.

Cleaning rules:
- Drop rows with a null appliance tag or null power reading.
- Drop rows with non-physical voltage (<= 0 or > 400V) -- sensor glitches.
- Cap current outliers using an IQR-based bound (sensor spikes).
- Remove exact duplicate readings.
- Parse timestamp into a proper timestamp type and derive hour/day/month/weekday.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

RAW_SCHEMA = StructType([
    StructField("timestamp", StringType(), True),
    StructField("appliance", StringType(), True),
    StructField("voltage", DoubleType(), True),
    StructField("current", DoubleType(), True),
    StructField("power", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
])


def load_raw(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, schema=RAW_SCHEMA)


def clean(df: DataFrame) -> DataFrame:
    before = df.count()

    df = df.dropna(subset=["appliance", "power", "voltage", "current"])

    # Physically implausible voltage readings -> sensor glitch, drop.
    df = df.filter((F.col("voltage") > 0) & (F.col("voltage") < 400))
    df = df.filter(F.col("power") >= 0)

    # IQR-based outlier capping on current (keeps genuine spikes' rows,
    # just clamps the value so aggregates aren't skewed by one bad sensor tick).
    q1, q3 = df.approxQuantile("current", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr if iqr > 0 else q3 * 3
    df = df.withColumn(
        "current",
        F.when(F.col("current") > upper_bound, upper_bound).otherwise(F.col("current"))
    )

    df = df.dropDuplicates(["timestamp", "appliance", "power", "voltage"])

    df = df.withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))
    df = df.dropna(subset=["timestamp"])

    df = (
        df.withColumn("hour", F.hour("timestamp"))
          .withColumn("date", F.to_date("timestamp"))
          .withColumn("day_name", F.date_format("timestamp", "EEEE"))
          .withColumn("month", F.date_format("timestamp", "yyyy-MM"))
          .withColumn("is_weekend", F.dayofweek("timestamp").isin([1, 7]))
    )

    after = df.count()
    print(f"[data_cleaning] {before:,} raw rows -> {after:,} cleaned rows "
          f"({before - after:,} dropped)")

    return df


def clean_from_csv(spark: SparkSession, raw_path: str) -> DataFrame:
    return clean(load_raw(spark, raw_path))
