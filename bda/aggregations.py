"""
bda/aggregations.py

PySpark aggregation jobs over the cleaned electricity dataset:
hourly, daily, appliance-level and monthly consumption.

Every function takes the cleaned Spark DataFrame produced by
bda.data_cleaning.clean() and returns a Spark DataFrame with the
aggregate -- callers convert to pandas only at the very end for
plotting / API responses, so the heavy lifting stays distributed.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def hourly_consumption(df: DataFrame) -> DataFrame:
    """Average energy consumption per hour-of-day, across the whole dataset."""
    return (
        df.groupBy("hour")
          .agg(
              F.round(F.sum("energy"), 3).alias("total_energy_kwh"),
              F.round(F.avg("energy"), 4).alias("avg_energy_kwh"),
              F.round(F.avg("power"), 2).alias("avg_power_w"),
          )
          .orderBy("hour")
    )


def daily_consumption(df: DataFrame) -> DataFrame:
    """Total energy per calendar date."""
    return (
        df.groupBy("date")
          .agg(
              F.round(F.sum("energy"), 3).alias("total_energy_kwh"),
              F.round(F.avg("power"), 2).alias("avg_power_w"),
              F.max("power").alias("peak_power_w"),
          )
          .orderBy("date")
    )


def monthly_consumption(df: DataFrame) -> DataFrame:
    """Total energy per month."""
    return (
        df.groupBy("month")
          .agg(F.round(F.sum("energy"), 2).alias("total_energy_kwh"))
          .orderBy("month")
    )


def appliance_consumption(df: DataFrame) -> DataFrame:
    """Share of total energy consumption per appliance."""
    total = df.agg(F.sum("energy")).first()[0] or 1.0
    return (
        df.groupBy("appliance")
          .agg(F.round(F.sum("energy"), 3).alias("total_energy_kwh"))
          .withColumn(
              "share_pct",
              F.round(F.col("total_energy_kwh") / F.lit(total) * 100, 1)
          )
          .orderBy(F.desc("total_energy_kwh"))
    )


def weekday_vs_weekend(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("is_weekend")
          .agg(F.round(F.avg("energy"), 4).alias("avg_energy_kwh"))
    )


def refit_house_consumption(df: DataFrame) -> DataFrame:
    """Household-level power and estimated energy analysis."""

    clean_df = (
        df.filter(
            (F.col("appliance") == "Aggregate") &
            F.col("house_id").isNotNull() &
            F.col("timestamp").isNotNull() &
            F.col("power_w").isNotNull() &
            ~F.isnan("power_w") &
            (F.col("power_w") >= 0) &
            (F.col("power_w") <= 4000)
        )
    )

    return (
        clean_df
        .groupBy("house_id")
        .agg(
            F.round(
                F.avg("power_w"), 2
            ).alias("avg_power_w"),

            F.round(
                F.max("power_w"), 2
            ).alias("peak_power_w"),

            F.round(
                F.sum(
                    F.col("power_w") * F.lit(8) /
                    F.lit(3600000)
                ),
                2
            ).alias("estimated_energy_kwh"),

            F.count("*").alias("reading_count")
        )
        .orderBy("house_id")
    )
    
def refit_appliance_consumption(df: DataFrame) -> DataFrame:
    """Appliance-level power and estimated energy analysis."""

    clean_df = (
        df.filter(
            (F.col("appliance") != "Aggregate") &
            F.col("appliance").isNotNull() &
            F.col("timestamp").isNotNull() &
            F.col("power_w").isNotNull() &
            ~F.isnan("power_w") &
            (F.col("power_w") >= 0) &
            (F.col("power_w") <= 4000)
        )
    )

    return (
        clean_df
        .groupBy("appliance")
        .agg(
            F.round(
                F.avg("power_w"), 2
            ).alias("avg_power_w"),

            F.round(
                F.max("power_w"), 2
            ).alias("peak_power_w"),

            F.round(
                F.sum(
                    F.col("power_w") * F.lit(8) /
                    F.lit(3600000)
                ),
                2
            ).alias("estimated_energy_kwh"),

            F.count("*").alias("reading_count")
        )
        .orderBy(F.desc("estimated_energy_kwh"))
    )

def refit_hourly_consumption(df: DataFrame) -> DataFrame:
    """Average and peak household power by hour of day."""

    aggregate_df = (
        df.filter(
            (F.col("appliance") == "Aggregate") &
            F.col("timestamp").isNotNull() &
            F.col("power_w").isNotNull()
        )
        .select(
            "timestamp",
            "power_w"
        )
        .withColumn(
            "hour",
            F.hour("timestamp")
        )
    )

    return (
        aggregate_df
        .groupBy("hour")
        .agg(
            F.round(
                F.avg("power_w"),
                2
            ).alias("avg_power_w"),

            F.round(
                F.max("power_w"),
                2
            ).alias("peak_power_w"),

            F.count("*").alias(
                "reading_count"
            )
        )
        .orderBy("hour")
    )