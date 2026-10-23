"""
ml/feature_engineering.py

REFIT-based feature engineering for the Smart Home Energy project.

Large REFIT household data is processed with PySpark first.
Only the final hourly household feature table is collected into Pandas.

Sampling interval:
    8 seconds

Energy calculation:
    energy_kwh = power_w * 8 / 3,600,000
"""

import os

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from bda.spark_session import get_spark
from bda.refit_loader import load_refit


# ============================================================
# PATHS
# ============================================================

REFIT_RAW_DIR = "data/raw"
FEATURES_PATH = "data/processed/features.csv"


# ============================================================
# FEATURE / TARGET COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "avg_power_w",
    "peak_power_w",
    "previous_consumption",
    "appliance_count",
]

TARGET_COLUMN = "total_energy_kwh"


# ============================================================
# BUILD HOURLY HOUSEHOLD TABLE
# ============================================================

def build_hourly_household_table(
    raw_dir: str = REFIT_RAW_DIR,
) -> pd.DataFrame:

    print("\n======================================")
    print("REFIT ML FEATURE ENGINEERING")
    print("======================================")

    print("\nStarting Spark...")
    spark = get_spark("REFIT-ML-FeatureEngineering")

    try:
        print("\nLoading REFIT data...")
        df = load_refit(spark, raw_dir)

        print("\nFiltering aggregate household readings...")

        # REFIT Aggregate channel represents total household power.
        aggregate = (
            df.filter(
                (F.col("appliance") == "Aggregate")
                & F.col("timestamp").isNotNull()
                & F.col("power_w").isNotNull()
                & (F.col("power_w") >= 0)
            )
            .select(
                "house_id",
                "timestamp",
                "power_w",
            )
        )

        # ----------------------------------------------------
        # Convert timestamp to hourly bucket
        # ----------------------------------------------------

        aggregate = aggregate.withColumn(
            "hour_timestamp",
            F.date_trunc("hour", F.col("timestamp"))
        )

        print("\nAggregating household consumption by hour...")

        hourly = (
            aggregate
            .groupBy("house_id", "hour_timestamp")
            .agg(
                F.round(F.avg("power_w"), 4).alias("avg_power_w"),
                F.round(F.max("power_w"), 4).alias("peak_power_w"),
                F.sum(
                    F.col("power_w") * F.lit(8.0) / F.lit(3600000.0)
                ).alias("total_energy_kwh"),
                F.count("*").alias("reading_count"),
            )
        )

        # ----------------------------------------------------
        # Appliance usage count
        # ----------------------------------------------------

        print("\nCalculating appliance usage count...")

        appliance_usage = (
            df.filter(
                (F.col("appliance") != "Aggregate")
                & F.col("appliance").isNotNull()
                & F.col("timestamp").isNotNull()
                & F.col("power_w").isNotNull()
                & (F.col("power_w") > 0)
            )
            .withColumn(
                "hour_timestamp",
                F.date_trunc("hour", F.col("timestamp"))
            )
            .groupBy("house_id", "hour_timestamp")
            .agg(
                F.countDistinct("appliance").alias("appliance_count")
            )
        )

        # ----------------------------------------------------
        # Join aggregate + appliance information
        # ----------------------------------------------------

        print("\nJoining hourly household features...")

        hourly = (
            hourly
            .join(
                appliance_usage,
                on=["house_id", "hour_timestamp"],
                how="left",
            )
            .fillna({"appliance_count": 0})
        )

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        hourly = (
            hourly
            .withColumn("hour", F.hour("hour_timestamp"))
            .withColumn("day_of_week", F.dayofweek("hour_timestamp") - 2)
            .withColumn("month", F.month("hour_timestamp"))
        )

        # Spark dayofweek:
        # Sunday = 1
        # Monday = 2
        # ...
        # Saturday = 7
        #
        # Convert to:
        # Monday = 0
        # ...
        # Sunday = 6

        hourly = hourly.withColumn(
            "day_of_week",
            F.when(F.col("day_of_week") < 0, 6)
             .otherwise(F.col("day_of_week"))
        )

        hourly = hourly.withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin([5, 6]), 1)
             .otherwise(0)
        )

        # ----------------------------------------------------
        # Previous hourly consumption
        # ----------------------------------------------------

        print("\nCreating previous-consumption feature...")

        window = (
            Window
            .partitionBy("house_id")
            .orderBy("hour_timestamp")
        )

        hourly = hourly.withColumn(
            "previous_consumption",
            F.lag("total_energy_kwh").over(window)
        )

        # ----------------------------------------------------
        # Fill first record of each household
        # ----------------------------------------------------

        global_mean = hourly.select(
            F.avg("total_energy_kwh").alias("mean_energy")
        ).collect()[0]["mean_energy"]

        if global_mean is None:
            global_mean = 0.0

        hourly = hourly.fillna(
            {"previous_consumption": float(global_mean)}
        )

        # ----------------------------------------------------
        # Select final feature table
        # ----------------------------------------------------

        result = (
            hourly
            .select(
                "house_id",
                "hour_timestamp",
                "hour",
                "day_of_week",
                "month",
                "is_weekend",
                "avg_power_w",
                "peak_power_w",
                "previous_consumption",
                "appliance_count",
                "total_energy_kwh",
                "reading_count",
            )
            .orderBy("house_id", "hour_timestamp")
        )

        print("\nCollecting hourly aggregated data...")

        # IMPORTANT:
        # We collect ONLY the aggregated hourly table.
        # We are NOT collecting the 1.33B raw REFIT rows.

        rows = result.collect()

        pandas_df = pd.DataFrame(
            [row.asDict() for row in rows]
        )

        if pandas_df.empty:
            raise RuntimeError(
                "REFIT feature engineering produced no hourly records."
            )

        pandas_df["hour_timestamp"] = pd.to_datetime(
            pandas_df["hour_timestamp"]
        )

        pandas_df["total_energy_kwh"] = pandas_df[
            "total_energy_kwh"
        ].round(6)

        pandas_df["avg_power_w"] = pandas_df[
            "avg_power_w"
        ].round(4)

        pandas_df["peak_power_w"] = pandas_df[
            "peak_power_w"
        ].round(4)

        pandas_df["previous_consumption"] = pandas_df[
            "previous_consumption"
        ].round(6)

        pandas_df["appliance_count"] = pandas_df[
            "appliance_count"
        ].astype(int)

        pandas_df["reading_count"] = pandas_df[
            "reading_count"
        ].astype(int)

        print(
            f"\nHourly feature records: "
            f"{len(pandas_df):,}"
        )

        print(
            f"Houses represented: "
            f"{pandas_df['house_id'].nunique()}"
        )

        print(
            f"Date range: "
            f"{pandas_df['hour_timestamp'].min()} "
            f"to "
            f"{pandas_df['hour_timestamp'].max()}"
        )

        return pandas_df

    finally:
        spark.stop()
        print("\nSpark stopped.")


# ============================================================
# BUILD AND SAVE FEATURES
# ============================================================

def build_and_save(
    raw_dir: str = REFIT_RAW_DIR,
    out_path: str = FEATURES_PATH,
) -> pd.DataFrame:

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    features = build_hourly_household_table(raw_dir)

    features.to_csv(
        out_path,
        index=False
    )

    print(
        f"\nSaved REFIT ML features:"
        f"\n{out_path}"
    )

    print("\nFeature columns:")

    for column in features.columns:
        print(f"  - {column}")

    print("\n======================================")
    print("REFIT FEATURE ENGINEERING COMPLETE")
    print("======================================")

    return features


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_and_save()