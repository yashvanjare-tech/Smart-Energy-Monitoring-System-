"""
bda/refit_loader.py

REFIT dataset loader and transformation using PySpark.

Loads all REFIT household CSV files and transforms
wide-format appliance measurements into a long-format
Spark DataFrame.

Input:
    data/raw/House1.csv ... House21.csv

Output columns:
    house_id
    timestamp
    channel
    appliance
    power_w
"""

import os
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType,
)


# =========================================================
# REFIT CSV SCHEMA
# =========================================================

REFIT_SCHEMA = StructType([
    StructField("unix_timestamp", LongType(), True),
    StructField("aggregate", DoubleType(), True),
    StructField("appliance1", DoubleType(), True),
    StructField("appliance2", DoubleType(), True),
    StructField("appliance3", DoubleType(), True),
    StructField("appliance4", DoubleType(), True),
    StructField("appliance5", DoubleType(), True),
    StructField("appliance6", DoubleType(), True),
    StructField("appliance7", DoubleType(), True),
    StructField("appliance8", DoubleType(), True),
    StructField("appliance9", DoubleType(), True),
])


# =========================================================
# REFIT APPLIANCE MAPPING
# =========================================================

REFIT_APPLIANCES = {

    1: [
        "Aggregate",
        "Fridge",
        "Freezer(1)",
        "Freezer(2)",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television Site",
        "Electric Heater",
    ],

    2: [
        "Aggregate",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Television Site",
        "Microwave",
        "Toaster",
        "Hi-Fi",
        "Kettle",
        "Overhead Fan",
    ],

    3: [
        "Aggregate",
        "Toaster",
        "Fridge-Freezer",
        "Freezer",
        "Tumble Dryer",
        "Dishwasher",
        "Washing Machine",
        "Television Site",
        "Microwave",
        "Kettle",
    ],

    4: [
        "Aggregate",
        "Fridge",
        "Freezer",
        "Fridge-Freezer",
        "Washing Machine(1)",
        "Washing Machine(2)",
        "Desktop Computer",
        "Television Site",
        "Microwave",
        "Kettle",
    ],

    5: [
        "Aggregate",
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Desktop Computer",
        "Television Site",
        "Microwave",
        "Kettle",
        "Toaster",
    ],

    6: [
        "Aggregate",
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "MJY Computer",
        "TV/Satellite",
        "Microwave",
        "Kettle",
        "Toaster",
        "PGM Computer",
    ],

    7: [
        "Aggregate",
        "Fridge",
        "Freezer(1)",
        "Freezer(2)",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Television Site",
        "Toaster",
        "Kettle",
    ],

    8: [
        "Aggregate",
        "Fridge",
        "Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Toaster",
        "Computer",
        "Television Site",
        "Microwave",
        "Kettle",
    ],

    9: [
        "Aggregate",
        "Fridge-Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Television Site",
        "Microwave",
        "Kettle",
        "Hi-Fi",
        "Electric Heater",
    ],

    10: [
        "Aggregate",
        "Magimix(Blender)",
        "Toaster",
        "Chest Freezer",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Television Site",
        "Microwave",
        "K Mix",
    ],

    11: [
        "Aggregate",
        "Firdge",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Computer Site",
        "Microwave",
        "Kettle",
        "Router",
        "Hi-Fi",
    ],

    12: [
        "Aggregate",
        "Fridge-Freezer",
        "Unknown",
        "Unknown",
        "Computer Site",
        "Microwave",
        "Kettle",
        "Toaster",
        "Television",
        "Unknown",
    ],

    13: [
        "Aggregate",
        "Television Site",
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Unknown",
        "Network Site",
        "Microwave",
        "Microwave",
        "Kettle",
    ],

    15: [
        "Aggregate",
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer Site",
        "Television Site",
        "Microwave",
        "Hi-Fi",
        "Toaster",
    ],

    16: [
        "Aggregate",
        "Fridge-Freezer(1)",
        "Fridge-Freezer(2)",
        "Electric Heater(1)",
        "Electric Heater(2)",
        "Washing Machine",
        "Dishwasher",
        "Computer Site",
        "Television Site",
        "Dehumidifier",
    ],

    17: [
        "Aggregate",
        "Freezer",
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Computer Site",
        "Television Site",
        "Microwave",
        "Kettle",
        "TV Site(Bedroom)",
    ],

    18: [
        "Aggregate",
        "Fridge(garage)",
        "Freezer(garage)",
        "Fridge-Freezer",
        "Washer Dryer(garage)",
        "Washing Machine",
        "Dishwasher",
        "Desktop Computer",
        "Television Site",
        "Microwave",
    ],

    19: [
        "Aggregate",
        "Fridge Freezer",
        "Washing Machine",
        "Television Site",
        "Microwave",
        "Kettle",
        "Toaster",
        "Bread-maker",
        "Games Console",
        "Hi-Fi",
    ],

    20: [
        "Aggregate",
        "Fridge",
        "Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer Site",
        "Television Site",
        "Microwave",
        "Kettle",
    ],

    21: [
        "Aggregate",
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Food Mixer",
        "Television",
        "Unknown",
        "Vivarium",
        "Pond Pump",
    ],
}


# =========================================================
# HOUSE IDs
# =========================================================

HOUSE_IDS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13,
    15, 16, 17, 18, 19, 20, 21
]


# =========================================================
# LOAD REFIT CSV FILES
# =========================================================

def load_refit_wide(
    spark: SparkSession,
    data_dir: str = "data/raw",
) -> DataFrame:

    paths = [
        os.path.join(
            data_dir,
            f"House{house_id}.csv"
        )
        for house_id in HOUSE_IDS
    ]

    df = (
        spark.read
        .option("header", "false")
        .schema(REFIT_SCHEMA)
        .csv(paths)
    )

    # Get House ID from source file path.
    df = df.withColumn(
        "source_file",
        F.input_file_name()
    )

    df = df.withColumn(
        "house_id",
        F.regexp_extract(
            F.col("source_file"),
            r"House(\d+)\.csv",
            1
        ).cast("int")
    )

    return df


# =========================================================
# WIDE → LONG TRANSFORMATION
# =========================================================

def transform_refit(
    df: DataFrame,
) -> DataFrame:

    # UNIX timestamp → Spark timestamp
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.from_unixtime(
                F.col("unix_timestamp")
            )
        )
    )

    # Remove invalid timestamps
    df = df.filter(
        F.col("timestamp").isNotNull()
    )

    # Power columns
    power_columns = [
        "aggregate",
        "appliance1",
        "appliance2",
        "appliance3",
        "appliance4",
        "appliance5",
        "appliance6",
        "appliance7",
        "appliance8",
        "appliance9",
    ]

    # Remove negative power values
    for column in power_columns:

        df = df.filter(
            (F.col(column) >= 0)
            | F.col(column).isNull()
        )

    # Wide → Long
    long_df = df.select(
        "house_id",
        "timestamp",

        F.expr("""
            stack(
                10,

                'Aggregate', aggregate,

                'Appliance1', appliance1,

                'Appliance2', appliance2,

                'Appliance3', appliance3,

                'Appliance4', appliance4,

                'Appliance5', appliance5,

                'Appliance6', appliance6,

                'Appliance7', appliance7,

                'Appliance8', appliance8,

                'Appliance9', appliance9
            )
        """).alias(
            "channel",
            "power_w"
        )
    )

    return long_df


# =========================================================
# ADD REAL APPLIANCE NAMES
# =========================================================

def add_appliance_names(
    df: DataFrame,
) -> DataFrame:

    mapping_rows = []

    for house_id, appliances in REFIT_APPLIANCES.items():

        for channel_number, appliance_name in enumerate(
            appliances
        ):

            if channel_number == 0:

                channel = "Aggregate"

            else:

                channel = (
                    f"Appliance{channel_number}"
                )

            mapping_rows.append(
                (
                    house_id,
                    channel,
                    appliance_name,
                )
            )

    spark = df.sparkSession

    mapping_df = spark.createDataFrame(
        mapping_rows,
        [
            "house_id",
            "channel",
            "appliance",
        ],
    )

    result = (
        df.join(
            mapping_df,
            on=[
                "house_id",
                "channel"
            ],
            how="left",
        )
        .select(
            "house_id",
            "timestamp",
            "channel",
            "appliance",
            "power_w",
        )
    )

    return result


# =========================================================
# STANDARDIZE REFIT APPLIANCE NAMES
# =========================================================
#
# Purpose:
#   1. Merge duplicate/variant appliance names.
#   2. Remove location-specific naming variants.
#   3. Remove spelling variants.
#   4. Remove "Unknown" from analytics.
#
# IMPORTANT:
#   Raw REFIT CSV files are NOT modified.
#   Standardization happens only in the Spark processing layer.
# =========================================================

APPLIANCE_NAME_MAP = {

    # -----------------------------------------------------
    # FREEZER
    # -----------------------------------------------------

    "Freezer(1)": "Freezer",
    "Freezer(2)": "Freezer",
    "Freezer(garage)": "Freezer",
    "Chest Freezer": "Freezer",

    # -----------------------------------------------------
    # FRIDGE
    # -----------------------------------------------------

    "Fridge(garage)": "Fridge",
    "Firdge": "Fridge",

    # -----------------------------------------------------
    # FRIDGE-FREEZER
    # -----------------------------------------------------

    "Fridge Freezer": "Fridge-Freezer",
    "Fridge-Freezer(1)": "Fridge-Freezer",
    "Fridge-Freezer(2)": "Fridge-Freezer",

    # -----------------------------------------------------
    # WASHING MACHINE
    # -----------------------------------------------------

    "Washing Machine(1)": "Washing Machine",
    "Washing Machine(2)": "Washing Machine",

    # -----------------------------------------------------
    # ELECTRIC HEATER
    # -----------------------------------------------------

    "Electric Heater(1)": "Electric Heater",
    "Electric Heater(2)": "Electric Heater",

    # -----------------------------------------------------
    # WASHER DRYER
    # -----------------------------------------------------

    "Washer Dryer(garage)": "Washer Dryer",

    # -----------------------------------------------------
    # TELEVISION
    # -----------------------------------------------------

    "Television Site": "Television",
    "TV Site(Bedroom)": "Television",
    "TV/Satellite": "Television",

    # -----------------------------------------------------
    # COMPUTER
    # -----------------------------------------------------

    "Computer Site": "Computer",
    "MJY Computer": "Computer",
    "PGM Computer": "Computer",
    "Desktop Computer": "Computer",

    # -----------------------------------------------------
    # FAN
    # -----------------------------------------------------

    "Overhead Fan": "Fan",

    # -----------------------------------------------------
    # BLENDER
    # -----------------------------------------------------

    "Magimix(Blender)": "Blender",

    # -----------------------------------------------------
    # KITCHEN MIXER
    # -----------------------------------------------------

    "K Mix": "Kitchen Mixer",
    "Food Mixer": "Kitchen Mixer",

    # -----------------------------------------------------
    # NETWORK
    # -----------------------------------------------------

    "Network Site": "Network",
}


def standardize_appliance_names(
    df: DataFrame,
) -> DataFrame:

    # Build Spark map expression.
    mapping_expr = F.create_map(
        *[
            item
            for pair in APPLIANCE_NAME_MAP.items()
            for item in (
                F.lit(pair[0]),
                F.lit(pair[1]),
            )
        ]
    )

    # Replace variants with standardized names.
    df = df.withColumn(
        "appliance",
        F.coalesce(
            mapping_expr[F.col("appliance")],
            F.col("appliance")
        )
    )

    # -----------------------------------------------------
    # REMOVE UNKNOWN
    # -----------------------------------------------------

    df = df.filter(
        F.col("appliance").isNotNull()
        &
        (
            F.trim(
                F.col("appliance")
            ) != ""
        )
        &
        (
            F.lower(
                F.trim(
                    F.col("appliance")
                )
            ) != "unknown"
        )
    )

    return df


# =========================================================
# COMPLETE REFIT PIPELINE
# =========================================================

def load_refit(
    spark: SparkSession,
    data_dir: str = "data/raw",
) -> DataFrame:

    print("\n======================================")
    print("REFIT BIG DATA LOADER")
    print("======================================")

    print("\nLoading household CSV files...")

    wide_df = load_refit_wide(
        spark,
        data_dir,
    )

    print("\nWide-format schema:")
    wide_df.printSchema()

    print("\nTransforming wide → long format...")

    long_df = transform_refit(
        wide_df
    )

    print("\nAdding appliance names...")

    final_df = add_appliance_names(
        long_df
    )

    print("\nStandardizing appliance names...")

    final_df = standardize_appliance_names(
        final_df
    )

    return final_df


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from bda.spark_session import get_spark

    spark = get_spark(
        "REFIT-All-Houses-Test"
    )

    try:

        df = load_refit(
            spark,
            "data/raw",
        )

        print("\n======================================")
        print("FINAL REFIT SCHEMA")
        print("======================================")

        df.printSchema()

        print("\n======================================")
        print("SAMPLE DATA")
        print("======================================")

        df.show(
            30,
            truncate=False
        )

        print("\n======================================")
        print("HOUSE DISTRIBUTION")
        print("======================================")

        (
            df.groupBy("house_id")
            .count()
            .orderBy("house_id")
            .show(30)
        )

        print("\n======================================")
        print("FINAL CLEAN APPLIANCE CATEGORIES")
        print("======================================")

        (
            df.filter(
                F.col("appliance") != "Aggregate"
            )
            .select("appliance")
            .distinct()
            .orderBy("appliance")
            .show(100, truncate=False)
        )

        print("\n======================================")
        print("APPLIANCE DISTRIBUTION")
        print("======================================")

        (
            df.groupBy("appliance")
            .count()
            .orderBy(
                F.desc("count")
            )
            .show(
                100,
                truncate=False
            )
        )

        print("\n======================================")
        print("FINAL APPLIANCE CATEGORY COUNT")
        print("======================================")

        category_count = (
            df.filter(
                F.col("appliance") != "Aggregate"
            )
            .select("appliance")
            .distinct()
            .count()
        )

        print(
            f"Clean appliance categories: "
            f"{category_count}"
        )

        print("\n======================================")
        print("UNKNOWN CHECK")
        print("======================================")

        unknown_count = (
            df.filter(
                F.lower(
                    F.trim(
                        F.col("appliance")
                    )
                ) == "unknown"
            )
            .count()
        )

        print(
            f"Unknown records remaining: "
            f"{unknown_count}"
        )

        print("\n======================================")
        print("REFIT ALL-HOUSES TEST COMPLETE")
        print("======================================")

    finally:

        spark.stop()