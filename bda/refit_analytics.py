"""
bda/refit_analytics.py

REFIT Big Data Analytics pipeline.

Uses PySpark for large-scale processing.
Only small aggregated results are collected
for dashboard/JSON output.
"""

import os
import json

from bda.spark_session import get_spark
from bda.refit_loader import load_refit
from bda import aggregations as agg


OUTPUT_DIR = "data/processed/refit"


def save_spark_result(df, filename):
    """
    Collect only the small aggregated result and save it as JSON.
    """

    rows = [
        row.asDict()
        for row in df.collect()
    ]

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            rows,
            f,
            indent=2,
            default=str
        )

    print(f"Saved: {path}")


def run_refit_analytics():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    spark = get_spark(
        "REFIT-BDA-Analytics"
    )

    try:

        print("\n======================================")
        print("REFIT BIG DATA ANALYTICS")
        print("======================================")

        print("\nLoading REFIT data...")

        df = load_refit(
            spark,
            "data/raw"
        )

        # --------------------------------------------------
        # HOUSE-WISE ANALYSIS
        # --------------------------------------------------

        print("\n[1/3] Running household analysis...")

        house_df = agg.refit_house_consumption(df)

        save_spark_result(
            house_df,
            "house_consumption.json"
        )

        print("Household analysis complete.")

        # --------------------------------------------------
        # APPLIANCE-WISE ANALYSIS
        # --------------------------------------------------

        print("\n[2/3] Running appliance analysis...")

        appliance_df = agg.refit_appliance_consumption(df)

        save_spark_result(
            appliance_df,
            "appliance_consumption.json"
        )

        print("Appliance analysis complete.")

        # --------------------------------------------------
        # HOURLY ANALYSIS
        # --------------------------------------------------

        print("\n[3/3] Running hourly analysis...")

        hourly_df = agg.refit_hourly_consumption(df)

        save_spark_result(
            hourly_df,
            "hourly_consumption.json"
        )

        print("Hourly analysis complete.")

        # --------------------------------------------------
        # FINAL STATUS
        # --------------------------------------------------

        print("\n======================================")
        print("REFIT BDA ANALYTICS COMPLETE")
        print("======================================")

        print("\nGenerated files:")

        print(
            "1. data/processed/refit/house_consumption.json"
        )

        print(
            "2. data/processed/refit/appliance_consumption.json"
        )

        print(
            "3. data/processed/refit/hourly_consumption.json"
        )

    finally:

        spark.stop()


if __name__ == "__main__":
    run_refit_analytics()