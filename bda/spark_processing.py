"""
bda/spark_processing.py

Orchestrates the full BDA pipeline:

  raw CSV --> clean (Spark) --> aggregations (Spark) --> analytics (Spark)
      |                                                        |
      v                                                        v
 data/processed/cleaned_data.parquet          data/processed/*.csv + summary.json

Run directly:
    python -m bda.spark_processing
"""

import json
import os

from bda.spark_session import get_spark
from bda.data_cleaning import clean_from_csv
from bda import aggregations as agg
from bda import analytics as an

RAW_PATH = "data/raw/electricity_data.csv"
PROCESSED_DIR = "data/processed"


def run_pipeline(raw_path: str = RAW_PATH, out_dir: str = PROCESSED_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    spark = get_spark()
    try:
        df = clean_from_csv(spark, raw_path)
        df.cache()

        # Persist cleaned data for the ML feature-engineering stage.
        cleaned_pandas = df.toPandas()
        cleaned_pandas.to_parquet(f"{out_dir}/cleaned_data.parquet", index=False)
        cleaned_pandas.to_csv(f"{out_dir}/cleaned_data.csv", index=False)

        # Aggregations -> CSV (small, dashboard-friendly)
        agg.hourly_consumption(df).toPandas().to_csv(f"{out_dir}/hourly_consumption.csv", index=False)
        agg.daily_consumption(df).toPandas().to_csv(f"{out_dir}/daily_consumption.csv", index=False)
        agg.monthly_consumption(df).toPandas().to_csv(f"{out_dir}/monthly_consumption.csv", index=False)
        agg.appliance_consumption(df).toPandas().to_csv(f"{out_dir}/appliance_consumption.csv", index=False)
        agg.weekday_vs_weekend(df).toPandas().to_csv(f"{out_dir}/weekday_vs_weekend.csv", index=False)

        # Analytics -> single summary JSON
        summary = {
            "peak_usage": an.peak_usage(df),
            "cost_analysis": an.cost_analysis(df),
            "energy_trend": an.energy_trend(df),
            "row_count": df.count(),
        }
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("[spark_processing] BDA pipeline complete.")
        print(json.dumps(summary, indent=2, default=str))
        return summary
    finally:
        spark.stop()


if __name__ == "__main__":
    run_pipeline()
