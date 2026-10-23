"""
bda/spark_session.py

Single shared SparkSession builder so every BDA module configures Spark
the same way (local mode is fine for a mini-project; swap master to a
cluster URL for a real deployment).
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "SmartHomeEnergyBDA") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
