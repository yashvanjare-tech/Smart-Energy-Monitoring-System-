"""
ml/anomaly_detection.py

Flags each hourly household reading as Normal / Unusual / High consumption
using an IsolationForest trained on the same feature table used for
forecasting. This run is also logged to MLflow so it shows up alongside
the forecasting experiments.

Run:
    python -m ml.anomaly_detection
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_and_save, FEATURES_PATH
from ml.train import MLFLOW_TRACKING_URI

ARTIFACT_DIR = "ml/artifacts"
EXPERIMENT_NAME = "Energy_Anomaly_Detection"


def _load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        return build_and_save()
    return pd.read_csv(FEATURES_PATH)


def classify_status(row_energy: float, mean: float, std: float) -> str:
    """Normal / Unusual / High based on distance from the mean in std-devs."""
    if std == 0:
        return "Normal"
    z = (row_energy - mean) / std
    if z > 2.5:
        return "High"
    if z > 1.5 or z < -1.5:
        return "Unusual"
    return "Normal"


def train_and_score() -> pd.DataFrame:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = _load_features()
    X = df[FEATURE_COLUMNS]

    contamination = 0.07
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)

    with mlflow.start_run(run_name="IsolationForest"):
        model.fit(X)
        raw_scores = model.decision_function(X)
        is_outlier = model.predict(X) == -1

        mean_e, std_e = df[TARGET_COLUMN].mean(), df[TARGET_COLUMN].std()
        df["anomaly_score"] = raw_scores
        df["is_outlier"] = is_outlier
        df["status"] = df[TARGET_COLUMN].apply(lambda e: classify_status(e, mean_e, std_e))
        # Isolation Forest outliers that aren't already "High" get bumped to "Unusual".
        df.loc[df["is_outlier"] & (df["status"] == "Normal"), "status"] = "Unusual"

        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_metric("pct_flagged", float(is_outlier.mean() * 100))
        mlflow.sklearn.log_model(model, artifact_path="model")

        joblib.dump(
            {"model": model, "features": FEATURE_COLUMNS, "mean": mean_e, "std": std_e},
            f"{ARTIFACT_DIR}/anomaly_model.pkl",
        )

    df.to_csv(f"{ARTIFACT_DIR}/scored_readings.csv", index=False)
    print(f"[anomaly_detection] Flagged {is_outlier.sum():,} / {len(df):,} "
          f"readings ({is_outlier.mean()*100:.1f}%) as outliers.")
    print(df["status"].value_counts())
    return df


if __name__ == "__main__":
    train_and_score()
