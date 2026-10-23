"""
ml/train.py

Trains the energy-consumption forecasting model.

Two candidate models (RandomForest, XGBoost) are trained on the feature
table produced by ml/feature_engineering.py. Every run -- params, metrics,
and the fitted model itself -- is logged to MLflow. The best model
(lowest RMSE on the held-out test split) is registered in the MLflow
Model Registry as "energy-consumption-predictor" and also dumped to
ml/artifacts/best_model.pkl so the FastAPI service can load it even
without a running MLflow server.

Run:
    python -m ml.train
"""

import json
import os

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_and_save, FEATURES_PATH

ARTIFACT_DIR = "ml/artifacts"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "Energy_Consumption_Prediction"


def _load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        return build_and_save()
    return pd.read_csv(FEATURES_PATH)


def _evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_log() -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = _load_features()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            {"n_estimators": 200, "max_depth": 10},
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=250, max_depth=5, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
            ),
            {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.08},
        ),
    }

    results = {}
    best_name, best_model, best_rmse = None, None, float("inf")

    for name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = _evaluate(y_test, preds)

            mlflow.log_params(params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_metrics(metrics)
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            results[name] = metrics
            print(f"[train] {name}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_name = name
                best_model = model

    # Register the winning model in the MLflow Model Registry.
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_metrics(results[best_name])
        log_fn = mlflow.xgboost.log_model if best_name == "XGBoost" else mlflow.sklearn.log_model
        log_fn(
            best_model,
            artifact_path="model",
            registered_model_name="energy-consumption-predictor",
        )

    # Also persist a plain joblib artifact -- lets FastAPI serve
    # predictions even if no MLflow tracking server is reachable.
    joblib.dump(
        {"model": best_model, "features": FEATURE_COLUMNS, "model_name": best_name},
        f"{ARTIFACT_DIR}/best_model.pkl",
    )

    summary = {
        "results": results,
        "best_model": best_name,
        "best_rmse": best_rmse,
    }
    with open(f"{ARTIFACT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Best model: {best_name} (RMSE={best_rmse:.4f}) -> {ARTIFACT_DIR}/best_model.pkl")
    return summary


if __name__ == "__main__":
    train_and_log()
