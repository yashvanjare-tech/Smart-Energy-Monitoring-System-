"""
pages/3_🧪_MLOps.py

Surfaces the MLOps side of the project: MLflow experiment results, the
registered model version, and training history — reading directly from
the MLflow tracking store (sqlite) and the training summary artifact so
this works even if the MLflow UI (`mlflow ui`) isn't running separately.
"""

import json
import os

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLOps", page_icon="🧪", layout="wide")

st.title("🧪 Model Dashboard")
st.markdown("Experiment tracking and model registry status, powered by **MLflow**.")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SUMMARY_PATH = "ml/artifacts/training_summary.json"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.MlflowClient()

st.subheader("📦 Model Registry")
try:
    versions = client.search_model_versions("name='energy-consumption-predictor'")
    if versions:
        reg_df = pd.DataFrame([{
            "Version": v.version,
            "Stage": v.current_stage,
            "Run ID": v.run_id[:8],
            "Status": v.status,
            "Registered": pd.to_datetime(v.creation_timestamp, unit="ms"),
        } for v in versions]).sort_values("Version", ascending=False)
        st.dataframe(reg_df, use_container_width=True)
        latest = reg_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Version", f"v{latest['Version']}")
        c2.metric("Stage", latest["Stage"] or "None")
        c3.metric("Status", latest["Status"])
    else:
        st.info("No model registered yet. Run `python -m ml.train`.")
except Exception as e:
    st.warning(f"No MLflow registry data found yet ({e}). Run `python -m ml.train` first.")

st.divider()

st.subheader("📊 Experiment Runs — Energy Consumption Prediction")
try:
    experiment = client.get_experiment_by_name("Energy_Consumption_Prediction")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs.empty:
            display_cols = [c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.model_type")]
            table = runs[["run_id"] + display_cols].copy()
            table["run_id"] = table["run_id"].str[:8]
            table.columns = [c.replace("metrics.", "").replace("params.", "").upper() for c in table.columns]
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No runs logged yet.")
    else:
        st.info("Experiment not found yet — run `python -m ml.train`.")
except Exception as e:
    st.warning(f"Could not load experiment runs ({e}).")

st.divider()

st.subheader("🚨 Experiment Runs — Anomaly Detection")
try:
    exp2 = client.get_experiment_by_name("Energy_Anomaly_Detection")
    if exp2:
        runs2 = mlflow.search_runs(experiment_ids=[exp2.experiment_id])
        if not runs2.empty:
            cols2 = [c for c in runs2.columns if c.startswith("metrics.") or c.startswith("params.")]
            table2 = runs2[["run_id"] + cols2].copy()
            table2["run_id"] = table2["run_id"].str[:8]
            st.dataframe(table2, use_container_width=True)
        else:
            st.info("No anomaly-detection runs logged yet.")
    else:
        st.info("Run `python -m ml.anomaly_detection` to populate this section.")
except Exception as e:
    st.warning(f"Could not load anomaly experiment runs ({e}).")

st.divider()

st.subheader("🏆 Best Model Summary")
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    st.json(summary)
else:
    st.info("Run `python -m ml.train` to generate a training summary.")

with st.expander("ℹ️ How this fits together"):
    st.markdown("""
- **Training** (`ml/train.py`) fits RandomForest and XGBoost, logs every run's
  parameters/metrics/model artifact to MLflow, and registers the best one.
- **Model Registry** tracks versions of `energy-consumption-predictor` so the
  serving layer always knows which model is "current".
- **FastAPI** (`api/main.py`) loads the registered/joblib model and serves
  predictions to the Streamlit UI.
- **This page** reads directly from the MLflow tracking store — the same
  data you'd see in `mlflow ui`.
""")
