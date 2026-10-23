"""
pages/4_⚙️_System.py

Operational status page: is the FastAPI service up, is a model loaded,
how fresh is the processed BDA data, how big are the raw/processed
datasets. Useful both for demoing the project and for sanity-checking
that the whole pipeline (Spark -> ML -> API) has actually been run.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append(".")
from src.api_client import api_health, model_info  # noqa: E402

st.set_page_config(page_title="System Status", page_icon="⚙️", layout="wide")

st.title("⚙️ System Status")

st.subheader("🌐 API Status")
health = api_health()
col1, col2, col3 = st.columns(3)

if health.get("status") == "ok":
    col1.success("FastAPI: Online ✅")
else:
    col1.error("FastAPI: Offline ❌")
    st.caption("Start it with: `uvicorn api.main:app --reload --port 8000`")

col2.info(f"Forecast model loaded: {'✅' if health.get('predictor_loaded') else '❌'}")
col3.info(f"Anomaly model loaded: {'✅' if health.get('anomaly_model_loaded') else '❌'}")

if health.get("status") == "ok":
    with st.expander("Model training summary (from API)"):
        st.json(model_info())

st.divider()

st.subheader("💾 Data Pipeline Status")


def file_status(path, label):
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        try:
            rows = sum(1 for _ in open(path)) - 1 if path.endswith(".csv") else None
        except Exception:
            rows = None
        st.success(f"✅ {label} — {size_kb:.1f} KB — last updated {mtime}"
                    + (f" — {rows:,} rows" if rows is not None else ""))
    else:
        st.error(f"❌ {label} — not found")


file_status("data/raw/electricity_data.csv", "Raw electricity data")
file_status("data/processed/cleaned_data.csv", "Cleaned data (Spark output)")
file_status("data/processed/summary.json", "BDA analytics summary")
file_status("data/processed/features.csv", "ML feature table")
file_status("ml/artifacts/best_model.pkl", "Trained forecasting model")
file_status("ml/artifacts/anomaly_model.pkl", "Trained anomaly model")
file_status("mlflow.db", "MLflow tracking database")

st.divider()

st.subheader("🔁 Run the Full Pipeline")
st.markdown("Run these in order from the project root to (re)build everything from scratch:")
st.code("""# 1. Generate synthetic IoT-style electricity data
python -m bda.data_generator --days 90

# 2. Run the PySpark BDA pipeline (cleaning + aggregations + analytics)
python -m bda.spark_processing

# 3. Build ML features and train models (tracked in MLflow)
python -m ml.feature_engineering
python -m ml.train
python -m ml.anomaly_detection

# 4. Start the prediction API
uvicorn api.main:app --reload --port 8000

# 5. (in another terminal) Start the dashboard
streamlit run app.py

# Optional: view the MLflow UI directly
mlflow ui --backend-store-uri sqlite:///mlflow.db
""", language="bash")

st.caption("Or use `docker compose up` to start Streamlit, FastAPI and MLflow together.")
