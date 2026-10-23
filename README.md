# ⚡ Smart Home Energy Analytics & MLOps Platform

An end-to-end Big Data Analytics + MLOps platform built on top of a Smart Home
Energy Monitoring System. Appliance-level electricity data is processed at
scale with **Apache PySpark**, used to train and track **ML models with
MLflow**, served through a **FastAPI** prediction service, and visualized in
a multi-page **Streamlit** dashboard — all containerized with **Docker** and
validated by a **GitHub Actions** CI/CD pipeline.

---

## 📌 Project Overview

The original project simulated household energy usage and visualized it in a
single-page Streamlit app. This version keeps that simulator as the "live
monitoring" entry point, but adds the full data/ML lifecycle around it:

- **Big Data Analytics (BDA):** PySpark ingests, cleans, and aggregates a
  large appliance-level electricity dataset (hourly, daily, monthly, peak
  usage, cost analysis).
- **Machine Learning:** an energy-consumption forecaster (RandomForest /
  XGBoost) and an anomaly detector (IsolationForest), trained on
  Spark-processed features.
- **MLOps:** every training run is tracked in **MLflow** (params, metrics,
  model artifacts); the best model is registered in the MLflow Model
  Registry.
- **Serving:** a **FastAPI** service exposes `/predict` and
  `/detect-anomaly` endpoints backed by the registered model.
- **Ops:** Docker Compose runs Streamlit + FastAPI + MLflow together;
  GitHub Actions lints, tests, and builds the pipeline on every push.

---

## 🏗️ System Architecture

```text
                  ELECTRICITY DATA (IoT-style, appliance-level)
                             │
                             ▼
                     ┌───────────────┐
                     │    PySpark    │
                     │  BDA Pipeline │
                     └───────┬───────┘
                             │  clean + aggregate
                             ▼
                ┌────────────────────────┐
                │      BDA Analytics     │
                │  hourly / daily / peak │
                │  appliance / cost      │
                └────────────┬───────────┘
                             │
                             ▼
                     Feature Engineering
                             │
                             ▼
                     ┌───────────────┐
                     │  ML Training  │  RandomForest / XGBoost / IsolationForest
                     └───────┬───────┘
                             │
                             ▼
                        ┌─────────┐
                        │ MLflow  │  experiment tracking + model registry
                        └────┬────┘
                             │
                             ▼
                          FastAPI       /predict  /detect-anomaly
                             │
                             ▼
                        Streamlit UI
                             │
                             ▼
                     🌐 Web Dashboard
```

---

## 📂 Project Structure

```text
Smart-Home-Energy-Monitoring-System/
│
├── app.py                     # Live monitoring dashboard (original simulator)
├── pages/                     # Streamlit multi-page dashboard
│   ├── 1_📊_BDA_Analytics.py
│   ├── 2_🤖_ML_Prediction.py
│   ├── 3_🧪_MLOps.py
│   └── 4_⚙️_System.py
│
├── src/                       # Original core engine + shared helpers
│   ├── simulator.py, energy_engine.py, alerts.py, logger.py, config.py
│   ├── forecast.py, report_generator.py
│   └── api_client.py          # Streamlit -> FastAPI client
│
├── bda/                       # Big Data Analytics (PySpark)
│   ├── data_generator.py      # Synthetic appliance-level dataset generator
│   ├── spark_session.py
│   ├── data_cleaning.py       # Cleaning / dedup / outlier handling
│   ├── aggregations.py        # Hourly / daily / monthly / appliance aggregations
│   ├── analytics.py           # Peak usage, cost analysis, trend detection
│   └── spark_processing.py    # Orchestrates the full BDA pipeline
│
├── ml/                        # Machine learning + MLOps
│   ├── feature_engineering.py
│   ├── train.py                # RandomForest + XGBoost, MLflow tracking + registry
│   ├── anomaly_detection.py    # IsolationForest, MLflow tracking
│   └── artifacts/              # best_model.pkl, anomaly_model.pkl (generated)
│
├── api/
│   └── main.py                 # FastAPI: /predict, /detect-anomaly, /model/info
│
├── docker/
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   └── docker-compose.yml
│
├── .github/workflows/ci.yml    # Lint, test, smoke-run pipeline, build images
├── tests/                      # pytest unit tests
├── data/
│   ├── raw/                    # Generated synthetic electricity data
│   └── processed/              # Spark pipeline output (CSV/parquet/JSON)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Run

### 1. Clone and set up

```bash
git clone https://github.com/arshkaur2405/Smart-Home-Energy-Monitoring-System.git
cd Smart-Home-Energy-Monitoring-System
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

PySpark requires a JRE — install one (`sudo apt install default-jre` /
`brew install openjdk`) if you don't already have Java available.

### 2. Build the data + ML pipeline

```bash
# Generate a synthetic 90-day, appliance-level dataset
python -m bda.data_generator --days 90

# Run the PySpark BDA pipeline: cleaning + aggregations + analytics
python -m bda.spark_processing

# Build ML features and train models (tracked in MLflow)
python -m ml.feature_engineering
python -m ml.train
python -m ml.anomaly_detection
```

### 3. Start the services

```bash
# Terminal 1 — prediction API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — dashboard
streamlit run app.py
```

Open the Streamlit sidebar to navigate between **Dashboard**, **BDA
Analytics**, **ML Prediction**, **MLOps**, and **System** pages.

### 4. (Optional) Inspect experiments directly in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 🐳 Run Everything with Docker

```bash
cd docker
docker compose up --build
```

This starts:
- **Streamlit** dashboard → http://localhost:8501
- **FastAPI** prediction service → http://localhost:8000/docs
- **MLflow** tracking server → http://localhost:5000

Run the pipeline commands from step 2 above (locally, or via
`docker compose exec streamlit ...`) before starting the containers so
there's a trained model for the API to serve.

---

## 🔄 CI/CD

Every push to `main` triggers `.github/workflows/ci.yml`, which:

1. Lints the codebase with `flake8`
2. Runs the `pytest` unit tests
3. Smoke-tests the full pipeline (data generation → Spark BDA → ML
   training → anomaly detection)
4. Builds the Streamlit, FastAPI, and MLflow Docker images

---

## 🔥 BDA Component

`bda/spark_processing.py` runs a full PySpark job over the appliance-level
dataset and produces:

- **Hourly consumption** — total/average energy by hour of day
- **Daily consumption** — total energy, average and peak power per day
- **Monthly consumption** — total energy per month
- **Appliance analysis** — % share of total energy per appliance
- **Peak usage** — peak hour, peak day, highest-consuming appliance, single
  highest instantaneous reading
- **Cost analysis** — total/average/estimated monthly & yearly electricity
  cost and carbon emissions
- **Trend detection** — whether consumption is rising, falling, or stable

## 🤖 ML Component

- **Energy Consumption Prediction** — RandomForest and XGBoost regressors
  predict next-hour household energy use from hour, day-of-week, month,
  temperature, humidity, previous consumption, and active-appliance count.
- **Anomaly Detection** — an IsolationForest flags each reading as
  🟢 Normal, 🟡 Unusual, or 🔴 High.

## 🧪 MLOps Component

- **MLflow** tracks every training run's parameters, metrics, and model
  artifacts for both the forecaster and the anomaly detector.
- The best-performing forecasting model is registered in the **MLflow
  Model Registry** as `energy-consumption-predictor`.
- **FastAPI** loads the registered/best model and serves it over HTTP.
- **Docker Compose** containerizes Streamlit, FastAPI, and MLflow.
- **GitHub Actions** automates linting, testing, pipeline smoke-tests, and
  image builds on every push.

---

## 🎓 Viva / Presentation Talking Points

**"Where is the Big Data Analytics component?"**
> We use PySpark to clean and aggregate large-scale, appliance-level
> electricity consumption data — computing hourly, daily, monthly,
> appliance-level and cost analytics, plus peak-usage detection, all as
> distributed Spark DataFrame operations.

**"Where is the MLOps component?"**
> Training experiments (RandomForest and XGBoost for forecasting,
> IsolationForest for anomaly detection) are tracked with MLflow, with the
> best model versioned through the MLflow Model Registry. FastAPI serves
> the registered model for real-time inference, Docker containerizes all
> three services, and GitHub Actions automates linting, testing, and
> image builds on every push.

**"Where is the website?"**
> The Streamlit dashboard visualizes the BDA results, lets users request
> predictions and anomaly checks through the FastAPI service, and
> surfaces MLflow experiment history and model registry status directly
> in the UI.

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Dashboard | Streamlit, Plotly |
| Big Data Analytics | Apache PySpark |
| ML | scikit-learn (RandomForest, IsolationForest), XGBoost |
| MLOps | MLflow (tracking + model registry) |
| Model Serving | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Original core engine | Python, Pandas |

---

## 🔮 Future Enhancements

- ESP32 / MQTT integration for real sensor data instead of the synthetic
  generator
- Deploy the MLflow Model Registry + FastAPI service to a cloud provider
- Feature store for reusable, versioned features across models
- Drift detection comparing live predictions against training-time
  distributions
- Mobile push notifications for high-usage / anomaly alerts

---

## 👨‍💻 Author

Arshdeep Kaur — B.Tech Student
Smart Home Energy Analytics & MLOps Platform

---

## ⭐ Support

If you found this project useful: star it, fork it, share it.
