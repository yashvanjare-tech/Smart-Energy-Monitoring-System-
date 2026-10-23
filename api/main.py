"""
api/main.py

FastAPI model-serving layer for the Smart Home Energy Analytics
& MLOps Platform.

Uses the models trained on REFIT-based hourly household features.

Run:
    python -m uvicorn api.main:app --reload --port 8000
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ARTIFACT_DIR = "ml/artifacts"

MODEL_PATH = f"{ARTIFACT_DIR}/best_model.pkl"
ANOMALY_PATH = f"{ARTIFACT_DIR}/anomaly_model.pkl"
TRAINING_SUMMARY_PATH = f"{ARTIFACT_DIR}/training_summary.json"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Smart Home Energy Analytics API",
    description=(
        "REFIT-based energy consumption forecasting and "
        "anomaly detection API for the Smart Home Energy "
        "Analytics & MLOps Platform."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL STATE
# ============================================================

_state = {
    "predictor": None,
    "anomaly": None,
}


def _load_artifacts():
    """Load trained model artifacts."""

    if os.path.exists(MODEL_PATH):
        _state["predictor"] = joblib.load(MODEL_PATH)

    if os.path.exists(ANOMALY_PATH):
        _state["anomaly"] = joblib.load(ANOMALY_PATH)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    _load_artifacts()


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day, 0-23",
    )

    day_of_week: int = Field(
        0,
        ge=0,
        le=6,
        description="0=Monday ... 6=Sunday",
    )

    month: int = Field(
        default_factory=lambda: datetime.now().month,
        ge=1,
        le=12,
    )

    avg_power_w: float = Field(
        ...,
        ge=0,
        description="Average household power during the hour, Watts",
    )

    peak_power_w: float = Field(
        ...,
        ge=0,
        description="Peak household power during the hour, Watts",
    )

    previous_consumption: float = Field(
        ...,
        ge=0,
        description="Previous hour's household energy use, kWh",
    )

    appliance_count: int = Field(
        3,
        ge=0,
        description="Number of active appliances during the hour",
    )


class PredictionResponse(BaseModel):
    predicted_energy_kwh: float
    status: str
    model_used: str


class AnomalyRequest(PredictionRequest):
    actual_energy_kwh: float = Field(
        ...,
        ge=0,
        description="Observed household energy use, kWh",
    )


class AnomalyResponse(BaseModel):
    status: str
    anomaly_score: float
    is_outlier: bool


# ============================================================
# STATUS
# ============================================================

def _status_from_prediction(kwh: float) -> str:

    if kwh >= 7:
        return "High Usage"

    if kwh >= 3:
        return "Moderate Usage"

    return "Low Usage"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Smart Home Energy Analytics API",
        "status": "running",
        "data_source": "REFIT",
        "model_type": "REFIT-trained ML models",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "predictor_loaded": _state["predictor"] is not None,
        "anomaly_model_loaded": _state["anomaly"] is not None,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model/info")
def model_info():

    if not os.path.exists(TRAINING_SUMMARY_PATH):

        raise HTTPException(
            status_code=404,
            detail=(
                "No training summary found. "
                "Run ml.train.py first."
            ),
        )

    with open(
        TRAINING_SUMMARY_PATH,
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(req: PredictionRequest):

    bundle = _state["predictor"]

    if bundle is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction model not loaded. "
                "Run ml.train.py first."
            ),
        )

    model = bundle["model"]
    features = bundle["features"]

    is_weekend = int(
        req.day_of_week in (5, 6)
    )

    row = pd.DataFrame(
        [
            {
                "hour": req.hour,
                "day_of_week": req.day_of_week,
                "month": req.month,
                "is_weekend": is_weekend,
                "avg_power_w": req.avg_power_w,
                "peak_power_w": req.peak_power_w,
                "previous_consumption": req.previous_consumption,
                "appliance_count": req.appliance_count,
            }
        ]
    )

    # Ensure exact feature order used during training.
    row = row[features]

    prediction = float(
        model.predict(row)[0]
    )

    prediction = max(
        0.0,
        round(prediction, 3),
    )

    return PredictionResponse(
        predicted_energy_kwh=prediction,
        status=_status_from_prediction(prediction),
        model_used=bundle.get(
            "model_name",
            "unknown",
        ),
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

@app.post(
    "/detect-anomaly",
    response_model=AnomalyResponse,
)
def detect_anomaly(req: AnomalyRequest):

    bundle = _state["anomaly"]

    if bundle is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Anomaly model not loaded. "
                "Run ml.anomaly_detection.py first."
            ),
        )

    model = bundle["model"]
    features = bundle["features"]

    is_weekend = int(
        req.day_of_week in (5, 6)
    )

    row = pd.DataFrame(
        [
            {
                "hour": req.hour,
                "day_of_week": req.day_of_week,
                "month": req.month,
                "is_weekend": is_weekend,
                "avg_power_w": req.avg_power_w,
                "peak_power_w": req.peak_power_w,
                "previous_consumption": req.previous_consumption,
                "appliance_count": req.appliance_count,
            }
        ]
    )

    # Ensure exact feature order used during training.
    row = row[features]

    score = float(
        model.decision_function(row)[0]
    )

    is_outlier = bool(
        model.predict(row)[0] == -1
    )

    mean_energy = bundle.get(
        "mean",
        0,
    )

    std_energy = bundle.get(
        "std",
        0,
    )

    if std_energy:

        z = (
            req.actual_energy_kwh - mean_energy
        ) / std_energy

    else:

        z = 0

    if z > 2.5:

        status = "High"

    elif is_outlier or abs(z) > 1.5:

        status = "Unusual"

    else:

        status = "Normal"

    return AnomalyResponse(
        status=status,
        anomaly_score=round(score, 4),
        is_outlier=is_outlier,
    )