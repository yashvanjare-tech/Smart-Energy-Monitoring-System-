"""
src/api_client.py

Thin HTTP client the Streamlit dashboard uses to talk to the FastAPI
model-serving layer (api/main.py). Kept separate from the Streamlit
pages so both the app and any tests can reuse it.
"""

import os

import requests

API_BASE_URL = os.environ.get("ENERGY_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 5


def api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"status": "unreachable", "error": str(e)}


def model_info():
    try:
        r = requests.get(f"{API_BASE_URL}/model/info", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def predict(payload: dict):
    r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def detect_anomaly(payload: dict):
    r = requests.post(f"{API_BASE_URL}/detect-anomaly", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
