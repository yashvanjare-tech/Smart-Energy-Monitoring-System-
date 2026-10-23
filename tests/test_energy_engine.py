"""
tests/test_energy_engine.py

Unit tests for the original, still-in-use business logic (src/energy_engine.py,
src/alerts.py) plus the new ML feature-engineering helpers. Run with:

    pytest tests/ -v
"""

import pandas as pd
import pytest

from src.energy_engine import EnergyEngine
from src.alerts import AlertEngine


@pytest.fixture
def engine():
    return EnergyEngine()


def test_calculate_energy(engine):
    assert engine.calculate_energy(1000) == 1.0
    assert engine.calculate_energy(500) == 0.5


def test_calculate_cost(engine):
    # TARIFF_RATE = 8
    assert engine.calculate_cost(1.0) == 8.0


def test_efficiency_grade_bounds(engine):
    assert engine.efficiency_grade(95) == "A+"
    assert engine.efficiency_grade(85) == "A"
    assert engine.efficiency_grade(75) == "B"
    assert engine.efficiency_grade(65) == "C"
    assert engine.efficiency_grade(10) == "D"


def test_predict_monthly_bill(engine):
    assert engine.predict_monthly_bill(10) == 300


def test_alert_engine_overload():
    alerts = AlertEngine()
    result = alerts.check_alerts(2600)
    assert any("CRITICAL" in a for a in result)


def test_alert_engine_normal():
    alerts = AlertEngine()
    result = alerts.check_alerts(200)
    assert any("Excellent" in a for a in result)


def test_feature_engineering_columns():
    from ml.feature_engineering import engineer_features, FEATURE_COLUMNS

    sample = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "total_energy_kwh": [1.0] * 10,
        "total_power_w": [1000.0] * 10,
        "temperature": [28.0] * 10,
        "humidity": [60.0] * 10,
        "appliance_count": [3] * 10,
    })

    result = engineer_features(sample)
    for col in FEATURE_COLUMNS:
        assert col in result.columns
    assert not result["previous_consumption"].isna().any()
