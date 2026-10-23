"""
pages/2🤖_Smart_Predictions.py

REFIT-based ML prediction and anomaly detection UI.

Streamlit UI --> FastAPI --> REFIT-trained ML models
"""

import sys
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.append(".")

from src.api_client import api_health, predict, detect_anomaly


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart Energy Prediction")

FEATURES_PATH = "data/processed/features.csv"
APPLIANCE_PATH = "data/processed/refit/appliance_consumption.json"


# ============================================================
# LOAD REFIT FEATURES
# ============================================================

@st.cache_data
def load_refit_features():

    df = pd.read_csv(FEATURES_PATH)

    df["hour_timestamp"] = pd.to_datetime(
        df["hour_timestamp"]
    )

    return df


# ============================================================
# LOAD REFIT APPLIANCES
# ============================================================

@st.cache_data
def load_refit_appliances():

    df = pd.read_json(APPLIANCE_PATH)

    if "appliance" not in df.columns:
        return pd.DataFrame()

    if "avg_power_w" not in df.columns:
        return pd.DataFrame()

    df["appliance"] = (
        df["appliance"]
        .astype(str)
        .str.strip()
    )

    df["avg_power_w"] = pd.to_numeric(
        df["avg_power_w"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["appliance", "avg_power_w"]
    )

    df = df[
        (df["appliance"] != "") &
        (df["appliance"].str.lower() != "unknown")
    ]

    df = df.sort_values(
        "appliance"
    ).reset_index(drop=True)

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    refit_df = load_refit_features()

    refit_appliance_df = load_refit_appliances()

except Exception as e:

    st.error("REFIT ML data could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# REFIT APPLIANCE POWER MAP
# ============================================================

if refit_appliance_df.empty:

    st.error(
        "REFIT appliance analytics data is empty or invalid."
    )

    st.info(
        "Run: python -m bda.refit_analytics"
    )

    st.stop()


REFIT_APPLIANCES = {
    row["appliance"]: float(row["avg_power_w"])
    for _, row in refit_appliance_df.iterrows()
}


REFIT_APPLIANCE_NAMES = sorted(
    REFIT_APPLIANCES.keys()
)


# ============================================================
# DATE / TIME
# ============================================================

now = datetime.now()

today_date = now.strftime(
    "%d %B %Y"
)

current_hour = now.hour

next_hour = (
    current_hour + 1
) % 24

day_of_week = now.weekday()

month = now.month


# ============================================================
# HEADER INFORMATION
# ============================================================

st.info(
    f"📅 Today: {today_date} | "
    f"🕒 Current Hour: {current_hour:02d}:00 | "
    f"🔮 Predicting Next Hour: {next_hour:02d}:00"
)


# ============================================================
# REFIT DATA INFORMATION
# ============================================================

st.markdown(
    """
    Uses **REFIT-trained machine learning models** through the
    FastAPI model-serving layer.

    Historical electricity patterns come from the
    **REFIT Electrical Load Measurements dataset**.
    """
)

st.caption(
    f"REFIT Appliance Categories Available: "
    f"{len(REFIT_APPLIANCE_NAMES)}"
)


# ============================================================
# API HEALTH CHECK
# ============================================================

health = api_health()

if health.get("status") != "ok":

    st.error(
        "⚠️ Prediction API is not reachable."
    )

    st.code(
        "python -m uvicorn api.main:app --reload --port 8000",
        language="powershell"
    )

    st.caption(
        f"Details: {health.get('error', 'unknown error')}"
    )

    st.stop()


if not health.get("predictor_loaded"):

    st.warning(
        "API is running but the prediction model "
        "is not loaded."
    )


if not health.get("anomaly_model_loaded"):

    st.warning(
        "API is running but the anomaly model "
        "is not loaded."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "⚡ Energy Forecast",
        "🚨 Anomaly Detection"
    ]
)


# ============================================================
# ENERGY FORECAST
# ============================================================

with tab1:

    st.subheader(
        "🔮 Today's Next-Hour Energy Prediction"
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    st.write(
        f"📅 **Today:** {today_date}  |  "
        f"🕒 **Current Hour:** {current_hour:02d}:00"
    )


    prediction_hour = st.selectbox(
        "🔮 Select Prediction Hour",
        options=[
            (current_hour + i) % 24
            for i in range(1, 9)
        ],
        format_func=lambda h:
            f"{h:02d}:00",
        index=0
    )


    st.info(
        f"Prediction will be generated for "
        f"**{prediction_hour:02d}:00** "
        f"using historical REFIT patterns."
    )


    # --------------------------------------------------------
    # REFIT APPLIANCE SELECTION
    # --------------------------------------------------------

    st.markdown(
        "### 🏠 Select REFIT Appliances"
    )


    default_appliances = []

    if "Fridge" in REFIT_APPLIANCE_NAMES:
        default_appliances.append("Fridge")

    if "Computer" in REFIT_APPLIANCE_NAMES:
        default_appliances.append("Computer")


    selected_appliances = st.multiselect(
        "Choose appliances that you expect to use:",
        REFIT_APPLIANCE_NAMES,
        default=default_appliances
    )


    appliance_count = len(
        selected_appliances
    )


    # --------------------------------------------------------
    # SELECTED REFIT POWER
    # --------------------------------------------------------

    selected_power = sum(
        REFIT_APPLIANCES.get(
            appliance,
            0.0
        )
        for appliance in selected_appliances
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Active Appliances",
        appliance_count
    )


    c2.metric(
        "REFIT Average Appliance Power",
        f"{selected_power:,.2f} W"
    )


    # --------------------------------------------------------
    # REFIT HISTORICAL PATTERN
    # --------------------------------------------------------

    pattern_df = refit_df[
        (refit_df["hour"] == prediction_hour) &
        (refit_df["day_of_week"] == day_of_week) &
        (refit_df["month"] == month)
    ]


    # Fallback 1
    if pattern_df.empty:

        pattern_df = refit_df[
            (refit_df["hour"] == prediction_hour) &
            (refit_df["day_of_week"] == day_of_week)
        ]


    # Fallback 2
    if pattern_df.empty:

        pattern_df = refit_df[
            refit_df["hour"] == prediction_hour
        ]


    # Final safety fallback
    if pattern_df.empty:

        st.error(
            "No REFIT historical pattern is available "
            "for the selected prediction hour."
        )

        st.stop()


    # --------------------------------------------------------
    # BASE REFIT VALUES
    # --------------------------------------------------------

    base_avg_power = float(
        pattern_df["avg_power_w"].median()
    )


    base_peak_power = float(
        pattern_df["peak_power_w"].median()
    )


    previous_consumption = float(
        pattern_df["previous_consumption"].median()
    )


    # --------------------------------------------------------
    # APPLIANCE ADJUSTMENT
    # --------------------------------------------------------

    if appliance_count > 0:

        # REFIT historical power + selected appliance usage
        appliance_factor = (
            appliance_count / 4
        )

        avg_power_w = (
            base_avg_power *
            appliance_factor
        )

        peak_power_w = (
            base_peak_power *
            appliance_factor
        )

    else:

        avg_power_w = (
            base_avg_power * 0.25
        )

        peak_power_w = (
            base_peak_power * 0.25
        )


    # --------------------------------------------------------
    # PREDICTION INPUT
    # --------------------------------------------------------

    prediction_input = {

        "hour":
            prediction_hour,

        "day_of_week":
            day_of_week,

        "month":
            month,

        "avg_power_w":
            round(avg_power_w, 2),

        "peak_power_w":
            round(peak_power_w, 2),

        "previous_consumption":
            round(previous_consumption, 4),

        "appliance_count":
            appliance_count
    }


    # --------------------------------------------------------
    # SELECTED APPLIANCES
    # --------------------------------------------------------

    if selected_appliances:

        st.markdown(
            "### ⚡ Selected REFIT Appliances"
        )

        st.write(
            " • ".join(
                f"**{a}**"
                for a in selected_appliances
            )
        )

    else:

        st.warning(
            "No appliances selected. "
            "Prediction will assume low appliance usage."
        )


    # --------------------------------------------------------
    # PREDICTION INPUTS
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Prediction Inputs"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Average Power",
        f"{avg_power_w:,.2f} W"
    )


    c2.metric(
        "Peak Power",
        f"{peak_power_w:,.2f} W"
    )


    c3.metric(
        "Previous Hour",
        f"{previous_consumption:.2f} kWh"
    )


    c4.metric(
        "Appliances",
        appliance_count
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Energy Consumption",
        type="primary"
    ):

        try:

            result = predict(
                prediction_input
            )


            predicted = float(
                result[
                    "predicted_energy_kwh"
                ]
            )


            st.success(
                f"🔮 Predicted consumption for "
                f"**{prediction_hour:02d}:00**: "
                f"**{predicted:.2f} kWh**"
            )


            c1, c2 = st.columns(2)


            c1.metric(
                "Usage Status",
                result["status"]
            )


            c2.metric(
                "Model Used",
                result["model_used"]
            )


            # ------------------------------------------------
            # SMART SUGGESTION
            # ------------------------------------------------

            st.markdown(
                "### 💡 Smart Suggestion"
            )


            if predicted >= 7:

                st.error(
                    "🚨 High energy consumption is expected. "
                    "Try to avoid using multiple high-power "
                    "appliances together."
                )

            elif predicted >= 3:

                st.warning(
                    "⚠️ Moderate energy consumption is expected. "
                    "Consider reducing simultaneous appliance usage."
                )

            else:

                st.success(
                    "🟢 Low energy consumption is expected. "
                    "The selected appliance usage looks relatively low."
                )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

with tab2:

    st.subheader(
        "🚨 Smart Energy Anomaly Detection"
    )


    # --------------------------------------------------------
    # DATE / TIME
    # --------------------------------------------------------

    now = datetime.now()

    anomaly_date = now.strftime(
        "%d %B %Y"
    )

    anomaly_current_hour = now.hour

    anomaly_day = now.weekday()

    anomaly_month = now.month


    st.info(
        f"📅 Today: **{anomaly_date}** | "
        f"🕒 Current Hour: "
        f"**{anomaly_current_hour:02d}:00** | "
        f"📍 Day: **{[
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday',
            'Sunday'
        ][anomaly_day]}**"
    )


    st.markdown(
        """
        Check whether an observed energy reading is
        **Normal, Unusual, or High** based on the
        **REFIT-trained IsolationForest model**.
        """
    )


    # --------------------------------------------------------
    # TIME INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### 🕒 Reading Information"
    )


    col1, col2 = st.columns(2)


    a_hour = col1.selectbox(
        "Hour of Reading",
        options=list(range(24)),
        index=anomaly_current_hour,
        format_func=lambda h:
            f"{h:02d}:00",
        key="a_hour"
    )


    a_day = col2.selectbox(
        "Day of Week",
        options=list(range(7)),
        index=anomaly_day,
        format_func=lambda i: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ][i],
        key="a_day"
    )


    a_month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=anomaly_month - 1,
        format_func=lambda m: [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ][m - 1],
        key="a_month"
    )


    # --------------------------------------------------------
    # REFIT APPLIANCE SELECTION
    # --------------------------------------------------------

    st.markdown(
        "### 🏠 Appliances in This Reading"
    )


    selected_anomaly_appliances = st.multiselect(
        "Select REFIT appliances that were active:",
        REFIT_APPLIANCE_NAMES,
        default=default_appliances,
        key="anomaly_appliances"
    )


    anomaly_appliance_count = len(
        selected_anomaly_appliances
    )


    selected_anomaly_power = sum(
        REFIT_APPLIANCES.get(
            appliance,
            0.0
        )
        for appliance in selected_anomaly_appliances
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Active Appliances",
        anomaly_appliance_count
    )


    c2.metric(
        "REFIT Average Appliance Power",
        f"{selected_anomaly_power:,.2f} W"
    )


    # --------------------------------------------------------
    # POWER INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### ⚡ Power Information"
    )


    col1, col2 = st.columns(2)


    a_avg_power = col1.number_input(
        "Average Power (W)",
        value=float(
            max(
                selected_anomaly_power,
                100
            )
        ),
        min_value=0.0,
        step=10.0,
        key="a_avg_power"
    )


    a_peak_power = col2.number_input(
        "Peak Power (W)",
        value=float(
            max(
                selected_anomaly_power,
                200
            )
        ),
        min_value=0.0,
        step=50.0,
        key="a_peak_power"
    )


    # --------------------------------------------------------
    # CONSUMPTION INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### 🔋 Consumption Information"
    )


    col1, col2 = st.columns(2)


    a_prev = col1.number_input(
        "Previous Hour's Consumption (kWh)",
        value=1.0,
        min_value=0.0,
        step=0.1,
        key="a_prev"
    )


    a_actual = col2.number_input(
        "Actual Observed Consumption (kWh)",
        value=1.0,
        min_value=0.0,
        step=0.1,
        key="a_actual"
    )


    # --------------------------------------------------------
    # SELECTED APPLIANCES
    # --------------------------------------------------------

    if selected_anomaly_appliances:

        st.markdown(
            "### ⚡ Selected REFIT Appliances"
        )

        st.write(
            " • ".join(
                f"**{appliance}**"
                for appliance in selected_anomaly_appliances
            )
        )

    else:

        st.warning(
            "No appliances selected. "
            "The anomaly check will use zero active appliances."
        )


    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    if st.button(
        "🔍 Check for Anomaly",
        type="primary",
        key="check_anomaly_button"
    ):


        payload = {

            "hour":
                a_hour,

            "day_of_week":
                a_day,

            "month":
                a_month,

            "avg_power_w":
                a_avg_power,

            "peak_power_w":
                a_peak_power,

            "previous_consumption":
                a_prev,

            "appliance_count":
                anomaly_appliance_count,

            "actual_energy_kwh":
                a_actual,
        }


        try:

            result = detect_anomaly(
                payload
            )


            status = result["status"]


            icon = {

                "Normal": "🟢",

                "Unusual": "🟡",

                "High": "🔴"

            }.get(
                status,
                "⚪"
            )


            st.markdown("---")


            st.markdown(
                f"## {icon} Anomaly Status: **{status}**"
            )


            c1, c2, c3 = st.columns(3)


            c1.metric(
                "Anomaly Score",
                result["anomaly_score"]
            )


            c2.metric(
                "Outlier Detected",
                "Yes"
                if result["is_outlier"]
                else "No"
            )


            c3.metric(
                "Observed Energy",
                f"{a_actual:.2f} kWh"
            )


            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            if status == "Normal":

                st.success(
                    "🟢 The observed energy consumption "
                    "is within the normal range."
                )


            elif status == "Unusual":

                st.warning(
                    "🟡 The observed consumption is unusual "
                    "compared with learned REFIT energy patterns."
                )


            else:

                st.error(
                    "🔴 High energy consumption detected. "
                    "Review the active appliances and power usage."
                )


        except Exception as e:

            st.error(
                f"Anomaly detection failed: {e}"
            )