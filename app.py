import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from src.config import TARIFF_RATE, CARBON_FACTOR
from src.energy_engine import EnergyEngine
from src.alerts import AlertEngine


# ============================================================
# PAGE CONFIG
# ============================================================

# Page config
st.set_page_config(
    page_title="Smart Energy Monitoring System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HOUSE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "refit"
    / "house_consumption.json"
)

APPLIANCE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "refit"
    / "appliance_consumption.json"
)

HOURLY_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "refit"
    / "hourly_consumption.json"
)

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2533;
        border: 1px solid #30394a;
        border-radius: 15px;
        padding: 15px;
    }
    .info-box {
        background: #18202d;
        border: 1px solid #30394a;
        padding: 18px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD REFIT DATA
# ============================================================

@st.cache_data
def load_refit():
    house = pd.read_json(HOUSE_FILE)
    appliance = pd.read_json(APPLIANCE_FILE)
    hourly = pd.read_json(HOURLY_FILE)
    return house, appliance, hourly


@st.cache_data
def load_features():
    if FEATURE_FILE.exists():
        return pd.read_csv(FEATURE_FILE)
    return pd.DataFrame()


try:
    house_df, appliance_df, hourly_df = load_refit()
    features_df = load_features()
except Exception as e:
    st.error("REFIT analytics files could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

# Auto-detect column naming differences across pipelines
house_energy_col = "estimated_energy_kwh" if "estimated_energy_kwh" in house_df.columns else "total_energy_kwh"
app_energy_col = "estimated_energy_kwh" if "estimated_energy_kwh" in appliance_df.columns else "total_energy_kwh"

house_df = house_df.sort_values("house_id").reset_index(drop=True)
appliance_df = appliance_df.sort_values(app_energy_col, ascending=False).reset_index(drop=True)
hourly_df = hourly_df.sort_values("hour").reset_index(drop=True)


# ============================================================
# REFIT APPLIANCE LIST
# ============================================================

REFIT_APPLIANCES = {}

for _, row in appliance_df.iterrows():
    appliance_name = str(row["appliance"]).strip()

    if not appliance_name or appliance_name.lower() == "unknown":
        continue

    try:
        power_value = float(row["avg_power_w"])
    except Exception:
        power_value = 0.0

    if power_value < 0:
        power_value = 0.0

    REFIT_APPLIANCES[appliance_name] = power_value


# ============================================================
# SAFETY CHECK
# ============================================================

if not REFIT_APPLIANCES:
    st.error("No cleaned REFIT appliance categories were found.")
    st.info("Run: python -m bda.refit_analytics")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "generated" not in st.session_state:
    st.session_state.generated = False

if "reading" not in st.session_state:
    st.session_state.reading = None


# ============================================================
# OBJECTS
# ============================================================

engine = EnergyEngine()
alerts = AlertEngine()


# ============================================================
# SIDEBAR — REFIT APPLIANCE CONTROL
# ============================================================

st.sidebar.title("🏠 Appliance Control")
st.sidebar.caption(f"REFIT Appliance Categories: {len(REFIT_APPLIANCES)}")

default_appliance = ["Fridge"] if "Fridge" in REFIT_APPLIANCES else ([list(REFIT_APPLIANCES.keys())[0]] if REFIT_APPLIANCES else [])

selected_appliances = st.sidebar.multiselect(
    "Choose Appliances",
    options=list(REFIT_APPLIANCES.keys()),
    default=default_appliance
)

usage_hours = st.sidebar.number_input(
    "Usage Duration (hours)",
    min_value=0.1,
    max_value=24.0,
    value=1.0,
    step=0.5
)

generate = st.sidebar.button(
    "🚀 Generate Reading",
    use_container_width=True
)


# ============================================================
# GENERATE READING
# ============================================================

if generate:
    if not selected_appliances:
        st.warning("Please select at least one appliance.")
    else:
        total_power = sum(
            REFIT_APPLIANCES[a]
            for a in selected_appliances
            if a in REFIT_APPLIANCES
        )

        voltage = round(float(np.random.normal(230, 3)), 2)
        if voltage <= 0:
            voltage = 230.0

        current = round(total_power / voltage, 2)
        energy = round(total_power * usage_hours / 1000, 2)
        cost = round(energy * TARIFF_RATE, 2)
        carbon = round(energy * CARBON_FACTOR, 2)

        # Efficiency grading
        if total_power <= 500:
            score = 95
            grade = "A+"
            message = "Excellent efficiency. Home is operating in low-consumption mode."
        elif total_power <= 1000:
            score = 85
            grade = "A"
            message = "Good efficiency. Energy consumption is within a reasonable range."
        elif total_power <= 1800:
            score = 70
            grade = "B"
            message = "Moderate consumption detected. Consider reducing unnecessary usage."
        elif total_power <= 2500:
            score = 55
            grade = "C"
            message = "High consumption detected. Consider switching off high-power appliances."
        else:
            score = 35
            grade = "D"
            message = "Very high consumption detected. Reduce heavy appliance usage."

        status = "Normal"
        if total_power > 2500:
            status = "High Usage"
        elif total_power > 1800:
            status = "Moderate Usage"

        st.session_state.reading = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "power": total_power,
            "voltage": voltage,
            "current": current,
            "energy": energy,
            "cost": cost,
            "carbon": carbon,
            "score": score,
            "grade": grade,
            "message": message,
            "status": status,
            "appliances": selected_appliances,
            "hours": usage_hours
        }
        st.session_state.generated = True


reading = st.session_state.reading


# ============================================================
# HEADER (FIXED: Native Markdown avoids literal HTML codebox)
# ============================================================

st.title("⚡ Smart Energy Monitoring System")
st.markdown("### AI Powered Home Energy Analytics Dashboard")
st.write("")


# ============================================================
# LIVE MONITORING
# ============================================================

st.subheader("⚡ Live Monitoring")

c1, c2, c3, c4 = st.columns(4)

if reading:
    c1.metric("Power", f"{reading['power']:.2f} W")
    c2.metric("Voltage", f"{reading['voltage']:.2f} V")
    c3.metric("Current", f"{reading['current']:.2f} A")
    c4.metric("Cost", f"₹ {reading['cost']:.2f}")

    st.progress(reading["score"] / 100)
    st.success(f"Energy Rating: {reading['grade']}  |  {reading['message']}")
else:
    c1.metric("Power", "0.00 W")
    c2.metric("Voltage", "230.00 V")
    c3.metric("Current", "0.00 A")
    c4.metric("Cost", "₹ 0.00")
    st.info("Select appliances from Appliance Control and click Generate Reading.")


# ============================================================
# ENERGY SUMMARY
# ============================================================

st.subheader("💰 Energy Summary")

energy_used = reading["energy"] if reading else 0
current_bill = reading["cost"] if reading else 0
refit_total_energy = house_df[house_energy_col].sum()

monthly_prediction = current_bill * 30
yearly_prediction = current_bill * 365

e1, e2, e3, e4 = st.columns(4)
e1.metric("Energy Used", f"{energy_used:.2f} kWh")
e2.metric("Current Bill", f"₹ {current_bill:.2f}")
e3.metric("Monthly Prediction", f"₹ {monthly_prediction:.2f}")
e4.metric("Yearly Prediction", f"₹ {yearly_prediction:.2f}")


# ============================================================
# SMART HOME OVERVIEW
# ============================================================

st.subheader("🏠 Smart Home Overview")

connected_devices = len(REFIT_APPLIANCES)
running_devices = len(reading["appliances"]) if reading else 0
potential_savings = current_bill * 0.15 if current_bill > 0 else 0

o1, o2, o3, o4 = st.columns(4)
o1.metric("Connected Devices", connected_devices)
o2.metric("Running Devices", running_devices)
o3.metric("Potential Savings", f"₹ {potential_savings:.2f}")
o4.metric("System Health", "Healthy")


# ============================================================
# APPLIANCE ANALYTICS
# ============================================================

st.subheader("⚡ Appliance Analytics")

if reading:
    appliance_status = [
        {
            "Appliance": appliance,
            "REFIT Avg Power (W)": round(watt, 2),
            "Status": "ON" if appliance in reading["appliances"] else "OFF"
        }
        for appliance, watt in REFIT_APPLIANCES.items()
    ]
    status_df = pd.DataFrame(appliance_status)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
else:
    st.info("Generate a reading to see appliance status.")


# ============================================================
# GENERATED READING DETAILS
# ============================================================

if reading:
    st.subheader("🔌 Generated Appliance Reading")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Selected Appliances", len(reading["appliances"]))
    r2.metric("Total Power", f"{reading['power']:.2f} W")
    r3.metric("Usage Duration", f"{reading['hours']} h")
    r4.metric("Energy Generated", f"{reading['energy']:.2f} kWh")
    st.caption("Selected: " + ", ".join(reading["appliances"]))


# ============================================================
# REFIT BDA — HOUSE ANALYTICS
# ============================================================

st.subheader("🏘️ REFIT Household BDA")
st.caption("Historical household energy analytics processed using Apache PySpark.")

highest_house_row = house_df.loc[house_df[house_energy_col].idxmax()]

h1, h2, h3, h4 = st.columns(4)
h1.metric("REFIT Houses", house_df["house_id"].nunique())
h2.metric("Highest Consumption", f"House {int(highest_house_row['house_id'])}")
h3.metric("Highest House Energy", f"{highest_house_row[house_energy_col]:,.2f} kWh")
h4.metric("REFIT Total Energy", f"{refit_total_energy:,.2f} kWh")

st.bar_chart(
    house_df[["house_id", house_energy_col]].set_index("house_id"),
    use_container_width=True
)


# ============================================================
# APPLIANCE BDA
# ============================================================

st.subheader("🔌 REFIT Appliance Consumption")

top_refit_appliance = appliance_df.iloc[0]

a1, a2, a3 = st.columns(3)
a1.metric("Highest Appliance", top_refit_appliance["appliance"])
a2.metric("Estimated Energy", f"{top_refit_appliance[app_energy_col]:,.2f} kWh")
a3.metric("Peak Power", f"{top_refit_appliance['peak_power_w']:,.2f} W")

st.bar_chart(
    appliance_df[["appliance", app_energy_col]].set_index("appliance"),
    use_container_width=True
)


# ============================================================
# HOURLY ANALYTICS
# ============================================================

st.subheader("🕒 Hourly Energy Pattern")

peak_hour_row = hourly_df.loc[hourly_df["avg_power_w"].idxmax()]

q1, q2, q3 = st.columns(3)
q1.metric("Peak Hour", f"{int(peak_hour_row['hour']):02d}:00")
q2.metric("Average Power", f"{peak_hour_row['avg_power_w']:,.2f} W")
q3.metric("Peak Power", f"{peak_hour_row['peak_power_w']:,.2f} W")

st.line_chart(
    hourly_df[["hour", "avg_power_w"]].set_index("hour"),
    use_container_width=True
)


# ============================================================
# PEAK USAGE
# ============================================================

st.subheader("⚠️ Peak Usage Analysis")

refit_peak_power = house_df["peak_power_w"].max()
refit_peak_house = house_df.loc[house_df["peak_power_w"].idxmax(), "house_id"]

p1, p2 = st.columns(2)
p1.metric("Maximum REFIT Power", f"{refit_peak_power:,.2f} W")
p2.metric("House", f"House {int(refit_peak_house)}")

if reading:
    for alert in alerts.check_alerts(reading["power"]):
        st.warning(alert)


# ============================================================
# ENVIRONMENTAL IMPACT
# ============================================================

st.subheader("🌱 Environmental Impact")

carbon = reading["carbon"] if reading else 0.0

v1, v2 = st.columns(2)
v1.metric("CO₂ Emission", f"{carbon:.2f} kg")
v2.metric("Carbon Factor", f"{CARBON_FACTOR} kg CO₂/kWh")


# ============================================================
# SMART RECOMMENDATIONS
# ============================================================

st.subheader("💡 Smart Recommendations")

if reading:
    if reading["power"] > 2500:
        st.error("Very high power consumption. Consider switching off high-power appliances.")
    elif reading["power"] > 1800:
        st.warning("High consumption detected. Reduce unnecessary appliance usage.")
    elif reading["power"] > 1000:
        st.info("Moderate energy consumption. Monitor high-power appliances.")
    else:
        st.success("Excellent efficiency. Home is operating in low-consumption mode.")
else:
    st.info("Generate an appliance reading to receive smart recommendations.")


# ============================================================
# BDA PROCESSING INFORMATION
# ============================================================

st.subheader("🔬 Big Data Analytics")

b1, b2, b3, b4 = st.columns(4)
b1.metric("REFIT Houses", house_df["house_id"].nunique())
b2.metric("Appliance Categories", appliance_df["appliance"].nunique())
b3.metric("Hourly Records", len(hourly_df))
b4.metric("ML Feature Records", f"{len(features_df):,}" if not features_df.empty else "N/A")




# ============================================================
# CONSUMPTION HISTORY
# ============================================================

st.subheader("📜 REFIT Consumption History")

available_cols = ["house_id", "avg_power_w", "peak_power_w", house_energy_col]
if "reading_count" in house_df.columns:
    available_cols.append("reading_count")

history_df = house_df[available_cols].copy()
col_rename = {
    "house_id": "House",
    "avg_power_w": "Average Power (W)",
    "peak_power_w": "Peak Power (W)",
    house_energy_col: "Estimated Energy (kWh)",
    "reading_count": "Readings"
}
history_df.rename(columns=col_rename, inplace=True)

st.dataframe(history_df, use_container_width=True, hide_index=True)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader("📥 Export Analytics")

csv = history_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📊 Download REFIT House Analytics",
    data=csv,
    file_name="refit_house_analytics.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Smart Energy Monitoring System | "
    "REFIT Electrical Load Measurements | "
    "Apache PySpark"
)