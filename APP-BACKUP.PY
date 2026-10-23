
import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.config import APPLIANCES, ENERGY_LOG_FILE
from src.simulator import EnergySimulator
from src.energy_engine import EnergyEngine
from src.logger import EnergyLogger
from src.alerts import AlertEngine

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

div[data-testid="metric-container"]{
    background-color:#1e2533;
    border:1px solid #2d3748;
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

sim = EnergySimulator()
engine = EnergyEngine()
logger = EnergyLogger()
alerts = AlertEngine()

# --------------------------------------------------
# FILE SAFETY
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

if not os.path.exists(ENERGY_LOG_FILE):

    pd.DataFrame(columns=[
        "Timestamp",
        "Voltage",
        "Current",
        "Power",
        "Energy",
        "Cost",
        "Carbon"
    ]).to_csv(ENERGY_LOG_FILE, index=False)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚡ Smart Energy Monitoring System")

st.markdown(
    "### AI Powered Home Energy Analytics Dashboard"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🏠 Appliance Control")

selected = st.sidebar.multiselect(
    "Choose Appliances",
    list(APPLIANCES.keys())
)

start = st.sidebar.button(
    "🚀 Generate Reading"
)

# --------------------------------------------------
# LIVE READING
# --------------------------------------------------

if start:

    data = sim.calculate_load(selected)

    voltage = data["voltage"]
    current = data["current"]
    power = data["power"]

    energy = engine.calculate_energy(power)

    cost = engine.calculate_cost(
        energy
    )

    carbon = engine.calculate_carbon(
        energy
    )

    score = engine.efficiency_score(
        power
    )

    grade = engine.efficiency_grade(
        score
    )

    logger.log_data(
        voltage,
        current,
        power,
        energy,
        cost,
        carbon
    )

    st.subheader("⚡ Live Monitoring")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Power",
        f"{power} W"
    )

    c2.metric(
        "Voltage",
        f"{voltage} V"
    )

    c3.metric(
        "Current",
        f"{current} A"
    )

    c4.metric(
        "Cost",
        f"₹ {cost}"
    )

    st.progress(score / 100)

    st.success(
        f"Energy Rating: {grade}"
    )

    for alert in alerts.check_alerts(power):
        st.error(alert)

# --------------------------------------------------
# LOAD HISTORY
# --------------------------------------------------

try:

    df = pd.read_csv(
        ENERGY_LOG_FILE
    )

except:

    df = pd.DataFrame(columns=[
        "Timestamp",
        "Voltage",
        "Current",
        "Power",
        "Energy",
        "Cost",
        "Carbon"
    ])

# --------------------------------------------------
# BILL SUMMARY
# --------------------------------------------------

if not df.empty:

    total_energy = df["Energy"].sum()

    total_cost = df["Cost"].sum()

    total_carbon = df["Carbon"].sum()

    monthly_bill = engine.predict_monthly_bill(
        total_cost
    )

    yearly_bill = engine.predict_yearly_bill(
        total_cost
    )

    savings = engine.estimate_savings(
        total_cost
    )

    trees = engine.trees_required(
        total_carbon
    )

    st.subheader("💰 Energy Summary")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Energy Used",
        f"{round(total_energy,2)} kWh"
    )

    b2.metric(
        "Current Bill",
        f"₹ {round(total_cost,2)}"
    )

    b3.metric(
        "Monthly Prediction",
        f"₹ {monthly_bill}"
    )

    b4.metric(
        "Yearly Prediction",
        f"₹ {yearly_bill}"
    )

# --------------------------------------------------
# SMART HOME OVERVIEW
# --------------------------------------------------

st.subheader("🏠 Smart Home Overview")

o1, o2, o3, o4 = st.columns(4)

o1.metric(
    "Connected Devices",
    len(APPLIANCES)
)

o2.metric(
    "Running Devices",
    len(selected)
)

o3.metric(
    "Potential Savings",
    f"₹ {round(savings,2)}"
    if not df.empty else "₹ 0"
)

o4.metric(
    "System Health",
    "Healthy"
)

# --------------------------------------------------
# APPLIANCE ANALYTICS
# --------------------------------------------------

st.subheader(
    "⚡ Appliance Analytics"
)

analytics = []

for appliance, watt in APPLIANCES.items():

    analytics.append({

        "Appliance": appliance,

        "Power(W)": watt,

        "Status":
        "ON"
        if appliance in selected
        else "OFF"
    })

analytics_df = pd.DataFrame(
    analytics
)

st.dataframe(
    analytics_df,
    use_container_width=True
)

# --------------------------------------------------
# APPLIANCE PIE CHART
# --------------------------------------------------

if selected:

    appliance_df = pd.DataFrame({

        "Appliance": selected,

        "Power": [
            APPLIANCES[a]
            for a in selected
        ]
    })

    fig = px.pie(
        appliance_df,
        names="Appliance",
        values="Power",
        hole=0.5,
        title="Appliance Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# ENERGY CHARTS
# --------------------------------------------------

if not df.empty:

    st.subheader(
        "📈 Energy Analytics"
    )

    fig1 = px.line(
        df,
        x="Timestamp",
        y="Power",
        title="Power Trend"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.line(
        df,
        x="Timestamp",
        y="Cost",
        title="Cost Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.line(
        df,
        x="Timestamp",
        y="Energy",
        title="Energy Usage Trend"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# --------------------------------------------------
# PEAK USAGE
# --------------------------------------------------

if not df.empty:

    peak_row = df.loc[
        df["Power"].idxmax()
    ]

    st.subheader(
        "⚠ Peak Usage Analysis"
    )

    st.warning(
        f"""
Peak Power:
{peak_row['Power']} W

Time:
{peak_row['Timestamp']}
"""
    )

# --------------------------------------------------
# ENVIRONMENTAL IMPACT
# --------------------------------------------------

if not df.empty:

    st.subheader(
        "🌱 Environmental Impact"
    )

    e1, e2 = st.columns(2)

    e1.metric(
        "CO₂ Emission",
        f"{round(total_carbon,2)} kg"
    )

    e2.metric(
        "Trees Needed",
        trees
    )

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

st.subheader(
    "💡 Smart Recommendations"
)

if not df.empty:

    latest_power = df.iloc[-1]["Power"]

    if latest_power > 1800:

        st.error(
            "Reduce AC or Heater usage to lower electricity bills."
        )

    elif latest_power > 1000:

        st.warning(
            "Moderate energy usage detected."
        )

    else:

        st.success(
            "Excellent energy efficiency."
        )

# --------------------------------------------------
# HISTORY
# --------------------------------------------------

st.subheader(
    "📜 Consumption History"
)

if not df.empty:

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )
    # --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------

st.subheader("📥 Export Data")

if not df.empty:

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📊 Download Energy Log CSV",
        data=csv,
        file_name="energy_usage_report.csv",
        mime="text/csv"
    )
    # ...


# ============================================================
# 📊 REFIT HISTORICAL BDA ANALYTICS
# ============================================================

st.divider()

st.title("📊 REFIT Historical BDA Analytics")

st.caption(
    "Historical energy analytics generated from the official "
    "REFIT household dataset using Apache PySpark."
)

REFIT_DIR = "data/processed/refit"


@st.cache_data
def load_refit_analytics():

    house_df = pd.read_json(
        f"{REFIT_DIR}/house_consumption.json"
    )

    appliance_df = pd.read_json(
        f"{REFIT_DIR}/appliance_consumption.json"
    )

    hourly_df = pd.read_json(
        f"{REFIT_DIR}/hourly_consumption.json"
    )

    return house_df, appliance_df, hourly_df


try:

    house_df, appliance_df, hourly_df = load_refit_analytics()

    # --------------------------------------------------------
    # REFIT KPIs
    # --------------------------------------------------------

    st.subheader("🏠 REFIT Dataset Overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "🏠 Houses",
        house_df["house_id"].nunique()
    )

    k2.metric(
        "🔌 Appliances",
        appliance_df["appliance"].nunique()
    )

    k3.metric(
        "📊 House Records",
        f"{len(house_df):,}"
    )

    k4.metric(
        "🕒 Hourly Records",
        f"{len(hourly_df):,}"
    )

    # --------------------------------------------------------
    # HOUSE ANALYTICS
    # --------------------------------------------------------

    st.subheader("🏠 House-wise Energy Analytics")

    display_house = house_df.copy()

    st.dataframe(
        display_house,
        use_container_width=True,
        hide_index=True
    )

    # Highest average power house
    top_house = house_df.loc[
        house_df["avg_power_w"].idxmax()
    ]

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Highest Avg Power House",
        f"House {int(top_house['house_id'])}"
    )

    h2.metric(
        "Average Power",
        f"{top_house['avg_power_w']:,.2f} W"
    )

    h3.metric(
        "Peak Power",
        f"{top_house['peak_power_w']:,.2f} W"
    )

    # --------------------------------------------------------
    # HOUSE POWER CHART
    # --------------------------------------------------------

    fig_house = px.bar(
        house_df,
        x="house_id",
        y="avg_power_w",
        title="Average Power Consumption by House",
        labels={
            "house_id": "House",
            "avg_power_w": "Average Power (W)"
        }
    )

    st.plotly_chart(
        fig_house,
        use_container_width=True
    )

    # --------------------------------------------------------
    # APPLIANCE ANALYTICS
    # --------------------------------------------------------

    st.subheader("🔌 Appliance-wise BDA Analytics")

    st.dataframe(
        appliance_df,
        use_container_width=True,
        hide_index=True
    )

    fig_appliance = px.bar(
        appliance_df,
        x="appliance",
        y="avg_power_w",
        title="Average Power Consumption by Appliance",
        labels={
            "appliance": "Appliance",
            "avg_power_w": "Average Power (W)"
        }
    )

    st.plotly_chart(
        fig_appliance,
        use_container_width=True
    )

    # --------------------------------------------------------
    # HOURLY ANALYTICS
    # --------------------------------------------------------

    st.subheader("🕒 Hour-wise REFIT Energy Pattern")

    fig_hour = px.line(
        hourly_df,
        x="hour",
        y="avg_power_w",
        markers=True,
        title="Average Power Consumption by Hour",
        labels={
            "hour": "Hour of Day",
            "avg_power_w": "Average Power (W)"
        }
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PEAK HOUR
    # --------------------------------------------------------

    peak_hour = hourly_df.loc[
        hourly_df["avg_power_w"].idxmax()
    ]

    st.info(
        f"⏰ **Peak Average Usage Hour:** "
        f"{int(peak_hour['hour']):02d}:00  |  "
        f"Average Power: "
        f"{peak_hour['avg_power_w']:,.2f} W"
    )

    # --------------------------------------------------------
    # BDA EXPLANATION
    # --------------------------------------------------------

    st.subheader("🔬 Big Data Analytics Processing")

    st.markdown(
        """
        **REFIT → Apache PySpark → Data Processing → Aggregation → Analytics → Dashboard**

        - 🧹 Data cleaning and transformation
        - 🔄 Wide-to-long data transformation
        - 🏠 House-wise aggregation
        - 🔌 Appliance-wise aggregation
        - 🕒 Hour-wise time-series analysis
        - 📊 Statistical analysis and visualization
        - 💾 Processed analytical outputs
        """
    )

except Exception as e:

    st.error(
        "REFIT analytics data could not be loaded."
    )

    st.caption(
        f"Details: {e}"
    )


