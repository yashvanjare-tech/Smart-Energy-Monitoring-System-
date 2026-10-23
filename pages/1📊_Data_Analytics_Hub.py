import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BDA Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

REFIT_DIR = "data/processed/refit"

HOUSE_PATH = os.path.join(
    REFIT_DIR,
    "house_consumption.json"
)

APPLIANCE_PATH = os.path.join(
    REFIT_DIR,
    "appliance_consumption.json"
)

HOURLY_PATH = os.path.join(
    REFIT_DIR,
    "hourly_consumption.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


# ============================================================
# HEADER
# ============================================================

st.title("📊 Data Analytics")

st.markdown(
    """
    **Smart Home Energy Monitoring System**

    Large-scale electricity data processed using
    **Apache PySpark** for household, appliance and
    time-based analytics.
    """
)

st.caption(
    "BDA Engine: Apache PySpark | "
    "Dataset: REFIT Electrical Load Measurements"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    HOUSE_PATH,
    APPLIANCE_PATH,
    HOURLY_PATH
]

if not all(os.path.exists(x) for x in required_files):

    st.error("REFIT BDA output files are missing.")

    st.code(
        "python -m bda.refit_analytics",
        language="powershell"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

house = load_json(HOUSE_PATH)

appliance = load_json(APPLIANCE_PATH)

hourly = load_json(HOURLY_PATH)


# ============================================================
# CONVERT POWER TOTAL → ESTIMATED ENERGY
# ============================================================
# REFIT readings are sampled every 8 seconds.
# Energy (kWh) = Power(W) × seconds / 3,600,000
#
# total_power_readings = SUM(power_w)
# ============================================================

if (
    "estimated_energy_kwh" not in house.columns
    and "total_power_readings" in house.columns
):

    house["estimated_energy_kwh"] = (
        house["total_power_readings"] * 8 / 3600000
    )


if (
    "estimated_energy_kwh" not in appliance.columns
    and "total_power_readings" in appliance.columns
):

    appliance["estimated_energy_kwh"] = (
        appliance["total_power_readings"] * 8 / 3600000
    )


if (
    "estimated_energy_kwh" not in hourly.columns
    and "total_power_readings" in hourly.columns
):

   hourly["estimated_energy_kwh"] = (
    hourly["avg_power_w"]
    * hourly["reading_count"]
    * 8
    / 3600000
)


# ============================================================
# DATA CLEANING
# ============================================================

house = house.dropna(
    subset=[
        "avg_power_w",
        "peak_power_w",
        "estimated_energy_kwh"
    ]
)

appliance = appliance.dropna(
    subset=[
        "avg_power_w",
        "peak_power_w",
        "estimated_energy_kwh"
    ]
)

hourly = hourly.dropna(
    subset=[
        "avg_power_w",
        "peak_power_w"
    ]
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_energy = house[
    "estimated_energy_kwh"
].sum()

average_house_power = house[
    "avg_power_w"
].mean()

highest_peak_power = house[
    "peak_power_w"
].max()

total_readings = house[
    "reading_count"
].sum()

top_house = house.loc[
    house["estimated_energy_kwh"].idxmax()
]

top_appliance = appliance.loc[
    appliance["estimated_energy_kwh"].idxmax()
]

peak_hour = hourly.loc[
    hourly["avg_power_w"].idxmax()
]


# ============================================================
# KPI
# ============================================================

st.subheader("📌 BDA Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Estimated Energy",
    f"{total_energy:,.2f} kWh"
)

c2.metric(
    "Average Household Power",
    f"{average_house_power:,.2f} W"
)

c3.metric(
    "Highest Peak Power",
    f"{highest_peak_power:,.0f} W"
)

c4.metric(
    "Total Power Readings",
    f"{total_readings:,}"
)


st.divider()


# ============================================================
# HOUSEHOLD ANALYSIS
# ============================================================

st.subheader("🏠 Household Energy Analysis")

house_chart = house.copy()

house_chart["House"] = (
    "House "
    + house_chart["house_id"].astype(str)
)

fig_house = px.bar(
    house_chart,
    x="House",
    y="estimated_energy_kwh",
    title="Estimated Energy Consumption by Household",
    labels={
        "House": "Household",
        "estimated_energy_kwh":
            "Estimated Energy (kWh)"
    }
)

st.plotly_chart(
    fig_house,
    use_container_width=True
)

st.success(
    f"Highest energy consumption: "
    f"House {int(top_house['house_id'])} "
    f"({top_house['estimated_energy_kwh']:,.2f} kWh)"
)


# ============================================================
# HOUSEHOLD TABLE
# ============================================================

st.subheader("📋 Household Analytics")

st.dataframe(
    house[
        [
            "house_id",
            "avg_power_w",
            "peak_power_w",
            "estimated_energy_kwh",
            "reading_count"
        ]
    ].rename(
        columns={
            "house_id": "House",
            "avg_power_w": "Average Power (W)",
            "peak_power_w": "Peak Power (W)",
            "estimated_energy_kwh":
                "Estimated Energy (kWh)",
            "reading_count": "Readings"
        }
    ),
    use_container_width=True
)


st.divider()


# ============================================================
# APPLIANCE ANALYSIS
# ============================================================

st.subheader("🔌 Appliance-Level Analysis")

top_n = st.slider(
    "Number of appliances",
    min_value=5,
    max_value=min(20, len(appliance)),
    value=min(10, len(appliance))
)

top_appliances = appliance.nlargest(
    top_n,
    "estimated_energy_kwh"
)

fig_appliance = px.bar(
    top_appliances.sort_values(
        "estimated_energy_kwh"
    ),
    x="estimated_energy_kwh",
    y="appliance",
    orientation="h",
    title=f"Top {top_n} Appliances by Estimated Energy",
    labels={
        "estimated_energy_kwh":
            "Estimated Energy (kWh)",
        "appliance":
            "Appliance"
    }
)

st.plotly_chart(
    fig_appliance,
    use_container_width=True
)

st.info(
    f"Top appliance: {top_appliance['appliance']} "
    f"({top_appliance['estimated_energy_kwh']:,.2f} kWh)"
)


# ============================================================
# APPLIANCE TABLE
# ============================================================

st.subheader("📋 Appliance Analytics")

st.dataframe(
    appliance[
        [
            "appliance",
            "avg_power_w",
            "peak_power_w",
            "estimated_energy_kwh",
            "reading_count"
        ]
    ].rename(
        columns={
            "appliance": "Appliance",
            "avg_power_w": "Average Power (W)",
            "peak_power_w": "Peak Power (W)",
            "estimated_energy_kwh":
                "Estimated Energy (kWh)",
            "reading_count": "Readings"
        }
    ),
    use_container_width=True
)


st.divider()


# ============================================================
# HOURLY ANALYSIS
# ============================================================

st.subheader("🕐 Hourly Power Consumption")

fig_hourly = px.line(
    hourly,
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
    fig_hourly,
    use_container_width=True
)

st.success(
    f"Highest average-power hour: "
    f"{int(peak_hour['hour'])}:00 "
    f"({peak_hour['avg_power_w']:,.2f} W)"
)


# ============================================================
# HOURLY TABLE
# ============================================================

st.subheader("📋 Hourly Analytics")

hourly_columns = [
    "hour",
    "avg_power_w",
    "peak_power_w",
    "reading_count"
]

st.dataframe(
    hourly[hourly_columns].rename(
        columns={
            "hour": "Hour",
            "avg_power_w": "Average Power (W)",
            "peak_power_w": "Peak Power (W)",
            "reading_count": "Readings"
        }
    ),
    use_container_width=True
)


st.divider()


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📈 Dataset Summary")

s1, s2, s3 = st.columns(3)

s1.metric(
    "Households Analysed",
    house["house_id"].nunique()
)

s2.metric(
    "Appliance Categories",
    appliance["appliance"].nunique()
)

s3.metric(
    "Hours Analysed",
    hourly["hour"].nunique()
)


st.divider()


# ============================================================
# BDA INSIGHTS
# ============================================================

st.subheader("💡 Big Data Analytics Insights")

st.markdown(
    f"""
### Key Findings

**🏠 Household Analysis**
- Highest consuming household:
  **House {int(top_house['house_id'])}**
- Estimated energy:
  **{top_house['estimated_energy_kwh']:,.2f} kWh**

**🔌 Appliance Analysis**
- Highest consuming appliance:
  **{top_appliance['appliance']}**
- Estimated energy:
  **{top_appliance['estimated_energy_kwh']:,.2f} kWh**

**🕐 Time Analysis**
- Highest average-power hour:
  **{int(peak_hour['hour'])}:00**
- Average power:
  **{peak_hour['avg_power_w']:,.2f} W**
"""
)


st.divider()



# ============================================================
# 🔍 INTERACTIVE REFIT BDA ANALYTICS EXPLORER
# ============================================================

st.markdown("---")

st.header("🔍 Interactive REFIT BDA Analytics Explorer")

st.caption(
    "Explore REFIT Big Data Analytics results using household, "
    "appliance and hourly consumption views."
)

# ------------------------------------------------------------
# CREATE SAFE COPIES
# ------------------------------------------------------------

explorer_house = house.copy()
explorer_appliance = appliance.copy()
explorer_hourly = hourly.copy()


# ============================================================
# 🎛️ ANALYTICS CONTROLS
# ============================================================

st.subheader("🎛️ Analytics Controls")

control_col1, control_col2, control_col3 = st.columns(3)


# ------------------------------------------------------------
# HOUSE SELECTION
# ------------------------------------------------------------

with control_col1:

    house_options = ["All Houses"] + sorted(
        explorer_house["house_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_house = st.selectbox(
        "🏠 Select House",
        house_options,
        key="refit_explorer_house"
    )


# ------------------------------------------------------------
# APPLIANCE SELECTION
# ------------------------------------------------------------

with control_col2:

    appliance_options = ["All Appliances"] + sorted(
        explorer_appliance["appliance"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_appliance = st.selectbox(
        "🔌 Select Appliance",
        appliance_options,
        key="refit_explorer_appliance"
    )


# ------------------------------------------------------------
# METRIC SELECTION
# ------------------------------------------------------------

with control_col3:

    selected_metric = st.selectbox(
        "📊 Select Metric",
        [
            "Estimated Energy (kWh)",
            "Average Power (W)",
            "Peak Power (W)",
            "Reading Count"
        ],
        key="refit_explorer_metric"
    )


# ============================================================
# 📊 METRIC MAPPING
# ============================================================

metric_map = {
    "Estimated Energy (kWh)": "estimated_energy_kwh",
    "Average Power (W)": "avg_power_w",
    "Peak Power (W)": "peak_power_w",
    "Reading Count": "reading_count"
}

metric_column = metric_map[selected_metric]


# ============================================================
# 🏠 HOUSE FILTER
# ============================================================

if selected_house == "All Houses":

    filtered_house = explorer_house.copy()

else:

    filtered_house = explorer_house[
        explorer_house["house_id"] == selected_house
    ].copy()


# ============================================================
# 🔌 APPLIANCE FILTER
# ============================================================

if selected_appliance == "All Appliances":

    filtered_appliance = explorer_appliance.copy()

else:

    filtered_appliance = explorer_appliance[
        explorer_appliance["appliance"] == selected_appliance
    ].copy()


# ============================================================
# 📌 SELECTED ANALYTICS KPI
# ============================================================

st.markdown("---")

st.subheader("📌 Selected Analytics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


# ------------------------------------------------------------
# KPI 1 - HOUSES
# ------------------------------------------------------------

with kpi1:

    st.metric(
        "🏠 Houses",
        f"{filtered_house['house_id'].nunique():,}"
    )


# ------------------------------------------------------------
# KPI 2 - APPLIANCES
# ------------------------------------------------------------

with kpi2:

    st.metric(
        "🔌 Appliances",
        f"{filtered_appliance['appliance'].nunique():,}"
    )


# ------------------------------------------------------------
# KPI 3 - HOUSE ENERGY
# ------------------------------------------------------------

with kpi3:

    if not filtered_house.empty:

        total_house_energy = filtered_house[
            "estimated_energy_kwh"
        ].sum()

        st.metric(
            "⚡ House Energy",
            f"{total_house_energy:,.2f} kWh"
        )

    else:

        st.metric(
            "⚡ House Energy",
            "0 kWh"
        )


# ------------------------------------------------------------
# KPI 4 - SELECTED METRIC
# ------------------------------------------------------------

with kpi4:

    if not filtered_appliance.empty:

        metric_value = filtered_appliance[
            metric_column
        ].max()

        if metric_column == "reading_count":

            display_value = f"{metric_value:,.0f}"

        elif metric_column == "estimated_energy_kwh":

            display_value = f"{metric_value:,.2f} kWh"

        else:

            display_value = f"{metric_value:,.2f} W"

        st.metric(
            f"📊 {selected_metric}",
            display_value
        )

    else:

        st.metric(
            f"📊 {selected_metric}",
            "0"
        )


# ============================================================
# 🏠 HOUSEHOLD ANALYSIS
# ============================================================

st.markdown("---")

st.subheader("🏠 Household BDA Analysis")

if not filtered_house.empty:

    house_chart = filtered_house[
        [
            "house_id",
            metric_column
        ]
    ].copy()

    house_chart = house_chart.sort_values(
        metric_column,
        ascending=False
    )

    st.bar_chart(
        house_chart.set_index("house_id")[metric_column],
        width="stretch"
    )

else:

    st.warning(
        "No household analytics available for the selected house."
    )


# ============================================================
# 🔌 APPLIANCE ANALYSIS
# ============================================================

st.markdown("---")

st.subheader("🔌 Appliance BDA Analysis")

if not filtered_appliance.empty:

    appliance_chart = filtered_appliance[
        [
            "appliance",
            metric_column
        ]
    ].copy()

    appliance_chart = appliance_chart.sort_values(
        metric_column,
        ascending=False
    )

    st.bar_chart(
        appliance_chart.set_index("appliance")[metric_column],
        width="stretch"
    )

else:

    st.warning(
        "No appliance analytics available for the selected appliance."
    )

# ============================================================
# ⏰ HOURLY REFIT ANALYSIS
# ============================================================

st.markdown("---")

st.subheader("⏰ Hourly Consumption Pattern")

if not explorer_hourly.empty:

    # Hourly data has its own available columns.
    # Do not use appliance/house metric_column here.

    hourly_metric_map = {
        "Average Power (W)": "avg_power_w",
        "Peak Power (W)": "peak_power_w",
        "Reading Count": "reading_count",
        "Estimated Energy (kWh)": "estimated_energy_kwh"
    }

    # Create estimated hourly energy if it does not already exist
    if (
        "estimated_energy_kwh" not in explorer_hourly.columns
        and "avg_power_w" in explorer_hourly.columns
        and "reading_count" in explorer_hourly.columns
    ):
        explorer_hourly["estimated_energy_kwh"] = (
            explorer_hourly["avg_power_w"]
            * explorer_hourly["reading_count"]
            * 8
            / 3600000
        )

    hourly_metric_column = hourly_metric_map[selected_metric]

    if hourly_metric_column in explorer_hourly.columns:

        hourly_chart = explorer_hourly[
            [
                "hour",
                hourly_metric_column
            ]
        ].copy()

        hourly_chart = hourly_chart.sort_values("hour")

        st.line_chart(
            hourly_chart.set_index("hour")[hourly_metric_column],
            width="stretch"
        )

    else:

        st.warning(
            f"{selected_metric} is not available in hourly analytics."
        )

else:

    st.warning(
        "No hourly analytics available."
    )





