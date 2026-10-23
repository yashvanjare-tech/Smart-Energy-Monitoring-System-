"""
bda/analytics.py

Higher-level BDA insights built on top of the aggregation jobs:
peak usage, cost analysis, and trend summaries. These are the numbers
the Streamlit "BDA Analytics" page and the viva answers point to.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import TARIFF_RATE, CARBON_FACTOR
from bda import aggregations as agg


def peak_usage(df: DataFrame) -> dict:
    peak_hour_row = agg.hourly_consumption(df).orderBy(F.desc("total_energy_kwh")).first()
    peak_day_row = agg.daily_consumption(df).orderBy(F.desc("total_energy_kwh")).first()
    peak_appliance_row = agg.appliance_consumption(df).first()
    single_reading_peak = df.orderBy(F.desc("power")).select(
        "timestamp", "appliance", "power"
    ).first()

    return {
        "peak_hour": int(peak_hour_row["hour"]) if peak_hour_row else None,
        "peak_hour_energy_kwh": float(peak_hour_row["total_energy_kwh"]) if peak_hour_row else None,
        "peak_day": str(peak_day_row["date"]) if peak_day_row else None,
        "peak_day_energy_kwh": float(peak_day_row["total_energy_kwh"]) if peak_day_row else None,
        "peak_appliance": peak_appliance_row["appliance"] if peak_appliance_row else None,
        "peak_appliance_share_pct": float(peak_appliance_row["share_pct"]) if peak_appliance_row else None,
        "single_highest_reading_w": float(single_reading_peak["power"]) if single_reading_peak else None,
        "single_highest_reading_appliance": single_reading_peak["appliance"] if single_reading_peak else None,
        "single_highest_reading_time": str(single_reading_peak["timestamp"]) if single_reading_peak else None,
    }


def cost_analysis(df: DataFrame, tariff_rate: float = TARIFF_RATE) -> dict:
    total_energy = df.agg(F.sum("energy")).first()[0] or 0.0
    n_days = df.select("date").distinct().count() or 1

    avg_daily_energy = total_energy / n_days
    total_cost = total_energy * tariff_rate
    avg_daily_cost = avg_daily_energy * tariff_rate

    return {
        "total_energy_kwh": round(total_energy, 2),
        "total_cost_inr": round(total_cost, 2),
        "avg_daily_energy_kwh": round(avg_daily_energy, 3),
        "avg_daily_cost_inr": round(avg_daily_cost, 2),
        "estimated_monthly_cost_inr": round(avg_daily_cost * 30, 2),
        "estimated_yearly_cost_inr": round(avg_daily_cost * 365, 2),
        "total_carbon_kg": round(total_energy * CARBON_FACTOR, 2),
        "days_covered": n_days,
    }


def energy_trend(df: DataFrame) -> dict:
    """Simple trend signal: is consumption rising or falling recently?"""
    daily = agg.daily_consumption(df).orderBy("date")
    rows = daily.collect()
    if len(rows) < 4:
        return {"trend": "insufficient_data"}

    half = len(rows) // 2
    first_half_avg = sum(r["total_energy_kwh"] for r in rows[:half]) / half
    second_half_avg = sum(r["total_energy_kwh"] for r in rows[half:]) / (len(rows) - half)

    change_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg else 0

    return {
        "first_period_avg_kwh": round(first_half_avg, 3),
        "second_period_avg_kwh": round(second_half_avg, 3),
        "change_pct": round(change_pct, 1),
        "trend": "rising" if change_pct > 3 else ("falling" if change_pct < -3 else "stable"),
    }
