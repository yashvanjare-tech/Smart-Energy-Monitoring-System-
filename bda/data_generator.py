"""
bda/data_generator.py

Generates a large, realistic, appliance-level electricity consumption
dataset that acts as the "raw" data source for the Big Data Analytics
(PySpark) pipeline.

The original project only logged a handful of aggregate rows whenever a
user clicked a button in the Streamlit app. For a BDA component we need
a properly sized time-series dataset (many appliances x many hours x many
days), so this script simulates that data the same way an IoT sensor
network would have produced it.

Run:
    python -m bda.data_generator --days 90 --out data/raw/electricity_data.csv
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import APPLIANCES, VOLTAGE

# Rough duty-cycle (probability the appliance is drawing power in a given
# hour) and a per-appliance power variability factor. These are simple,
# explainable assumptions -- good enough for a mini-project simulation.
DUTY_CYCLE = {
    "LED Bulb": 0.55,
    "Fan": 0.45,
    "TV": 0.30,
    "Laptop": 0.35,
    "Refrigerator": 0.95,   # essentially always cycling
    "Microwave": 0.05,
    "Washing Machine": 0.06,
    "Air Conditioner": 0.30,
    "Water Heater": 0.15,
    "WiFi Router": 0.98,
    "Desktop PC": 0.25,
    "Air Purifier": 0.20,
    "Water Pump": 0.08,
}

# Appliances that are more likely to run in the evening peak (18-23h)
EVENING_BIASED = {"TV", "Air Conditioner", "Water Heater", "Washing Machine", "LED Bulb"}
# Appliances that run more in the morning (6-10h)
MORNING_BIASED = {"Water Heater", "Water Pump", "Microwave"}


def _hour_factor(hour: int, appliance: str) -> float:
    """Multiplier on duty-cycle probability based on time of day."""
    if appliance in EVENING_BIASED and 18 <= hour <= 23:
        return 1.8
    if appliance in MORNING_BIASED and 6 <= hour <= 9:
        return 1.6
    if appliance == "Air Conditioner" and 12 <= hour <= 17:
        return 1.5
    if 0 <= hour <= 5:
        return 0.4 if appliance != "Refrigerator" else 1.0
    return 1.0


def _seasonal_temperature(day_index: int, hour: int) -> float:
    """Simple daily + seasonal temperature curve (Celsius)."""
    seasonal = 27 + 6 * np.sin(2 * np.pi * day_index / 365)
    daily = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = random.uniform(-1.2, 1.2)
    return round(seasonal + daily + noise, 1)


def _humidity(temperature: float) -> float:
    base = 75 - (temperature - 27) * 1.5
    return round(min(95, max(25, base + random.uniform(-6, 6))), 1)


def generate(days: int = 90, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.now() - timedelta(days=days)
    rows = []

    for day_index in range(days):
        current_day = start + timedelta(days=day_index)
        temp_today = None

        for hour in range(24):
            ts = current_day.replace(hour=hour, minute=0, second=0, microsecond=0)
            temperature = _seasonal_temperature(day_index, hour)
            humidity = _humidity(temperature)

            for appliance, rated_watt in APPLIANCES.items():
                prob = min(0.99, DUTY_CYCLE[appliance] * _hour_factor(hour, appliance))
                is_on = random.random() < prob
                if not is_on:
                    continue

                fluctuation = random.uniform(0.85, 1.15)
                # Hotter days -> AC and fans draw a bit more
                if appliance in ("Air Conditioner", "Fan") and temperature > 30:
                    fluctuation *= 1.15

                power = round(rated_watt * fluctuation, 2)
                voltage = round(random.uniform(VOLTAGE - 10, VOLTAGE + 10), 2)
                current = round(power / voltage, 3) if voltage else 0
                energy = round(power / 1000, 4)  # kWh for that 1-hour reading

                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "appliance": appliance,
                    "voltage": voltage,
                    "current": current,
                    "power": power,
                    "energy": energy,
                    "temperature": temperature,
                    "humidity": humidity,
                })

    df = pd.DataFrame(rows)

    # Inject a small amount of "dirty" data so the cleaning step in the
    # BDA pipeline has something real to do -- this mirrors what actually
    # happens with IoT sensor feeds (dropouts, glitches, duplicates).
    n_dirty = max(1, int(len(df) * 0.01))
    dirty_idx = np.random.choice(df.index, size=n_dirty, replace=False)
    for i in dirty_idx:
        choice = random.random()
        if choice < 0.3:
            df.loc[i, "voltage"] = -1  # sensor glitch
        elif choice < 0.6:
            df.loc[i, "power"] = None  # missing reading
        elif choice < 0.85:
            df.loc[i, "current"] = df.loc[i, "current"] * 50  # spike / outlier
        else:
            df.loc[i, "appliance"] = None  # tag lost

    dup_rows = df.sample(n=max(1, int(len(df) * 0.005)), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)  # duplicate readings

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic smart-home electricity data")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=str, default="data/raw/electricity_data.csv")
    args = parser.parse_args()

    df = generate(days=args.days)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} rows across {args.days} days -> {args.out}")
