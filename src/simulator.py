import random
from datetime import datetime

from src.config import APPLIANCES, VOLTAGE


class EnergySimulator:

    def calculate_load(self, selected_appliances):

        current_hour = datetime.now().hour

        # Base appliance load
        base_load = sum(
            APPLIANCES[a]
            for a in selected_appliances
        )

        # ---------------------------
        # Time-based household profile
        # ---------------------------

        if 6 <= current_hour < 10:
            # Morning
            usage_factor = random.uniform(1.0, 1.2)

        elif 10 <= current_hour < 17:
            # Afternoon
            usage_factor = random.uniform(0.7, 0.95)

        elif 17 <= current_hour < 23:
            # Evening peak
            usage_factor = random.uniform(1.2, 1.5)

        else:
            # Night
            usage_factor = random.uniform(0.5, 0.8)

        # Appliance fluctuation
        fluctuation = random.uniform(0.95, 1.05)

        power = base_load * usage_factor * fluctuation

        # Voltage fluctuation
        voltage = random.uniform(
            VOLTAGE - 10,
            VOLTAGE + 10
        )

        current = (
            power / voltage
            if voltage > 0
            else 0
        )

        return {
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power": round(power, 2)
        }