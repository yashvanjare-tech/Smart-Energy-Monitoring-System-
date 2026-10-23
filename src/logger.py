import os
import pandas as pd
from datetime import datetime
from src.config import ENERGY_LOG_FILE


class EnergyLogger:

    def __init__(self):

        os.makedirs("data", exist_ok=True)

        # ✅ ALWAYS ensure file exists WITH DATA STRUCTURE
        if not os.path.exists(ENERGY_LOG_FILE) or os.path.getsize(ENERGY_LOG_FILE) == 0:

            df = pd.DataFrame(columns=[
                "Timestamp",
                "Voltage",
                "Current",
                "Power",
                "Energy",
                "Cost",
                "Carbon"
            ])

            df.to_csv(ENERGY_LOG_FILE, index=False)

    def log_data(self, voltage, current, power, energy, cost, carbon):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_row = pd.DataFrame([{
            "Timestamp": timestamp,
            "Voltage": voltage,
            "Current": current,
            "Power": power,
            "Energy": energy,
            "Cost": cost,
            "Carbon": carbon
        }])

        # SAFE READ
        try:
            if os.path.exists(ENERGY_LOG_FILE) and os.path.getsize(ENERGY_LOG_FILE) > 0:
                df = pd.read_csv(ENERGY_LOG_FILE)
            else:
                df = pd.DataFrame(columns=new_row.columns)
        except:
            df = pd.DataFrame(columns=new_row.columns)

        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(ENERGY_LOG_FILE, index=False)