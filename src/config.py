# src/config.py

TARIFF_RATE = 8
CARBON_FACTOR = 0.82

POWER_THRESHOLD = 1800
OVERLOAD_THRESHOLD = 2500

VOLTAGE = 230

ENERGY_LOG_FILE = "data/energy_log.csv"

APPLIANCES = {

    "LED Bulb": 15,
    "Fan": 80,
    "TV": 120,
    "Laptop": 65,

    "Refrigerator": 250,
    "Microwave": 1200,
    "Washing Machine": 500,

    "Air Conditioner": 1500,
    "Water Heater": 2000,

    "WiFi Router": 20,
    "Desktop PC": 250,
    "Air Purifier": 50,
    "Water Pump": 800
}