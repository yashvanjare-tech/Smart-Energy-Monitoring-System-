from src.config import POWER_THRESHOLD, OVERLOAD_THRESHOLD


class AlertEngine:

    def check_alerts(self, power):

        alerts = []

        # Critical overload
        if power >= OVERLOAD_THRESHOLD:
            alerts.append(
                "🚨 CRITICAL ALERT: Potential overload detected. Immediate action recommended."
            )

        # High usage
        elif power >= POWER_THRESHOLD:
            alerts.append(
                "⚠ HIGH USAGE: Energy consumption is above the recommended threshold."
            )

        # Efficiency recommendations
        if power > 2000:
            alerts.append(
                "💡 Recommendation: Reduce Heater or AC usage to lower electricity costs."
            )

        elif power > 1500:
            alerts.append(
                "💡 Recommendation: Consider switching off non-essential appliances."
            )

        elif power < 300:
            alerts.append(
                "✅ Excellent efficiency. Home is operating in low-consumption mode."
            )

        return alerts