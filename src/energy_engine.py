from src.config import TARIFF_RATE, CARBON_FACTOR


class EnergyEngine:

    # ---------------- ENERGY ----------------

    def calculate_energy(self, power):
        """
        Convert power (W) to energy (kWh)
        """
        return round(power / 1000, 4)

    # ---------------- COST ----------------

    def calculate_cost(self, energy):
        """
        Electricity cost calculation
        """
        return round(energy * TARIFF_RATE, 2)

    # ---------------- CARBON ----------------

    def calculate_carbon(self, energy):
        """
        Carbon emission estimation
        """
        return round(energy * CARBON_FACTOR, 3)

    # ---------------- EFFICIENCY SCORE ----------------

    def efficiency_score(self, power):

        if power < 500:
            return 95

        elif power < 1000:
            return 85

        elif power < 1500:
            return 75

        elif power < 2000:
            return 65

        return 50

    # ---------------- EFFICIENCY GRADE ----------------

    def efficiency_grade(self, score):

        if score >= 90:
            return "A+"

        elif score >= 80:
            return "A"

        elif score >= 70:
            return "B"

        elif score >= 60:
            return "C"

        return "D"

    # ---------------- MONTHLY BILL PREDICTION ----------------

    def predict_monthly_bill(self, daily_cost):

        return round(daily_cost * 30, 2)

    # ---------------- ANNUAL BILL ----------------

    def predict_yearly_bill(self, daily_cost):

        return round(daily_cost * 365, 2)

    # ---------------- SAVINGS ESTIMATION ----------------

    def estimate_savings(self, current_bill):

        return round(current_bill * 0.15, 2)

    # ---------------- CARBON IMPACT ----------------

    def trees_required(self, carbon):

        return round(carbon / 21, 2)