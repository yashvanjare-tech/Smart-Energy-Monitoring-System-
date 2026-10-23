"""
PDF Report Generator
"""

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd


class ReportGenerator:

    def generate_report(
            self,
            csv_file,
            output_file):

        df = pd.read_csv(csv_file)

        total_energy = round(
            df["Energy"].sum(),
            2
        )

        total_cost = round(
            df["Cost"].sum(),
            2
        )

        avg_power = round(
            df["Power"].mean(),
            2
        )

        doc = SimpleDocTemplate(output_file)

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "Smart Home Energy Report",
                styles["Title"]
            )
        )

        content.append(Spacer(1, 20))

        content.append(
            Paragraph(
                f"Total Energy: {total_energy} kWh",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Total Cost: ₹{total_cost}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Average Power: {avg_power} W",
                styles["Normal"]
            )
        )

        doc.build(content)