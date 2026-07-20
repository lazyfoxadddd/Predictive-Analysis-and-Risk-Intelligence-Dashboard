"""
Simple Executive PDF Report Generator
Fast, clean, manager-friendly reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64


class ExecutiveReportGenerator:
    """Generate simple, fast PDF reports for management"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1a237e"),
                alignment=TA_CENTER,
                spaceAfter=30,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CustomHeading",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#0d47a1"),
                spaceAfter=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=self.styles["Normal"],
                fontSize=11,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricValue",
                parent=self.styles["Normal"],
                fontSize=20,
                textColor=colors.HexColor("#1a237e"),
                alignment=TA_CENTER,
                spaceAfter=2,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricLabel",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#666666"),
                alignment=TA_CENTER,
                spaceAfter=12,
            )
        )

    def generate_summary_stats(self, machines_df):
        """Generate summary statistics from machine data"""

        if len(machines_df) == 0:
            return None

        # Count risk levels
        risk_counts = machines_df["risk_level"].value_counts()

        stats = {
            "total_machines": len(machines_df),
            "high_risk": risk_counts.get("HIGH RISK", 0),
            "medium_risk": risk_counts.get("MEDIUM RISK", 0),
            "low_risk": risk_counts.get("LOW RISK", 0),
            "safe": risk_counts.get("SAFE", 0),
            "anomalies": machines_df["anomaly"].sum(),
            "avg_risk": machines_df["risk_score"].mean(),
            "max_risk": machines_df["risk_score"].max(),
            "avg_failure": machines_df["failure_probability"].mean(),
            "high_risk_machines": machines_df[machines_df["risk_level"] == "HIGH RISK"][
                "machine_id"
            ].tolist()[:5],
            "top_risk_machine": machines_df.loc[
                machines_df["risk_score"].idxmax(), "machine_id"
            ]
            if len(machines_df) > 0
            else "N/A",
            "top_risk_score": machines_df["risk_score"].max()
            if len(machines_df) > 0
            else 0,
            "generated_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        }

        return stats

    def create_pie_chart(self, data, labels, title):
        """Create a pie chart for the report"""
        drawing = Drawing(300, 200)
        pie = Pie()
        pie.x = 30
        pie.y = 30
        pie.width = 200
        pie.height = 140
        pie.data = data
        pie.labels = labels
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = colors.HexColor("#d32f2f")  # High Risk - Red
        pie.slices[1].fillColor = colors.HexColor("#f57c00")  # Medium Risk - Orange
        pie.slices[2].fillColor = colors.HexColor("#4caf50")  # Low Risk - Green
        pie.slices[3].fillColor = colors.HexColor("#1976d2")  # Safe - Blue

        drawing.add(pie)
        return drawing

    def create_risk_chart(self, risk_data):
        """Create a bar chart for risk distribution"""
        drawing = Drawing(400, 200)
        chart = VerticalBarChart()
        chart.x = 30
        chart.y = 30
        chart.width = 340
        chart.height = 150

        chart.data = [risk_data]
        chart.categoryAxis.categoryNames = ["High", "Medium", "Low", "Safe"]
        chart.categoryAxis.labels.boxAnchor = "ne"
        chart.categoryAxis.labels.dx = 8
        chart.categoryAxis.labels.dy = -2
        chart.categoryAxis.labels.angle = 0
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(risk_data) * 1.2 if max(risk_data) > 0 else 10

        chart.bars[0].fillColor = colors.HexColor("#1976d2")
        chart.bars[0].strokeColor = colors.white
        chart.bars[0].strokeWidth = 1

        drawing.add(chart)
        return drawing

    def generate_pdf(self, machines_df):
        """Generate the PDF report"""

        stats = self.generate_summary_stats(machines_df)
        if stats is None:
            return None

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Build story (content)
        story = []

        # === HEADER ===
        story.append(
            Paragraph("🏭 Predictive Maintenance Report", self.styles["CustomTitle"])
        )
        story.append(
            Paragraph(
                f"Generated: {stats['generated_date']}", self.styles["CustomBody"]
            )
        )
        story.append(Spacer(1, 20))

        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("Executive Summary", self.styles["CustomHeading"]))
        story.append(Spacer(1, 10))

        # Summary paragraph
        summary_text = f"""
        This report analyzes {stats["total_machines"]} machines. 
        Currently, <b>{stats["high_risk"]}</b> machines are at HIGH RISK requiring immediate attention,
        <b>{stats["medium_risk"]}</b> at MEDIUM RISK requiring scheduled maintenance,
        and <b>{stats["low_risk"] + stats["safe"]}</b> machines operating safely.
        <b>{stats["anomalies"]}</b> abnormal behavior patterns have been detected.
        """
        story.append(Paragraph(summary_text, self.styles["CustomBody"]))
        story.append(Spacer(1, 20))

        # === KEY METRICS ===
        story.append(Paragraph("Key Metrics", self.styles["CustomHeading"]))

        # Create metrics table
        metrics_data = [
            ["Metric", "Value", "Status"],
            ["Total Machines", str(stats["total_machines"]), "✓"],
            [
                "High Risk Machines",
                str(stats["high_risk"]),
                "🔴 CRITICAL" if stats["high_risk"] > 0 else "✅",
            ],
            [
                "Medium Risk Machines",
                str(stats["medium_risk"]),
                "🟡 WARNING" if stats["medium_risk"] > 0 else "✅",
            ],
            [
                "Anomalies Detected",
                str(stats["anomalies"]),
                "⚠️" if stats["anomalies"] > 0 else "✅",
            ],
            ["Average Risk Score", f"{stats['avg_risk']:.3f}", ""],
            ["Maximum Risk Score", f"{stats['max_risk']:.3f}", ""],
            [
                "Top Risk Machine",
                stats["top_risk_machine"],
                "🔴" if stats["top_risk_score"] > 0.65 else "🟢",
            ],
        ]

        metrics_table = Table(
            metrics_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch]
        )
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                    (
                        "BACKGROUND",
                        (0, 2),
                        (-1, 2),
                        colors.HexColor("#ffcdd2")
                        if stats["high_risk"] > 0
                        else colors.HexColor("#c8e6c9"),
                    ),
                    (
                        "BACKGROUND",
                        (0, 3),
                        (-1, 3),
                        colors.HexColor("#ffecb3")
                        if stats["medium_risk"] > 0
                        else colors.HexColor("#c8e6c9"),
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#e0e0e0")),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                ]
            )
        )

        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # === RISK DISTRIBUTION ===
        story.append(Paragraph("Risk Distribution", self.styles["CustomHeading"]))

        # Add pie chart
        risk_data = [
            stats["high_risk"],
            stats["medium_risk"],
            stats["low_risk"],
            stats["safe"],
        ]
        risk_labels = ["High Risk", "Medium Risk", "Low Risk", "Safe"]

        if sum(risk_data) > 0:
            pie_chart = self.create_pie_chart(
                risk_data, risk_labels, "Risk Distribution"
            )
            story.append(pie_chart)
        else:
            story.append(Paragraph("No risk data available", self.styles["CustomBody"]))

        story.append(Spacer(1, 20))

        # === HIGH RISK MACHINES ===
        if stats["high_risk"] > 0:
            story.append(
                Paragraph("🔴 High Risk Machines", self.styles["CustomHeading"])
            )

            high_risk_text = f"""
            <b>{stats["high_risk"]}</b> machines require immediate attention:
            """
            story.append(Paragraph(high_risk_text, self.styles["CustomBody"]))

            # List high risk machines
            for i, machine_id in enumerate(stats["high_risk_machines"], 1):
                # Get machine details
                machine_row = machines_df[machines_df["machine_id"] == machine_id]
                if len(machine_row) > 0:
                    machine = machine_row.iloc[0]
                    risk_score = machine.get("risk_score", 0)
                    failure_prob = machine.get("failure_probability", 0)

                    machine_text = f"""
                    {i}. <b>{machine_id}</b> - Risk Score: {risk_score:.3f}, 
                    Failure Probability: {failure_prob:.1%}
                    """
                    story.append(Paragraph(machine_text, self.styles["CustomBody"]))

            story.append(Spacer(1, 10))
            story.append(
                Paragraph(
                    "⚠️ Action Required: Schedule immediate inspection for all high risk machines.",
                    self.styles["CustomBody"],
                )
            )
        else:
            story.append(
                Paragraph(
                    "✅ No high risk machines detected. All machines operating within safe limits.",
                    self.styles["CustomBody"],
                )
            )

        story.append(Spacer(1, 20))

        # === RECOMMENDATIONS ===
        story.append(Paragraph("Recommendations", self.styles["CustomHeading"]))

        recommendations = []
        if stats["high_risk"] > 0:
            recommendations.append(
                "🔴 IMMEDIATE: Inspect all high risk machines within 24 hours"
            )
        if stats["medium_risk"] > 0:
            recommendations.append(
                "🟡 SCHEDULE: Plan maintenance for medium risk machines within 1 week"
            )
        if stats["anomalies"] > 0:
            recommendations.append(
                "⚠️ INVESTIGATE: Review anomalous behavior patterns detected"
            )
        if stats["avg_risk"] > 0.3:
            recommendations.append(
                "📊 MONITOR: Overall risk level is elevated - review operational parameters"
            )
        if stats["high_risk"] == 0 and stats["medium_risk"] == 0:
            recommendations.append(
                "✅ MAINTAIN: Continue regular monitoring and maintenance schedule"
            )

        if recommendations:
            for rec in recommendations:
                story.append(Paragraph(rec, self.styles["CustomBody"]))
        else:
            story.append(
                Paragraph(
                    "All systems operating normally. Continue regular maintenance schedule.",
                    self.styles["CustomBody"],
                )
            )

        story.append(Spacer(1, 30))

        # === FOOTER ===
        story.append(
            Paragraph(
                "Report generated by Predictive Maintenance System | Banaras Locomotive Works",
                self.styles["CustomBody"],
            )
        )

        # Build PDF
        doc.build(story)

        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    def get_download_link(self, pdf_data, filename=None):
        """Generate download link for PDF"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d')}.pdf"

        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download PDF Report</a>'
        return href
