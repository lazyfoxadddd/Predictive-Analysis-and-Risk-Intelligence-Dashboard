"""
Risk engine combining ML predictions with domain logic
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class RiskEngine:
    """Hybrid risk scoring engine with proper calibration"""

    def __init__(self):
        self.risk_thresholds = {"high": 0.65, "medium": 0.35, "low": 0.15}

    def calculate_risk_score(
        self,
        failure_prob: np.ndarray,
        anomaly_score: np.ndarray,
        stress_index: np.ndarray,
        torque: np.ndarray = None,
        tool_wear: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate risk score using weighted combination with proper scaling
        """
        # Ensure inputs are numpy arrays
        failure_prob = np.array(failure_prob).flatten()
        anomaly_score = np.array(anomaly_score).flatten()
        stress_index = np.array(stress_index).flatten()

        # Convert anomaly score: -1 to 1 -> 0 to 1 (higher = more anomalous)
        # Isolation Forest: -1 = anomaly, 1 = normal
        anomaly_norm = (anomaly_score + 1) / 2
        anomaly_risk = 1 - anomaly_norm  # Higher risk for anomalies

        # Normalize stress index using percentiles
        if len(stress_index) > 0 and stress_index.std() > 0:
            # Use robust scaling with percentiles
            p95 = np.percentile(stress_index, 95)
            p5 = np.percentile(stress_index, 5)
            if p95 > p5:
                stress_norm = np.clip((stress_index - p5) / (p95 - p5), 0, 1)
            else:
                stress_norm = np.zeros_like(stress_index)
        else:
            stress_norm = np.zeros_like(stress_index)

        # Additional risk factors
        torque_risk = np.zeros_like(failure_prob)
        if torque is not None:
            torque = np.array(torque).flatten()
            # High torque (>45 Nm) increases risk
            torque_risk = np.clip((torque - 35) / 50, 0, 1)

        wear_risk = np.zeros_like(failure_prob)
        if tool_wear is not None:
            tool_wear = np.array(tool_wear).flatten()
            # High tool wear (>200 min) increases risk
            wear_risk = np.clip((tool_wear - 100) / 300, 0, 1)

        # Combined risk score with improved weights
        risk_score = (
            0.35 * failure_prob  # ML prediction
            + 0.25 * anomaly_risk  # Anomaly detection
            + 0.15 * stress_norm  # Mechanical stress
            + 0.15 * torque_risk  # Torque impact
            + 0.10 * wear_risk  # Tool wear impact
        )

        # Boost risk for high failure probability
        risk_score = np.where(failure_prob > 0.3, risk_score + 0.2, risk_score)
        risk_score = np.where(failure_prob > 0.5, risk_score + 0.3, risk_score)

        return np.clip(risk_score, 0, 1)

    def get_risk_level(self, risk_score: float) -> Tuple[str, str]:
        """Determine risk level and corresponding color"""
        if risk_score >= self.risk_thresholds["high"]:
            return "HIGH RISK", "🔴"
        elif risk_score >= self.risk_thresholds["medium"]:
            return "MEDIUM RISK", "🟡"
        elif risk_score >= self.risk_thresholds["low"]:
            return "LOW RISK", "🟢"
        else:
            return "SAFE", "✅"

    def generate_insights(self, machine_data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights for a machine"""
        insights = []

        risk_score = machine_data.get("risk_score", 0)
        failure_prob = machine_data.get("failure_probability", 0)
        anomaly = machine_data.get("anomaly", False)

        # Risk-based insights
        if risk_score >= 0.65:
            insights.append("🚨 CRITICAL: Immediate action required!")
        elif risk_score >= 0.35:
            insights.append("⚠️ WARNING: Schedule maintenance soon")
        else:
            insights.append("✅ Machine operating normally")

        # Feature-based insights
        torque = machine_data.get("torque", 0)
        if torque > 50:
            insights.append(f"⚙️ High torque ({torque:.1f} Nm) - Reduce load")
        elif torque > 40:
            insights.append(f"⚙️ Elevated torque ({torque:.1f} Nm) - Monitor")

        tool_wear = machine_data.get("toolwear", 0)
        if tool_wear > 250:
            insights.append(
                f"🔧 Critical tool wear ({tool_wear:.0f} min) - Replace immediately"
            )
        elif tool_wear > 200:
            insights.append(
                f"🔧 High tool wear ({tool_wear:.0f} min) - Plan replacement"
            )

        temp_diff = machine_data.get("tempdiff", 0)
        if temp_diff > 20:
            insights.append(
                f"🌡️ High temperature differential ({temp_diff:.1f} K) - Check cooling"
            )
        elif temp_diff > 15:
            insights.append(f"🌡️ Elevated temperature ({temp_diff:.1f} K) - Monitor")

        stress_idx = machine_data.get("stressindex", 0)
        if stress_idx > 80:
            insights.append(
                f"💪 High mechanical stress ({stress_idx:.0f}) - Reduce speed"
            )

        if failure_prob > 0.5:
            insights.append("📈 High failure probability - Priority inspection needed")

        if anomaly:
            insights.append("⚠️ Unusual behavior detected - Investigate immediately")

        return insights[:5]  # Limit to top 5 insights

    def generate_risk_report(
        self, all_machines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report"""

        if not all_machines:
            return {"error": "No machine data available"}

        df = pd.DataFrame(all_machines)

        # Count by risk level
        risk_levels = df["risk_level"].value_counts().to_dict()

        # Top risky machines
        top_risky = df.nlargest(10, "risk_score")[
            ["machine_id", "risk_score", "risk_level"]
        ].to_dict("records")

        # Calculate averages
        avg_risk = df["risk_score"].mean()
        avg_failure = df["failure_probability"].mean()
        anomaly_count = df["anomaly"].sum()

        return {
            "total_machines": len(all_machines),
            "high_risk_count": risk_levels.get("HIGH RISK", 0),
            "medium_risk_count": risk_levels.get("MEDIUM RISK", 0),
            "low_risk_count": risk_levels.get("LOW RISK", 0),
            "safe_count": risk_levels.get("SAFE", 0),
            "top_risky_machines": top_risky,
            "anomaly_count": anomaly_count,
            "avg_risk_score": avg_risk,
            "avg_failure_probability": avg_failure,
        }
