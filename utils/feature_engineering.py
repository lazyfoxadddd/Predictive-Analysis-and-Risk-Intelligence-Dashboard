"""
Feature engineering for predictive maintenance
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create engineered features for better predictions"""

    def __init__(self):
        self.engineered_features = []

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered features to dataset"""
        df = df.copy()

        # Check if required columns exist
        has_airtemp = "AirTemp" in df.columns
        has_processtemp = "ProcessTemp" in df.columns
        has_speed = "Speed" in df.columns
        has_torque = "Torque" in df.columns
        has_toolwear = "ToolWear" in df.columns

        # 1. Temperature difference (process - air)
        if has_processtemp and has_airtemp:
            df["TempDiff"] = df["ProcessTemp"] - df["AirTemp"]
            self.engineered_features.append("TempDiff")

        # 2. Stress Index (Torque × Speed / 1000)
        if has_torque and has_speed:
            df["StressIndex"] = (df["Torque"] * df["Speed"]) / 1000
            self.engineered_features.append("StressIndex")

        # 3. Wear Rate (ToolWear / Speed)
        if has_toolwear and has_speed:
            df["WearRate"] = df["ToolWear"] / (df["Speed"] + 1)
            self.engineered_features.append("WearRate")

        # 4. Power (Speed × Torque / 9550) - approximate
        if has_speed and has_torque:
            df["Power"] = (df["Speed"] * df["Torque"]) / 9550
            self.engineered_features.append("Power")

        # 5. Temperature ratio (Process/Air)
        if has_processtemp and has_airtemp:
            df["TempRatio"] = df["ProcessTemp"] / df["AirTemp"]
            self.engineered_features.append("TempRatio")

        # 6. Efficiency indicator (Speed / Torque)
        if has_speed and has_torque:
            df["Efficiency"] = df["Speed"] / (df["Torque"] + 1)
            self.engineered_features.append("Efficiency")

        # 7. Thermal Stress (TempDiff * ToolWear)
        if "TempDiff" in df.columns and has_toolwear:
            df["ThermalStress"] = df["TempDiff"] * df["ToolWear"]
            self.engineered_features.append("ThermalStress")

        # 8. Operating Zone (categorize speed-torque combination) - using numeric encoding
        if has_speed and has_torque:
            df["OpZone"] = df.apply(
                lambda row: (
                    2 if row["Speed"] > 1600 else 1 if row["Torque"] > 45 else 0
                ),
                axis=1,
            )
            self.engineered_features.append("OpZone")

        print(
            f"✅ Added {len(self.engineered_features)} engineered features: {self.engineered_features}"
        )
        return df

    def get_engineered_feature_names(self) -> list:
        """Get list of engineered feature names"""
        return self.engineered_features
