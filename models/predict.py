"""
Prediction functions for trained models
"""

import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class ModelPredictor:
    """Make predictions using trained models"""

    def __init__(self, model_dir="models/saved"):
        self.model_dir = model_dir
        self.rf_model = None
        self.if_model = None
        self.feature_names = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Load trained models"""

        # Load Random Forest
        rf_path = os.path.join(self.model_dir, "random_forest.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                self.rf_model = pickle.load(f)
            print(f"✅ Loaded Random Forest from {rf_path}")
        else:
            print(f"⚠️ Random Forest model not found at {rf_path}")
            print("   Please run: python -m models.train_model first")

        # Load Isolation Forest
        if_path = os.path.join(self.model_dir, "isolation_forest.pkl")
        if os.path.exists(if_path):
            with open(if_path, "rb") as f:
                self.if_model = pickle.load(f)
            print(f"✅ Loaded Isolation Forest from {if_path}")

        # Load feature names
        feature_path = os.path.join(self.model_dir, "feature_names.pkl")
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Loaded feature names: {self.feature_names}")

        # Load scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✅ Loaded scaler")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df = df.copy()

        # Column name mapping
        column_mapping = {
            "Air temperature [K]": "AirTemp",
            "Process temperature [K]": "ProcessTemp",
            "Rotational speed [rpm]": "Speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "ToolWear",
        }

        for old, new in column_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        # Also try to find columns with similar names (case insensitive)
        for col in df.columns:
            col_lower = col.lower().strip()
            if "air temp" in col_lower or "airtemperature" in col_lower:
                if "AirTemp" not in df.columns:
                    df = df.rename(columns={col: "AirTemp"})
            elif "process temp" in col_lower or "processtemp" in col_lower:
                if "ProcessTemp" not in df.columns:
                    df = df.rename(columns={col: "ProcessTemp"})
            elif "speed" in col_lower or "rotational" in col_lower:
                if "Speed" not in df.columns:
                    df = df.rename(columns={col: "Speed"})
            elif "torque" in col_lower:
                if "Torque" not in df.columns:
                    df = df.rename(columns={col: "Torque"})
            elif "tool" in col_lower and "wear" in col_lower:
                if "ToolWear" not in df.columns:
                    df = df.rename(columns={col: "ToolWear"})

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""

        # Standardize column names first
        df = self._standardize_columns(df)

        # Check required columns
        required_cols = ["AirTemp", "ProcessTemp", "Speed", "Torque", "ToolWear"]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f"❌ Missing required columns: {missing}")
            print(f"📋 Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing}")

        # Create feature dataframe
        X = df[required_cols].copy()

        # Add engineered features
        X["TempDiff"] = X["ProcessTemp"] - X["AirTemp"]
        X["StressIndex"] = (X["Torque"] * X["Speed"]) / 1000
        X["WearRate"] = X["ToolWear"] / (X["Speed"] + 1)
        X["Power"] = (X["Speed"] * X["Torque"]) / 9550
        X["TempRatio"] = X["ProcessTemp"] / X["AirTemp"]
        X["Efficiency"] = X["Speed"] / (X["Torque"] + 1)
        X["ThermalStress"] = X["TempDiff"] * X["ToolWear"]

        # Add operating zone
        X["OpZone"] = X.apply(
            lambda row: 2 if row["Speed"] > 1600 else 1 if row["Torque"] > 45 else 0,
            axis=1,
        )

        print(f"🔢 Prepared {len(X.columns)} features")

        # Ensure all feature names match training
        if self.feature_names is not None:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0

            # Select only features used in training
            X = X[self.feature_names]
            print(f"🔢 Using {len(X.columns)} features for prediction")

        return X

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions for a dataframe"""

        if self.rf_model is None:
            raise ValueError(
                "Random Forest model not loaded. Please train the model first."
            )

        # Prepare features
        X = self._prepare_features(df)

        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X

        # Random Forest predictions
        failure_prob = self.rf_model.predict_proba(X_scaled)[:, 1]
        failure_pred = self.rf_model.predict(X_scaled)

        # Isolation Forest predictions
        anomaly_pred = None
        anomaly_score = None
        if self.if_model is not None:
            anomaly_pred = self.if_model.predict(X_scaled)
            anomaly_score = self.if_model.decision_function(X_scaled)

        # Return results
        results = {
            "failure_probability": failure_prob,
            "failure_prediction": failure_pred,
            "anomaly_prediction": anomaly_pred,
            "anomaly_score": anomaly_score,
            "features": X_scaled,
        }

        return results

    def predict_single(self, machine_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict for a single machine"""

        df = pd.DataFrame([machine_data])
        results = self.predict(df)

        return {
            "failure_probability": results["failure_probability"][0],
            "failure_prediction": results["failure_prediction"][0],
            "anomaly": results["anomaly_prediction"][0] == -1
            if results["anomaly_prediction"] is not None
            else None,
            "anomaly_score": results["anomaly_score"][0]
            if results["anomaly_score"] is not None
            else None,
        }
