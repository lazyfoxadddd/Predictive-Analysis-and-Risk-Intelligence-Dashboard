"""
Data preprocessing utilities for predictive maintenance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import re


class DataPreprocessor:
    """Handles data preprocessing for predictive maintenance"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.target_column = "Target"
        self.label_encoders = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} records from {filepath}")
            print(f"📋 Columns found: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to remove spaces and special characters"""
        df = df.copy()

        # Column name mapping
        column_mapping = {
            "Air temperature [K]": "AirTemp",
            "Process temperature [K]": "ProcessTemp",
            "Rotational speed [rpm]": "Speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "ToolWear",
            "UDI": "UDI",
            "Product ID": "ProductID",
            "Type": "Type",
            "Target": "Target",
            "Failure Type": "FailureType",
        }

        # Rename columns
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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        df = df.copy()

        # Standardize column names first
        df = self._standardize_column_names(df)

        print(f"📋 Standardized columns: {df.columns.tolist()}")

        # Check for required columns
        required_cols = ["AirTemp", "ProcessTemp", "Speed", "Torque", "ToolWear"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Drop unnecessary columns
        cols_to_drop = ["UDI", "ProductID"]
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Store failure type for later use
        if "FailureType" in df.columns:
            df["Failure_Type"] = df["FailureType"]
            df = df.drop(columns=["FailureType"])

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            if col not in ["Failure_Type"]:  # Keep Failure_Type for later
                # Convert to category and then to numeric codes
                df[col] = df[col].astype("category")
                self.label_encoders[col] = LabelEncoder()
                df[col + "_encoded"] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
                df = df.drop(columns=[col])

        # Handle missing values for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Remove duplicates
        df = df.drop_duplicates()

        print(f"✅ Cleaned data: {len(df)} records")
        print(f"   Final columns: {df.columns.tolist()}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        df = df.copy()

        # Check if Target exists, if not create from Failure_Type
        if "Target" not in df.columns and "Failure_Type" in df.columns:
            df["Target"] = df["Failure_Type"].apply(
                lambda x: 1 if x != "No Failure" else 0
            )

        # Select features (numeric columns only)
        feature_cols = ["AirTemp", "ProcessTemp", "Speed", "Torque", "ToolWear"]

        # Add encoded categorical columns if they exist
        for col in df.columns:
            if col.endswith("_encoded"):
                feature_cols.append(col)

        # Ensure all feature columns exist
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols].copy()
        y = df["Target"].copy() if "Target" in df.columns else None

        self.feature_columns = available_cols
        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Split data into train, validation, and test sets"""

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_train_val,
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
    ) -> Tuple:
        """Scale features using RobustScaler"""

        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )

        result = [X_train_scaled]

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(
                X_val_scaled, columns=X_val.columns, index=X_val.index
            )
            result.append(X_val_scaled)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled, columns=X_test.columns, index=X_test.index
            )
            result.append(X_test_scaled)

        return tuple(result)

    def get_feature_columns(self) -> list:
        """Get feature column names"""
        return self.feature_columns
