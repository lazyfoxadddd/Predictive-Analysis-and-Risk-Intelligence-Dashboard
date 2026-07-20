"""
Model training for predictive maintenance - Windows Compatible
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV

# Add parent directory to path for Windows
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Train and save predictive maintenance models"""

    def __init__(self, model_dir="models/saved"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.rf_model = None
        self.if_model = None
        self.feature_names = None

    def load_and_prepare_data(self, filepath: str):
        """Load and prepare data for training"""
        print(f"📂 Loading data from: {filepath}")

        # Load data
        df = self.preprocessor.load_data(filepath)
        if df is None:
            return None

        print(f"📊 Data shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")

        # Clean data
        df = self.preprocessor.clean_data(df)

        # Add engineered features
        df = self.engineer.add_features(df)

        # Prepare features
        X, y = self.preprocessor.prepare_features(df)

        # Update feature names with engineered features
        self.feature_names = X.columns.tolist()

        print(f"🔢 Features: {len(self.feature_names)}")
        print(f"🔢 Feature names: {self.feature_names}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")

        # Split data
        split_data = self.preprocessor.split_data(X, y, test_size=0.2, val_size=0.1)

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.preprocessor.scale_features(
            split_data["X_train"], split_data["X_val"], split_data["X_test"]
        )

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": split_data["y_train"],
            "y_val": split_data["y_val"],
            "y_test": split_data["y_test"],
        }

    def train_random_forest(self, data: dict, use_subset: bool = True):
        """Train Random Forest model with optional subset for efficiency"""

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        # Use subset for faster training
        if use_subset and len(X_train) > 1000:
            subset_size = min(int(0.8 * len(X_train)), 3000)
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_subset = X_train.iloc[indices]
            y_train_subset = y_train.iloc[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train

        print(f"🤖 Training Random Forest on {len(X_train_subset)} samples...")

        # Simplified hyperparameter grid for speed
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [10, 15],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"],
        }

        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Grid search with cross-validation (use fewer folds for speed)
        grid_search = GridSearchCV(
            rf, param_grid, cv=2, scoring="roc_auc", n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_subset, y_train_subset)

        # Best model
        self.rf_model = grid_search.best_estimator_

        # Evaluate on validation set
        val_pred = self.rf_model.predict(X_val)
        val_proba = self.rf_model.predict_proba(X_val)[:, 1]

        print(f"✅ Random Forest trained!")
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
        print(f"   Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

        return self.rf_model

    def train_isolation_forest(self, X_train: pd.DataFrame, contamination: float = 0.1):
        """Train Isolation Forest for anomaly detection"""

        print(f"🌲 Training Isolation Forest on {len(X_train)} samples...")

        # Use subset for faster training
        if len(X_train) > 2000:
            indices = np.random.choice(len(X_train), 2000, replace=False)
            X_subset = X_train.iloc[indices]
        else:
            X_subset = X_train

        self.if_model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )

        self.if_model.fit(X_subset)

        # Get anomaly scores on training data
        train_scores = self.if_model.predict(X_train)
        anomaly_count = np.sum(train_scores == -1)

        print(f"✅ Isolation Forest trained!")
        print(
            f"   Anomalies in training: {anomaly_count} ({anomaly_count / len(X_train) * 100:.2f}%)"
        )

        return self.if_model

    def save_models(self):
        """Save trained models and feature names"""

        if self.rf_model is None:
            print("❌ No Random Forest model to save")
            return

        # Save Random Forest
        rf_path = os.path.join(self.model_dir, "random_forest.pkl")
        with open(rf_path, "wb") as f:
            pickle.dump(self.rf_model, f)
        print(f"✅ Saved Random Forest to {rf_path}")

        # Save Isolation Forest
        if self.if_model is not None:
            if_path = os.path.join(self.model_dir, "isolation_forest.pkl")
            with open(if_path, "wb") as f:
                pickle.dump(self.if_model, f)
            print(f"✅ Saved Isolation Forest to {if_path}")

        # Save feature names
        if self.feature_names is not None:
            feature_path = os.path.join(self.model_dir, "feature_names.pkl")
            with open(feature_path, "wb") as f:
                pickle.dump(self.feature_names, f)
            print(f"✅ Saved feature names to {feature_path}")

        # Save scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.preprocessor.scaler, f)
        print(f"✅ Saved scaler to {scaler_path}")

    def train_full_pipeline(self, data_path: str):
        """Run complete training pipeline"""

        # Load and prepare data
        data = self.load_and_prepare_data(data_path)
        if data is None:
            print("❌ Data preparation failed")
            return

        # Train Random Forest
        self.train_random_forest(data)

        # Train Isolation Forest (using training data only)
        self.train_isolation_forest(data["X_train"])

        # Save models
        self.save_models()

        # Return evaluation results
        results = self.evaluate_models(data)
        return results

    def evaluate_models(self, data: dict) -> dict:
        """Evaluate trained models on test data"""

        X_test = data["X_test"]
        y_test = data["y_test"]

        results = {}

        # Random Forest evaluation
        if self.rf_model is not None:
            y_pred = self.rf_model.predict(X_test)
            y_proba = self.rf_model.predict_proba(X_test)[:, 1]

            results["rf"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
            }

            print(f"\n📊 Random Forest Test Results:")
            print(f"   Accuracy: {results['rf']['accuracy']:.4f}")
            print(f"   ROC-AUC: {results['rf']['roc_auc']:.4f}")

        # Isolation Forest evaluation
        if self.if_model is not None:
            anomaly_pred = self.if_model.predict(X_test)
            results["if"] = {
                "anomalies": np.sum(anomaly_pred == -1),
                "anomaly_rate": np.mean(anomaly_pred == -1),
            }
            print(f"\n📊 Isolation Forest Test Results:")
            print(
                f"   Anomalies detected: {results['if']['anomalies']} ({results['if']['anomaly_rate'] * 100:.2f}%)"
            )

        return results


def main():
    """Main training script"""

    print("\n" + "=" * 60)
    print("🚀 PREDICTIVE MAINTENANCE MODEL TRAINING")
    print("=" * 60 + "\n")

    # Initialize trainer
    trainer = ModelTrainer()

    # Set data path
    data_path = os.path.join("data", "predictive_maintenance.csv")

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists at the specified path.")
        return

    # Train pipeline
    results = trainer.train_full_pipeline(data_path)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
