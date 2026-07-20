"""
Predictive Maintenance & Risk Intelligence Dashboard
Streamlit Application for Banaras Locomotive Works
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer
from utils.risk_engine import RiskEngine
from utils.report_generator import ExecutiveReportGenerator
from models.predict import ModelPredictor

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance - BLW",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        padding: 1rem 0;
        text-align: center;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffd93d;
        color: #333;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #6bcb77;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-safe {
        background-color: #1976d2;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize components
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        predictor = ModelPredictor()
        if predictor.rf_model is None:
            st.warning("⚠️ Models not trained yet. Please train the models first.")
            st.info("Run: python -m models.train_model")
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


@st.cache_data
def load_sample_data():
    """Load sample dataset"""
    try:
        possible_paths = [
            os.path.join("data", "predictive_maintenance.csv"),
            os.path.join("..", "predictive_maintenance.csv"),
            "predictive_maintenance.csv",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Loaded sample data from: {path}")
                return df

        st.sidebar.error("❌ Sample data file not found!")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None


def preprocess_data(df):
    """Preprocess uploaded data"""
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()

    try:
        df = preprocessor.clean_data(df)
        df = engineer.add_features(df)
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None


def prepare_machine_data(df, predictor):
    """Prepare machine data with predictions and risk scores"""

    if predictor is None or predictor.rf_model is None:
        st.error("❌ Models not loaded. Please train the models first.")
        return None

    try:
        results = predictor.predict(df)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

    machines = []
    risk_engine = RiskEngine()

    for idx in range(len(df)):
        # Get machine ID
        machine_id = f"M{idx + 1}"
        if "UDI" in df.columns:
            machine_id = str(df["UDI"].iloc[idx])
        elif "Product ID" in df.columns:
            machine_id = str(df["Product ID"].iloc[idx])

        # Get feature values
        torque_val = df["Torque"].iloc[idx] if "Torque" in df.columns else 0
        toolwear_val = df["ToolWear"].iloc[idx] if "ToolWear" in df.columns else 0
        stress_idx = df["StressIndex"].iloc[idx] if "StressIndex" in df.columns else 0

        machine = {
            "machine_id": machine_id,
            "failure_probability": results["failure_probability"][idx],
            "failure_prediction": results["failure_prediction"][idx],
            "anomaly": results["anomaly_prediction"][idx] == -1
            if results["anomaly_prediction"] is not None
            else False,
            "anomaly_score": results["anomaly_score"][idx]
            if results["anomaly_score"] is not None
            else 0,
            "torque": torque_val,
            "toolwear": toolwear_val,
            "stressindex": stress_idx,
        }

        # Add temperature features
        if "AirTemp" in df.columns:
            machine["airtemp"] = df["AirTemp"].iloc[idx]
        if "ProcessTemp" in df.columns:
            machine["processtemp"] = df["ProcessTemp"].iloc[idx]
        if "TempDiff" in df.columns:
            machine["tempdiff"] = df["TempDiff"].iloc[idx]
        if "Speed" in df.columns:
            machine["speed"] = df["Speed"].iloc[idx]
        if "Power" in df.columns:
            machine["power"] = df["Power"].iloc[idx]

        # Calculate risk score using the improved engine
        risk_score = risk_engine.calculate_risk_score(
            np.array([machine["failure_probability"]]),
            np.array([machine["anomaly_score"]]),
            np.array([machine["stressindex"]]),
            np.array([machine["torque"]]),
            np.array([machine["toolwear"]]),
        )[0]

        machine["risk_score"] = risk_score
        risk_level, risk_icon = risk_engine.get_risk_level(risk_score)
        machine["risk_level"] = risk_level
        machine["risk_icon"] = risk_icon

        # Generate insights
        machine["insights"] = risk_engine.generate_insights(machine)

        machines.append(machine)

    return machines


def create_dashboard(df, machines):
    """Create the main dashboard"""

    if machines is None or len(machines) == 0:
        st.warning("⚠️ No machine data available. Please check your data.")
        return

    machines_df = pd.DataFrame(machines)

    # --- Header ---
    st.markdown(
        '<p class="main-header">🏭 Predictive Maintenance & Risk Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown("*Banaras Locomotive Works*")

    # --- Key Metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Machines", len(machines_df))
    with col2:
        high_risk = len(machines_df[machines_df["risk_level"] == "HIGH RISK"])
        st.metric("High Risk Machines", high_risk, delta_color="inverse")
    with col3:
        medium_risk = len(machines_df[machines_df["risk_level"] == "MEDIUM RISK"])
        st.metric("Medium Risk", medium_risk)
    with col4:
        anomalies = machines_df["anomaly"].sum()
        st.metric("Anomalies Detected", anomalies)
    with col5:
        avg_risk = machines_df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")

    st.divider()

    # --- Main Content ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dashboard",
            "🔍 Machine Details",
            "📈 Feature Analysis",
            "🔄 Anomaly Detection",
            "📋 Risk Report",
        ]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Risk Distribution
            risk_counts = machines_df["risk_level"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "HIGH RISK": "#ff6b6b",
                    "MEDIUM RISK": "#ffd93d",
                    "LOW RISK": "#6bcb77",
                    "SAFE": "#1976d2",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top 10 Risky Machines
            top_risky = machines_df.nlargest(10, "risk_score")[
                ["machine_id", "risk_score", "risk_level"]
            ]
            fig = px.bar(
                top_risky,
                x="machine_id",
                y="risk_score",
                color="risk_level",
                title="Top 10 Riskiest Machines",
                color_discrete_map={
                    "HIGH RISK": "#ff6b6b",
                    "MEDIUM RISK": "#ffd93d",
                    "LOW RISK": "#6bcb77",
                },
                text_auto=".2f",
            )
            fig.update_layout(xaxis_title="Machine ID", yaxis_title="Risk Score")
            st.plotly_chart(fig, use_container_width=True)

        # Failure Probability Distribution
        fig = px.histogram(
            machines_df,
            x="failure_probability",
            nbins=30,
            title="Failure Probability Distribution",
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(xaxis_title="Failure Probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # Risk vs Features scatter
        col3, col4 = st.columns(2)

        with col3:
            if "torque" in machines_df.columns:
                fig = px.scatter(
                    machines_df,
                    x="torque",
                    y="risk_score",
                    color="risk_level",
                    title="Risk Score vs Torque",
                    color_discrete_map={
                        "HIGH RISK": "#ff6b6b",
                        "MEDIUM RISK": "#ffd93d",
                        "LOW RISK": "#6bcb77",
                        "SAFE": "#1976d2",
                    },
                    hover_data=["machine_id"],
                )
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "toolwear" in machines_df.columns:
                fig = px.scatter(
                    machines_df,
                    x="toolwear",
                    y="failure_probability",
                    color="risk_level",
                    title="Failure Probability vs Tool Wear",
                    color_discrete_map={
                        "HIGH RISK": "#ff6b6b",
                        "MEDIUM RISK": "#ffd93d",
                        "LOW RISK": "#6bcb77",
                        "SAFE": "#1976d2",
                    },
                    hover_data=["machine_id"],
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Machine selector
        machine_ids = machines_df["machine_id"].tolist()
        selected_machine = st.selectbox("Select Machine", machine_ids)

        if selected_machine:
            machine_data = machines_df[
                machines_df["machine_id"] == selected_machine
            ].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### 📊 Machine: {selected_machine}")

                risk_level = machine_data["risk_level"]
                risk_icon = machine_data["risk_icon"]
                if risk_level == "HIGH RISK":
                    st.markdown(
                        f'<div class="risk-high">{risk_icon} {risk_level}</div>',
                        unsafe_allow_html=True,
                    )
                elif risk_level == "MEDIUM RISK":
                    st.markdown(
                        f'<div class="risk-medium">{risk_icon} {risk_level}</div>',
                        unsafe_allow_html=True,
                    )
                elif risk_level == "LOW RISK":
                    st.markdown(
                        f'<div class="risk-low">{risk_icon} {risk_level}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="risk-safe">{risk_icon} {risk_level}</div>',
                        unsafe_allow_html=True,
                    )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Failure Probability",
                        f"{machine_data['failure_probability']:.2%}",
                    )
                    st.metric("Risk Score", f"{machine_data['risk_score']:.3f}")
                with col_b:
                    st.metric(
                        "Anomaly", "⚠️ Yes" if machine_data["anomaly"] else "✅ No"
                    )
                    if "toolwear" in machine_data:
                        st.metric("Tool Wear", f"{machine_data['toolwear']:.0f} min")

            with col2:
                st.markdown("### 🔧 Machine Specifications")

                specs = {
                    "Air Temperature": f"{machine_data.get('airtemp', 0):.1f} K",
                    "Process Temperature": f"{machine_data.get('processtemp', 0):.1f} K",
                    "Speed": f"{machine_data.get('speed', 0):.0f} rpm",
                    "Torque": f"{machine_data.get('torque', 0):.1f} Nm",
                    "Temp Difference": f"{machine_data.get('tempdiff', 0):.1f} K",
                    "Stress Index": f"{machine_data.get('stressindex', 0):.2f}",
                }

                for key, value in specs.items():
                    st.markdown(f"**{key}:** {value}")

            st.markdown("### 💡 Insights")
            insights = machine_data.get("insights", [])
            for insight in insights:
                st.markdown(
                    f'<div class="insight-box">• {insight}</div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown("### 📈 Feature Importance Analysis")

        if hasattr(predictor, "rf_model") and predictor.rf_model is not None:
            importances = predictor.rf_model.feature_importances_
            features = predictor.feature_names

            if features is not None and len(features) == len(importances):
                importance_df = pd.DataFrame(
                    {"Feature": features, "Importance": importances}
                ).sort_values("Importance", ascending=True)

                top_n = min(20, len(importance_df))
                importance_df = importance_df.tail(top_n)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance - What Drives Failures?",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📝 Key Insights from Feature Importance")
                top_features = importance_df.tail(3)

                for _, row in top_features.iterrows():
                    feature = row["Feature"]
                    importance = row["Importance"]

                    if "torque" in feature.lower():
                        st.markdown(
                            f"🔧 **{feature}** ({importance:.1%}) - High torque is a major failure driver. Monitor torque levels closely."
                        )
                    elif "wear" in feature.lower() or "tool" in feature.lower():
                        st.markdown(
                            f"🔧 **{feature}** ({importance:.1%}) - Tool wear significantly impacts failure risk. Schedule regular tool replacements."
                        )
                    elif "temp" in feature.lower() or "thermal" in feature.lower():
                        st.markdown(
                            f"🌡️ **{feature}** ({importance:.1%}) - Temperature variations affect machine reliability. Check cooling systems."
                        )
                    elif "speed" in feature.lower():
                        st.markdown(
                            f"⚡ **{feature}** ({importance:.1%}) - Operating speed influences failure probability. Optimize speed settings."
                        )
                    else:
                        st.markdown(
                            f"📊 **{feature}** ({importance:.1%}) - Significant contributor to failure risk."
                        )

    with tab4:
        st.markdown("### 🔄 Anomaly Detection Analysis")

        col1, col2 = st.columns(2)

        with col1:
            anomaly_counts = (
                machines_df["anomaly"]
                .value_counts()
                .rename(index={True: "Anomaly", False: "Normal"})
            )
            fig = px.pie(
                values=anomaly_counts.values,
                names=anomaly_counts.index,
                title="Anomaly Detection Results",
                color_discrete_sequence=["#ff6b6b", "#1f77b4"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                machines_df,
                x="anomaly_score",
                color="anomaly",
                title="Anomaly Score Distribution",
                color_discrete_map={True: "#ff6b6b", False: "#1f77b4"},
                nbins=30,
            )
            fig.update_layout(xaxis_title="Anomaly Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🚨 Anomalous Machines")

        anomaly_machines = machines_df[machines_df["anomaly"] == True]

        if len(anomaly_machines) > 0:
            for _, machine in anomaly_machines.iterrows():
                with st.expander(
                    f"{machine['risk_icon']} Machine {machine['machine_id']} - Risk: {machine['risk_level']}"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Failure Probability",
                            f"{machine['failure_probability']:.2%}",
                        )
                        st.metric("Anomaly Score", f"{machine['anomaly_score']:.2f}")
                        st.metric("Risk Score", f"{machine['risk_score']:.3f}")

                    with col2:
                        if "toolwear" in machine:
                            st.metric("Tool Wear", f"{machine['toolwear']:.0f} min")
                        st.metric("Torque", f"{machine.get('torque', 0):.1f} Nm")
                        st.metric(
                            "Temp Difference", f"{machine.get('tempdiff', 0):.1f} K"
                        )

                    st.markdown("**Insights:**")
                    for insight in machine.get("insights", []):
                        st.markdown(f"• {insight}")
        else:
            st.success("✅ No anomalous machines detected!")

    with tab5:
        st.markdown("### 📋 Executive Risk Report")

        # Quick summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Machines", len(machines_df))
        with col2:
            high_risk = len(machines_df[machines_df["risk_level"] == "HIGH RISK"])
            st.metric("High Risk", high_risk, delta="⚠️" if high_risk > 0 else "✅")
        with col3:
            medium_risk = len(machines_df[machines_df["risk_level"] == "MEDIUM RISK"])
            st.metric("Medium Risk", medium_risk)
        with col4:
            anomalies = machines_df["anomaly"].sum()
            st.metric("Anomalies", anomalies)
        with col5:
            avg_risk = machines_df["risk_score"].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")

        st.divider()

        # Two columns for charts
        col1, col2 = st.columns(2)

        with col1:
            # Risk Distribution Pie Chart
            risk_counts = machines_df["risk_level"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "HIGH RISK": "#d32f2f",
                    "MEDIUM RISK": "#f57c00",
                    "LOW RISK": "#4caf50",
                    "SAFE": "#1976d2",
                },
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top 5 Risky Machines
            top_risky = machines_df.nlargest(5, "risk_score")[
                ["machine_id", "risk_score", "risk_level"]
            ]
            fig = px.bar(
                top_risky,
                x="machine_id",
                y="risk_score",
                color="risk_level",
                title="Top 5 Risky Machines",
                color_discrete_map={
                    "HIGH RISK": "#d32f2f",
                    "MEDIUM RISK": "#f57c00",
                    "LOW RISK": "#4caf50",
                },
            )
            fig.update_layout(
                height=300, xaxis_title="Machine", yaxis_title="Risk Score"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Simple summary table
        st.markdown("### 📊 Quick Summary")

        summary_data = {
            "Metric": [
                "Total Machines",
                "High Risk Machines",
                "Medium Risk Machines",
                "Low Risk Machines",
                "Safe Machines",
                "Anomalies Detected",
                "Average Risk Score",
                "Highest Risk Score",
                "Machine with Highest Risk",
            ],
            "Value": [
                len(machines_df),
                len(machines_df[machines_df["risk_level"] == "HIGH RISK"]),
                len(machines_df[machines_df["risk_level"] == "MEDIUM RISK"]),
                len(machines_df[machines_df["risk_level"] == "LOW RISK"]),
                len(machines_df[machines_df["risk_level"] == "SAFE"]),
                machines_df["anomaly"].sum(),
                f"{machines_df['risk_score'].mean():.3f}",
                f"{machines_df['risk_score'].max():.3f}",
                machines_df.loc[machines_df["risk_score"].idxmax(), "machine_id"]
                if len(machines_df) > 0
                else "N/A",
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.divider()

        # PDF Generation
        st.markdown("### 📥 Download Report")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF report... Please wait..."):
                    try:
                        report_gen = ExecutiveReportGenerator()
                        pdf_data = report_gen.generate_pdf(machines_df)

                        if pdf_data:
                            # Create download button
                            b64 = base64.b64encode(pdf_data).decode()
                            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                            st.markdown(
                                f"""
                            <div style="background:#e8f5e9;padding:15px;border-radius:10px;border:1px solid #4caf50;">
                                <p style="color:#2e7d32;font-weight:bold;margin:0;">✅ PDF Report Generated Successfully!</p>
                                <a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration:none;">
                                    <button style="background:#1a237e;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:16px;cursor:pointer;margin-top:10px;">
                                        📥 Download PDF Report ({filename})
                                    </button>
                                </a>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.error("❌ Failed to generate PDF report")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {e}")
                        st.info(
                            "Make sure reportlab is installed: pip install reportlab"
                        )

        with col2:
            # CSV Download
            csv_data = []
            for _, machine in machines_df.iterrows():
                csv_data.append(
                    {
                        "Machine_ID": machine["machine_id"],
                        "Risk_Score": round(machine["risk_score"], 3),
                        "Risk_Level": machine["risk_level"],
                        "Failure_Probability": round(machine["failure_probability"], 4),
                        "Anomaly": "Yes" if machine["anomaly"] else "No",
                    }
                )

            csv_df = pd.DataFrame(csv_data)

            st.download_button(
                label="📥 Download CSV Summary",
                data=csv_df.to_csv(index=False),
                file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main():
    """Main application entry point"""

    global predictor
    predictor = load_models()

    # Sidebar
    st.sidebar.title("🏭 Predictive Maintenance")
    st.sidebar.markdown("---")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source", ["Use Sample Data", "Upload CSV"]
    )

    df = None

    if data_source == "Use Sample Data":
        df = load_sample_data()
        if df is None:
            st.sidebar.error("❌ Sample data not found!")
            st.sidebar.info(
                "Please upload your CSV file using the 'Upload CSV' option."
            )
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} records")
                st.sidebar.info(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    # Check if models are loaded
    if predictor is None or predictor.rf_model is None:
        st.warning("⚠️ Models not loaded. Please train the models first.")
        st.info("Run the following command in your terminal:")
        st.code("python -m models.train_model", language="bash")

        if df is not None:
            st.info("✅ Data loaded but models are not trained yet.")
            st.info("Please train the models to see predictions.")
        return

    # Process data and make predictions
    if df is not None:
        try:
            # Preprocess
            df_processed = preprocess_data(df)

            if df_processed is None:
                st.error("❌ Data preprocessing failed. Please check your data format.")
                return

            # Prepare machine data with predictions
            machines = prepare_machine_data(df_processed, predictor)

            # Create dashboard
            create_dashboard(df_processed, machines)

        except Exception as e:
            st.error(f"❌ Error processing data: {e}")
            st.error("Please check your data format and try again.")
            st.info("Required columns: AirTemp, ProcessTemp, Speed, Torque, ToolWear")
            if df is not None:
                st.write("Available columns:", df.columns.tolist())
    else:
        st.info("👈 Please upload data or ensure sample data is available.")


if __name__ == "__main__":
    main()
