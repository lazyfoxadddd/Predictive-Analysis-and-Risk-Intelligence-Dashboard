# 🏭 Predictive Maintenance & Risk Intelligence System

## Banaras Locomotive Works - Internship Project

### 🌐 Live Demo

**Check out the live application:** [Predictive Maintenance Dashboard](https://predictive-analysis-and-risk.onrender.com/)

---

### 📋 Overview

This project implements a modern predictive maintenance system using ensemble machine learning and anomaly detection techniques to predict machine failures and recommend proactive maintenance actions. It is designed to help industrial operations reduce downtime, optimize maintenance schedules, and improve overall equipment effectiveness (OEE).

---

### 🧠 Key Features

- **Random Forest Classifier** for accurate failure probability prediction
- **Isolation Forest** for unsupervised anomaly detection
- **Hybrid Risk Engine** combining ML predictions with domain-specific logic
- **Feature Importance Analysis** to identify key failure drivers
- **Interactive Dashboard** with real-time insights and visualizations
- **Auto-generated Executive Reports** in PDF and CSV formats

---

### 🚀 Quick Start

#### 1. Clone the Repository

```bash
git clone https://github.com/lazyfoxadddd/Predictive-Analysis-and-Risk-Intelligence-Dashboard.git
cd Predictive-Analysis-and-Risk-Intelligence-Dashboard
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Train the Models

```bash
python -m models.train_model
```

This will:

- Load and preprocess the dataset
- Engineer new features (TempDiff, StressIndex, WearRate, etc.)
- Train Random Forest and Isolation Forest models
- Save the trained models to `models/saved/`

#### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### 5. Upload Your Data

- Use the sample data provided, or
- Upload your own CSV file through the dashboard sidebar
- The system will automatically process and analyze your data

---

### 📊 Dashboard Features

| Tab                   | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| **Dashboard**         | Risk distribution, top risky machines, failure probability distribution, and feature correlations |
| **Machine Details**   | In-depth view of individual machine metrics, risk factors, and actionable insights                |
| **Feature Analysis**  | Feature importance charts and correlation matrix to understand what drives failures               |
| **Anomaly Detection** | Identify and inspect machines with abnormal behavior patterns                                     |
| **Risk Report**       | Executive summary, risk distribution charts, and downloadable PDF/CSV reports                     |

---

### 🏗️ Project Structure

```
predictive-maintenance-project/
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── data/
│   └── predictive_maintenance.csv  # Sample dataset (10,000 records)
├── models/
│   ├── __init__.py
│   ├── train_model.py              # Model training pipeline
│   ├── predict.py                  # Prediction functions
│   └── saved/                      # Trained model files
│       ├── random_forest.pkl
│       ├── isolation_forest.pkl
│       ├── scaler.pkl
│       └── feature_names.pkl
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning and preparation
│   ├── feature_engineering.py      # Feature creation (TempDiff, StressIndex, etc.)
│   ├── risk_engine.py              # Hybrid risk scoring engine
│   └── report_generator.py         # PDF report generation
└── reports/
    └── generated/                  # Generated reports (auto-created)
```

---

### 🔧 Tech Stack

| Component            | Technology                                     |
| -------------------- | ---------------------------------------------- |
| **Machine Learning** | scikit-learn (Random Forest, Isolation Forest) |
| **Dashboard**        | Streamlit                                      |
| **Data Processing**  | pandas, numpy                                  |
| **Visualization**    | Plotly, matplotlib, seaborn                    |
| **Reporting**        | reportlab (PDF generation)                     |
| **Deployment**       | Render.com                                     |
| **Version Control**  | Git & GitHub                                   |

---

### 📈 Model Performance

| Metric                  | Random Forest |
| ----------------------- | ------------- |
| **Accuracy**            | ~98.7%        |
| **ROC-AUC**             | ~0.98         |
| **Precision (Failure)** | ~0.95         |
| **Recall (Failure)**    | ~0.92         |

**Top Feature Importances:**

1. **Torque** - 35%
2. **Temperature Difference** - 25%
3. **Tool Wear** - 20%
4. **Speed** - 15%
5. **Others** - 5%

---

### 💡 How It Works

#### 1. **Data Processing**

- Cleans and standardizes input data
- Engineers 8+ new features (TempDiff, StressIndex, WearRate, Power, etc.)
- Handles categorical variables (Machine Type, Operating Zone)

#### 2. **Machine Learning**

- **Random Forest**: Predicts failure probability (0-1)
- **Isolation Forest**: Detects anomalies (abnormal machine behavior)

#### 3. **Risk Engine**

```
Risk Score = 0.35 × Failure_Probability +
             0.25 × Anomaly_Risk +
             0.15 × Stress_Normalized +
             0.15 × Torque_Risk +
             0.10 × Wear_Risk
```

#### 4. **Risk Thresholds**

- **🔴 HIGH RISK** (> 0.65): Immediate action required
- **🟡 MEDIUM RISK** (0.35 - 0.65): Schedule maintenance soon
- **🟢 LOW RISK** (0.15 - 0.35): Monitor regularly
- **✅ SAFE** (< 0.15): Normal operation

---

### 📝 Future Scope

1. **Real-time Monitoring**
   - Integrate with IoT sensors for live data streaming
   - Implement WebSocket for real-time dashboard updates
   - Set up automated alerts (email, SMS, Slack notifications)

2. **Enhanced ML Capabilities**
   - Implement deep learning models (LSTM for time-series prediction)
   - Add ensemble methods (XGBoost, LightGBM) for comparison
   - Implement online learning for continuous model improvement

3. **Data Integration**
   - Connect to enterprise databases (MySQL, PostgreSQL, MongoDB)
   - Support for multiple data sources (SCADA, ERP, MES systems)
   - Implement data versioning and lineage tracking

4. **Advanced Analytics**
   - Remaining Useful Life (RUL) prediction
   - Root cause analysis for failures
   - Predictive scheduling optimization
   - Cost-benefit analysis for maintenance decisions

5. **User Experience**
   - Multi-language support
   - Mobile-responsive design
   - Customizable dashboards per user role
   - Interactive report builder

6. **Security & Compliance**
   - Role-based access control (RBAC)
   - Audit logging for all actions
   - Data encryption at rest and in transit
   - GDPR/ISO compliance for data handling

7. **Integration Ecosystem**
   - RESTful APIs for third-party integration
   - Export to Excel, PowerBI, Tableau
   - Integration with CMMS (Computerized Maintenance Management Systems)
   - IoT platform integration (Azure IoT, AWS IoT Core)

8. **Scalability**
   - Horizontal scaling with Kubernetes
   - Microservices architecture
   - Cloud-native design (AWS, Azure, GCP)
   - Support for large-scale industrial deployments

---

### 📄 License

This project is developed as part of an internship at **Banaras Locomotive Works (BLW)**. All rights reserved.

---

### 👨‍💻 Contributors

- **LazyFoxAdddd** - Developer

---

### 📧 Contact

For questions or feedback, please reach out via GitHub Issues.

---

**⭐ If you found this project useful, please give it a star on GitHub!**
