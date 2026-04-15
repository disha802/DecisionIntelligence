# 🛡️ Startup Mortality Decision Intelligence Platform

A sophisticated end-to-end analytical platform designed to provide systemic insights into startup survival and funding dynamics. This project provides a comprehensive suite of tools for data preprocessing, descriptive statistics, anomaly detection, predictive modeling, and prescriptive optimization, all unified under a professional Streamlit dashboard.

## 🚀 Key Features

### 📊 Descriptive Profile & Moments
- **Systemic Linkage Analysis**: Correlation heatmaps identifying hidden relationships between variables like funding rounds and total capital.
- **Statistical Health Checks**: Detailed summary statistics including skewness and kurtosis, with support for log-transformed normalization.

### ⚠️ Systemic Risk Monitor
- **Structural Anomaly Detection**: Z-Score based identification of outliers in critical risk factors (Funding intensity).
- **Hypothesis Testing (Inference)**: Statistical validation (T-Tests) of structural shifts between operating and closed startups.
- **Risk Distribution Visuals**: Visualization of extreme value risks in probabilistic density functions.

### 🔮 Predictive Intelligence
- **Forecasting (ARIMA)**: Time-series trend analysis to anticipate future funding movements with 95% confidence intervals.
- **Log-Scale Regression**: High-fidelity R²-driven capital prediction models using log-transformed features.
- **Balanced Classification Engine**: Risk-focused classification handling class imbalance to accurately identify failure probabilities.

### 🎯 Prescriptive Strategy
- **Budget Allocation (Optimization)**: Strategic ROI-driven decision support for startup resource management using Linear Programming.

---

## 🛠️ Installation & Setup

### 1. Environment Setup
Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Run the Processing Pipeline
The pipeline handles data cleaning (Regex-based), feature engineering (Log-transformations), and model training.

```bash
python main.py
```

### 3. Launch the Dashboard
Experience the insights through the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard application.
- `main.py`: Full analytical pipeline orchestrator.
- `utils/`: Core analytical engines (Stats, Risk, Predictive, TimeSeries, Prescriptive, Sampling, Clustering, Distribution).
- `reports/`: Generated analytical reports and visualizations.
- `datasets/`: Data source directory.

---
**Project Status**: v1.1.0 | **Presentation Edition**
