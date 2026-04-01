# 🛡️ Dual-Path Decision Intelligence Platform

A sophisticated end-to-end analytical platform designed to provide systemic insights into two critical domains: **Global Supply Chain Fragility** and **Startup Mortality Risk**.

This project provides a comprehensive suite of tools for data preprocessing, descriptive statistics, anomaly detection, predictive modeling, and prescriptive optimization, all unified under a professional Streamlit dashboard.

## 🚀 Key Features

### 📊 Descriptive Profile
- **Systemic Linkage Analysis**: Interactive correlation heatmaps identifying hidden relationships between variables.
- **Statistical Health Checks**: Detailed summary statistics including skewness and variance to assess data quality.

### ⚠️ Systemic Risk Monitor
- **Structural Anomaly Detection**: Z-Score based identification of outliers in critical risk factors.
- **Hypothesis Testing**: Statistical validation (T-Tests) of structural shifts across different segments.
- **Risk Distribution Visuals**: Visualization of extreme value risks in key performance areas.

### 🔮 Predictive Intelligence
- **Forecasting (ARIMA)**: Time-series trend analysis to anticipate future movements.
- **Multi-Model Regression**: Accuracy-driven profit and funding prediction models.
- **Classification Engine**: Risk-focused classification for late delivery and startup failure.

### 🎯 Prescriptive Strategy
- **Resource Optimization**: Pulse-based optimization for supply chain allocation.
- **Budget Allocation**: Strategic ROI-driven decision support for startup resource management.

---

## 🛠️ Installation & Setup

### 1. Environment Setup
Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Processing Pipeline
The pipeline handles data cleaning, feature engineering, and model training for both "Global Fragility" and "Startup Mortality" paths.

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
- `utils/`: Core analytical engines (Stats, Risk, Predictive, TimeSeries, Prescriptive).
- `reports/`: Generated analytical reports and visualizations.
- `datasets/`: Data source directory (contains CSVs after extraction).

---
**Project Status**: v1.0.0 | **Presentation Edition**
