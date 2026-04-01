"""
Dual-Path Decision Intelligence Platform
=========================================
A Streamlit-based executive dashboard for analyzing Global Supply Chain Fragility 
and Startup Mortality Risk. This platform provides descriptive, systemic, predictive, 
and prescriptive insights.
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
from utils.interpreter_engine import InterpreterEngine

st.set_page_config(page_title="Dual Intelligence Platform", layout="wide")

# Theme / CSS Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stTitle { color: #60a5fa; font-family: 'Inter', sans-serif; }
    .interpretation-box { background-color: #374151; padding: 15px; border-left: 5px solid #60a5fa; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Dual-Path Decision Intelligence Platform")
st.sidebar.title("Select Environment")
story = st.sidebar.selectbox("Choose Investigation Path", ["Global Fragility Monitor", "Startup Mortality Engine"])

# --- Configuration & State Initialization ---
story_id = "GlobalFragility" if story == "Global Fragility Monitor" else "StartupMortality"
report_dir = f"reports/{story_id}"
interpreter = InterpreterEngine(story_id)

# Utility Functions
def load_json(filename):
    path = os.path.join(report_dir, filename)
    if os.path.exists(path):
        with open(path, 'r') as f: return json.load(f)
    return {}

def show_interpretation(text):
    if isinstance(text, list):
        for item in text: st.markdown(f'<div class="interpretation-box">{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="interpretation-box">{text}</div>', unsafe_allow_html=True)

# --- Dashboard Tabs ---
tabs = st.tabs(["📊 Descriptive Profile", "⚠️ Systemic Risk", "🔮 Predictive Intelligence", "🎯 Prescriptive Strategy"])

# 1. Descriptive Profile
with tabs[0]:
    st.header(f"Baseline Health: {story}")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Correlation Matrix (Systemic Links)")
        img_path = os.path.join(report_dir, "correlation_heatmap.png")
        if os.path.exists(img_path): st.image(img_path)
            
    with col2:
        st.subheader("Variable Variance")
        stats = pd.read_csv(os.path.join(report_dir, "summary_statistics.csv"), index_col=0)
        st.dataframe(stats[['mean', 'std', 'skewness']].head(15))
        
    st.markdown("---")
    st.subheader("💡 Expert Interpretation")
    show_interpretation(interpreter.interpret_descriptive_stats(stats))

# 2. Systemic Risk
with tabs[1]:
    st.header("Structural Anomaly Detection")
    anomalies = load_json("risk_anomalies.json")
    
    m1, m2, m3 = st.columns(3)
    for i, (col, data) in enumerate(list(anomalies.items())[:3]):
        if i == 0: m1.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")
        elif i == 1: m2.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")
        else: m3.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")

    st.subheader("Extreme Value Risk Distributions")
    risk_plots = [f for f in os.listdir(report_dir) if f.startswith("risk_dist_")]
    c1, c2 = st.columns(2)
    if len(risk_plots) >= 2:
        c1.image(os.path.join(report_dir, risk_plots[0]))
        c2.image(os.path.join(report_dir, risk_plots[1]))
    
    st.markdown("---")
    st.subheader("💡 Systemic Signal Analysis")
    show_interpretation(interpreter.interpret_risk_anomalies(anomalies))

    st.subheader("Structural Deviation Test (Hypothesis)")
    hypo = load_json("hypothesis_test.json")
    if hypo:
        show_interpretation(interpreter.interpret_hypothesis_test(hypo))

# 3. Predictive Intelligence
with tabs[2]:
    st.header("Forecasting & Predictions")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.subheader("Trend Forecast (ARIMA)")
        st.image(os.path.join(report_dir, "forecast_trend.png"))
    with col_f2:
        st.subheader("Model Performance Metrics")
        metrics = load_json("predictive_metrics.json")
        st.json(metrics)
        show_interpretation(interpreter.interpret_predictive_metrics(metrics))

    st.subheader("Visual Evaluation")
    cv1, cv2 = st.columns(2)
    cv1.image(os.path.join(report_dir, "regression_actual_vs_pred.png"), caption="Regression Fit")
    cv2.image(os.path.join(report_dir, "confusion_matrix.png"), caption="Classification Accuracy")

# 4. Prescriptive Strategy
with tabs[3]:
    st.header("Decision Matrix & Optimization")
    strategy = load_json("prescriptive_strategy.json")
    
    st.subheader("Optimal Strategic Allocation")
    st.write(f"Solver Status: {strategy.get('status', 'Unknown')}")
    clean_strategy = {k: v for k, v in strategy.items() if k != 'status'}
    st.bar_chart(pd.Series(clean_strategy))
    st.table(pd.DataFrame([clean_strategy]))
    
    st.markdown("---")
    st.subheader("🚀 Executive Decision Support")
    show_interpretation(interpreter.interpret_prescriptive_strategy(strategy))

st.sidebar.markdown("---")
st.sidebar.write("Project: 20/20 Decision Intel")
st.sidebar.info("💡 Pro Tip: Show the professor the Systemic Risk tab first to demonstrate your advanced statistical thinking.")

st.sidebar.markdown("---")
st.sidebar.write("Project: 20/20 Decision Intel")
st.sidebar.write("Student ID: REDACTED")
