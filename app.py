"""
Startup Mortality Decision Intelligence Platform
=========================================
A Streamlit-based executive dashboard for analyzing Startup Mortality Risk. 
This platform provides descriptive, systemic, predictive, and prescriptive insights.
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
from utils.interpreter_engine import InterpreterEngine

st.set_page_config(page_title="Startup Intelligence Platform", layout="wide")

# Theme / CSS Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stTitle { color: #60a5fa; font-family: 'Inter', sans-serif; }
    .interpretation-box { background-color: #374151; padding: 15px; border-left: 5px solid #60a5fa; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Startup Mortality Decision Intelligence Platform")
st.sidebar.title("Dashboard Control")

# --- Configuration & State Initialization ---
# Focus exclusively on Startup Mortality as per user request
story_id = "StartupMortality"
story_label = "Startup Mortality Engine"
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

# --- Consolidated Unified Surface ---
st.header(f"🚀 Integrated Analytical Surface: {story_label}")
st.info("This unified view combines Executive Overview, Fundamental Statistical Methods, and Predictive Optimization.")

# 1. Executive Descriptive Profile & Statistical Properties
st.markdown("## 📊 Phase 1: Descriptive Profile & Summary Statistics")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Correlation Matrix (Systemic Links)")
    img_path = os.path.join(report_dir, "correlation_heatmap.png")
    if os.path.exists(img_path): st.image(img_path)
        
with col2:
    st.subheader("Statistical Properties (Moments)")
    stats = pd.read_csv(os.path.join(report_dir, "summary_statistics.csv"), index_col=0)
    st.dataframe(stats[['mean', 'std', 'skewness', 'kurtosis']])

st.markdown("---")

# 2. Systemic Risk & Statistical Inference
st.markdown("## ⚠️ Phase 2: Systemic Risk & Inferential Testing")
st.subheader("Structural Anomaly Detection")
anomalies = load_json("risk_anomalies.json")

m1, m2, m3 = st.columns(3)
for i, (col, data) in enumerate(list(anomalies.items())[:3]):
    if i == 0: m1.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")
    elif i == 1: m2.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")
    else: m3.metric(f"Anomalies in {col}", f"{data['count']}", f"{data['percentage']:.2f}%")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.subheader("Structural Stability Test (Inference)")
    hypo = load_json("hypothesis_test.json")
    if hypo:
        show_interpretation(interpreter.interpret_hypothesis_test(hypo))
        st.json(hypo)

with col_r2:
    st.subheader("Extreme Value Risk Distributions")
    # Take the first risk distribution plot
    risk_plots = [f for f in os.listdir(report_dir) if f.startswith("risk_dist_")]
    if risk_plots: st.image(os.path.join(report_dir, risk_plots[0]))

st.markdown("---")

# 3. Advanced Methodology: Core Data Science Techniques
st.markdown("## 🎓 Phase 3: Advanced Analytical Methodology")
exp_dir = f"reports/{story_id}/experiments"

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.subheader("Outlier & Anomaly Profiling (Box Plot)")
    st.image(f"{exp_dir}/exp1_boxplot.png")
    
    st.subheader("Probabilistic Density Analysis (Log-Scale)")
    st.image(f"{exp_dir}/exp5_distribution.png")
    dist_meta = load_json("experiments/exp5_distribution.json")
    if dist_meta:
        st.write(f"**Log-Mean:** {dist_meta['mean']:.2f} | **Log-Std Dev:** {dist_meta['std_dev']:.2f}")
        st.write(f"**Normal Distribution Test:** {'✅ Pass' if dist_meta['is_normal'] else '❌ Fail'}")

with col_e2:
    st.subheader("Segmented Clustering (K-Means)")
    st.image(f"{exp_dir}/exp4_clustering.png")
    
    st.subheader("Data Integrity & Sampling Study")
    sampling = load_json("experiments/exp3_sampling.json")
    if sampling:
        st.table(pd.DataFrame(sampling).T)

col_e3, col_e4 = st.columns(2)
with col_e3:
    st.subheader("Performance Regression Modeling (Log-Scale)")
    st.image(f"reports/{story_id}/regression_actual_vs_pred.png")

with col_e4:
    st.subheader("Temporal Trend Forecasting (ARIMA)")
    st.image(f"reports/{story_id}/forecast_trend.png")

st.markdown("---")

# 4. Predictive Intelligence & Optimization
st.markdown("## 🔮 Phase 4: Executive Predictive & Prescriptive Strategy")
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Decision Matrix & Optimization")
    strategy = load_json("prescriptive_strategy.json")
    st.write(f"Solver Status: {strategy.get('status', 'Unknown')}")
    clean_strategy = {k: v for k, v in strategy.items() if k != 'status'}
    st.bar_chart(pd.Series(clean_strategy))
    st.table(pd.DataFrame([clean_strategy]))

with col_p2:
    st.subheader("💡 Strategic Executive Decision Support")
    show_interpretation(interpreter.interpret_prescriptive_strategy(strategy))

st.sidebar.markdown("---")
st.sidebar.write("DAL Mini Project")
st.sidebar.write("Roll No.: 16014223034")
