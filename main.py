"""
Core Analytical Pipeline Orchestrator
=====================================
This script executes the full end-to-end analytical pipeline for Decision Intelligence.
Phases include: Data Engineering, Descriptive Analysis, Systemic Risk Detection,
Predictive Modeling (Regression, Classification, Time Series), and Prescriptive Optimization.
"""

import os
import sys
import json
import numpy as np
from utils.preprocessor import DataPreprocessor
from utils.stats_engine import StatsEngine
from utils.predictive_engine import PredictiveEngine
from utils.timeseries_engine import TimeSeriesEngine
from utils.risk_engine import RiskEngine
from utils.prescriptive_engine import PrescriptiveEngine
from utils.sampling_engine import SamplingEngine
from utils.clustering_engine import ClusteringEngine
from utils.distribution_engine import DistributionEngine

def run_project_pipeline(filepath, story_name, risk_cols, group_col, test_val_col, pred_config):
    print(f"\n{'='*50}")
    print(f"STARTING PIPELINE: {story_name}")
    print(f"{'='*50}\n")
    
    # Create directories
    os.makedirs(f"reports/{story_name}/experiments", exist_ok=True)
    os.makedirs(f"data", exist_ok=True)
    os.makedirs(f"models", exist_ok=True)
    
    # Phase 1: Data Engineering
    preprocessor = DataPreprocessor(filepath, story_name)
    if not preprocessor.load_data():
        return
    
    preprocessor.handle_missing_values()
    preprocessor.detect_outliers_zscore()
    
    # Save clean version
    clean_path = preprocessor.save_clean_data()
    
    # Phase 2: Descriptive Analysis
    stats_engine = StatsEngine(preprocessor.df, story_name)
    summary = stats_engine.generate_summary_statistics()
    summary.to_csv(f"reports/{story_name}/summary_statistics.csv")
    stats_engine.plot_correlation_heatmap(f"reports/{story_name}/correlation_heatmap.png")
    
    # Phase 3: Systemic Risk Detection
    risk_engine = RiskEngine(preprocessor.df, story_name)
    z_anomalies = risk_engine.detect_zscore_anomalies(risk_cols)
    
    # Simple JSON serializer for numpy types
    def numpy_to_python(obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict): return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [deep_convert(x) for x in obj]
        else: return numpy_to_python(obj)

    # Save anomalies to json
    with open(f"reports/{story_name}/risk_anomalies.json", "w") as f:
        json.dump(deep_convert(z_anomalies), f, indent=4)
        
    # Hypothesis Test
    t_test_results = risk_engine.hypothesis_test_mean_shift(group_col, test_val_col)
    if t_test_results:
        with open(f"reports/{story_name}/hypothesis_test.json", "w") as f:
            json.dump(deep_convert(t_test_results), f, indent=4)
            
    # Risk Plots
    for col in risk_cols[:2]:
        risk_engine.plot_risk_distribution(col, f"reports/{story_name}/risk_dist_{col}.png")
    
    # Phase 4: Predictive Modeling
    pred_engine = PredictiveEngine(preprocessor.df, story_name)
    
    # Regression
    reg_features = pred_config['reg_features']
    reg_target = pred_config['reg_target']
    y_test_reg, y_pred_reg = pred_engine.run_regression(reg_features, reg_target)
    pred_engine.plot_regression_results(y_test_reg, y_pred_reg, reg_target, f"reports/{story_name}/regression_actual_vs_pred.png")
    pred_engine.save_model(f"reg_{reg_target}")
    
    # Classification
    clf_features = pred_config['clf_features']
    clf_target = pred_config['clf_target']
    y_test_clf, y_pred_clf = pred_engine.run_classification(clf_features, clf_target)
    pred_engine.plot_confusion_matrix(y_test_clf, y_pred_clf, clf_target, f"reports/{story_name}/confusion_matrix.png")
    pred_engine.save_model(f"clf_{clf_target}")
    
    with open(f"reports/{story_name}/predictive_metrics.json", "w") as f:
        json.dump(deep_convert(pred_engine.results), f, indent=4)

    # Phase 4 Forecasting (Time Series)
    ts_engine = TimeSeriesEngine(preprocessor.df, story_name, pred_config['date_col'])
    ts, forecast = ts_engine.run_arima_forecast(pred_config['ts_target'])
    ts_engine.plot_forecast(ts, forecast, pred_config['ts_target'], f"reports/{story_name}/forecast_trend.png")
    
    # Phase 5: Prescriptive Optimization
    prescriptive = PrescriptiveEngine(story_name)
    if story_name == "GlobalFragility":
        # Scenario: Redistribute stock to high-demand regions while minimizing cost
        demo_demand = {'Americas': 5000, 'Europe': 3000, 'Asia': 2000}
        demo_costs = {'Americas': 10, 'Europe': 15, 'Asia': 25}
        opt_solution = prescriptive.solve_supply_chain_optimization(demo_demand, 10000, demo_costs)
    else:
        # Scenario: Startup budget allocation
        demo_channels = ['R&D', 'Marketing', 'Operations']
        demo_roi = {'R&D': 0.8, 'Marketing': 0.5, 'Operations': 0.3}
        opt_solution = prescriptive.solve_startup_budget_optimization(1000000, demo_channels, demo_roi)
        
    with open(f"reports/{story_name}/prescriptive_strategy.json", "w") as f:
        json.dump(deep_convert(opt_solution), f, indent=4)
    
    # ---------------------------------------------------------
    # ADVANCED ANALYTICAL PIPELINE INTEGRATION
    # ---------------------------------------------------------
    print(f"\n--- EXECUTING ADVANCED ANALYTICAL SUITE ---")
    
    # Outlier Analysis (Box Plot)
    stats_engine.plot_boxplots(risk_cols, f"reports/{story_name}/experiments/exp1_boxplot.png")
    
    # Performance Modeling (Regression Analysis)
    with open(f"reports/{story_name}/experiments/exp2_regression.json", "w") as f:
        json.dump(deep_convert(pred_engine.results.get(f"reg_{reg_target}")), f, indent=4)
    
    # Sampling Methodology Comparison
    sampling_engine = SamplingEngine(preprocessor.df, story_name)
    samples = {
        'random': sampling_engine.compare_sample_means(reg_target, sampling_engine.simple_random_sample(100), "Random"),
        'stratified': sampling_engine.compare_sample_means(reg_target, sampling_engine.stratified_sample(group_col, 100), "Stratified"),
        'systematic': sampling_engine.compare_sample_means(reg_target, sampling_engine.systematic_sample(20), "Systematic")
    }
    with open(f"reports/{story_name}/experiments/exp3_sampling.json", "w") as f:
        json.dump(deep_convert(samples), f, indent=4)
        
    # Cluster Segmentation
    clustering_engine = ClusteringEngine(preprocessor.df, story_name)
    cluster_results, inertia = clustering_engine.run_kmeans(risk_cols, n_clusters=3)
    clustering_engine.plot_clusters(cluster_results, f"reports/{story_name}/experiments/exp4_clustering.png")
    with open(f"reports/{story_name}/experiments/exp4_clustering.json", "w") as f:
        json.dump({'inertia': float(inertia)}, f, indent=4)
        
    # Probability Distribution Analysis
    dist_engine = DistributionEngine(preprocessor.df, story_name)
    dist_results = dist_engine.analyze_distribution(reg_target)
    dist_engine.plot_fitted_distribution(reg_target, f"reports/{story_name}/experiments/exp5_distribution.png")
    with open(f"reports/{story_name}/experiments/exp5_distribution.json", "w") as f:
        json.dump(deep_convert(dist_results), f, indent=4)
        
    # Experiment 6: Statistical Properties (Summary Stats already generated)
    # Experiment 7: Statistical Inference (Hypothesis Test/ANOVA already done)
    # Experiment 8: Time Series Data (ARIMA already run)
    
    print(f"\nCOMPLETED FULL PIPELINE FOR {story_name}")

if __name__ == "__main__":
    # Story B: Startup Mortality
    startup_data = "datasets/big_startup_secsees_dataset.csv"
    run_project_pipeline(
        startup_data, 
        "StartupMortality",
        risk_cols=['funding_total_usd', 'funding_rounds'],
        group_col='status',
        test_val_col='funding_total_usd',
        pred_config={
            'reg_features': ['funding_rounds'],
            'reg_target': 'log_funding_total_usd',
            'clf_features': ['log_funding_total_usd', 'funding_rounds'],
            'clf_target': 'status',
            'date_col': 'founded_at',
            'ts_target': 'log_funding_total_usd'
        }
    )
