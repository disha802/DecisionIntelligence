import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RiskEngine:
    """
    Specialized engine for detecting systemic risks using statistical anomalies
    and structural hypothesis testing.
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
        self.anomalies = {}

    def detect_zscore_anomalies(self, target_cols, threshold=3):
        """Phase 3: Z-score anomaly detection for structural deviations"""
        print(f"Detecting Systemic Risk (Z-score) for {self.story_name}...")
        results = {}
        for col in target_cols:
            if col in self.df.columns:
                z = np.abs(stats.zscore(self.df[col].dropna()))
                anomalous_indices = np.where(z > threshold)[0]
                results[col] = {
                    'count': len(anomalous_indices),
                    'percentage': (len(anomalous_indices) / len(self.df)) * 100,
                    'indices': anomalous_indices.tolist()[:10] # Top 10 for sample
                }
        self.anomalies['zscore'] = results
        return results

    def hypothesis_test_mean_shift(self, group_col, value_col):
        """Phase 3: Hypothesis testing for mean shifts (T-Test)"""
        print(f"Testing for Mean Shifts in {value_col} across {group_col}...")
        groups = self.df[group_col].unique()
        if len(groups) < 2:
            return None
        
        # Take two largest groups for comparison
        group_counts = self.df[group_col].value_counts()
        g1_name = group_counts.index[0]
        g2_name = group_counts.index[1]
        
        g1 = self.df[self.df[group_col] == g1_name][value_col].dropna()
        g2 = self.df[self.df[group_col] == g2_name][value_col].dropna()
        
        t_stat, p_val = stats.ttest_ind(g1, g2)
        
        result = {
            'groups': (g1_name, g2_name),
            't_statistic': t_stat,
            'p_value': p_val,
            'statistically_significant': p_val < 0.05
        }
        return result

    def plot_risk_distribution(self, col, output_path):
        """Visualizing the distribution and anomalies"""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[col], kde=True, color='red')
        plt.title(f"Risk Distribution: {col} ({self.story_name})")
        plt.axvline(self.df[col].mean(), color='blue', linestyle='--', label='Mean')
        plt.axvline(self.df[col].mean() + 3*self.df[col].std(), color='green', linestyle=':', label='3-Sigma Anomaly Threshold')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    pass
