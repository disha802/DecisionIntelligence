import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class StatsEngine:
    """
    Performs high-level descriptive statistical analysis and visualization.
    Focuses on understanding the baseline state of the data.
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
        self.stats_summary = None

    def generate_summary_statistics(self):
        print(f"Generating summary statistics for {self.story_name}...")
        self.stats_summary = self.df.describe(include='all').T
        # Add skewness and kurtosis
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.stats_summary['skewness'] = numeric_df.skew()
        self.stats_summary['kurtosis'] = numeric_df.kurtosis()
        return self.stats_summary

    def plot_correlation_heatmap(self, output_path):
        print("Plotting correlation heatmap...")
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Heatmap - {self.story_name}")
        plt.savefig(output_path)
        plt.close()
        print(f"Heatmap saved to {output_path}")

    def plot_boxplots(self, target_cols, output_path):
        """Exp 1: Implementing Box Plot for outlier visualization"""
        print(f"Plotting boxplots for {target_cols}...")
        plt.figure(figsize=(12, 6))
        # Filter for rows that have at least one of the target_cols
        cols_available = [c for c in target_cols if c in self.df.columns]
        if not cols_available:
            return
        sns.boxplot(data=self.df[cols_available])
        plt.title(f"Box Plot: Outlier Analysis ({self.story_name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Boxplots saved to {output_path}")

    def perform_anova(self, category_col, value_col):
        """Phase 3: ANOVA for segment comparison"""
        print(f"Performing ANOVA for {value_col} across {category_col}...")
        categories = self.df[category_col].unique()
        groups = [self.df[self.df[category_col] == cat][value_col] for cat in categories if len(self.df[self.df[category_col] == cat]) > 5]
        f_val, p_val = stats.f_oneway(*groups)
        print(f"ANOVA Results: F={f_val:.4f}, p={p_val:.4f}")
        return f_val, p_val

if __name__ == "__main__":
    pass
