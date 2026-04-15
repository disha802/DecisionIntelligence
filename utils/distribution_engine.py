import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class DistributionEngine:
    """
    Exp 5: Analyze random variables using probability distribution
    Exp 6: Compute statistical properties for probability distributions
    Fits empirical data to theoretical distributions (Normal, Poisson, etc.)
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
    
    def analyze_distribution(self, target_col):
        print(f"Analyzing probability distribution for {target_col}...")
        data = self.df[target_col].dropna()
        
        # Normality test (Shapiro-Wilk)
        if len(data) > 3:
            shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(len(data), 500), random_state=42))
        else:
            shapiro_stat, shapiro_p = 0, 0
            
        # Fitting stats
        mean = data.mean()
        std = data.std()
        variance = data.var()
        
        results = {
            'feature': target_col,
            'mean': float(mean),
            'std_dev': float(std),
            'variance': float(variance),
            'is_normal': bool(shapiro_p > 0.05),
            'shapiro_p_value': float(shapiro_p)
        }
        
        return results

    def plot_fitted_distribution(self, target_col, output_path):
        data = self.df[target_col].dropna()
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, stat="density", linewidth=0, alpha=0.5, color='purple')
        
        # Add fitted normal PDF
        mu, std = stats.norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal PDF')
        
        plt.title(f"Probability Distribution Fit: {target_col} ({self.story_name})")
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Distribution plot saved to {output_path}")

if __name__ == "__main__":
    pass
