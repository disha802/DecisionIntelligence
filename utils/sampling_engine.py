import pandas as pd
import numpy as np

class SamplingEngine:
    """
    Exp 3: Study of Sampling Techniques
    Provides methods for Random, Stratified, and Systematic sampling.
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
    
    def simple_random_sample(self, n=100):
        print(f"Executing Simple Random Sampling (n={n})...")
        return self.df.sample(n=n, random_state=42)
    
    def stratified_sample(self, stratify_col, n=100):
        print(f"Executing Stratified Sampling on {stratify_col} (n={n})...")
        # Ensure we don't sample more than available in small strata
        frac = n / len(self.df)
        return self.df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42) if len(x) > 0 else x)

    def systematic_sample(self, k=10):
        print(f"Executing Systematic Sampling (k={k})...")
        indices = np.arange(0, len(self.df), step=k)
        return self.df.iloc[indices]

    def compare_sample_means(self, target_col, sample_df, method_name):
        """Comparing sample mean with population mean for verification"""
        pop_mean = self.df[target_col].mean()
        sample_mean = sample_df[target_col].mean()
        diff = abs(pop_mean - sample_mean)
        return {
            'method': method_name,
            'population_mean': float(pop_mean),
            'sample_mean': float(sample_mean),
            'delta': float(diff),
            'accurate': diff < (0.1 * pop_mean) # threshold of 10%
        }

if __name__ == "__main__":
    pass
