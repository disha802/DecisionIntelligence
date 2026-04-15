import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class ClusteringEngine:
    """
    Exp 4: Perform Clustering on appropriate data set
    Implementing K-Means clustering and PCA-based visualization.
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
    
    def run_kmeans(self, features, n_clusters=3):
        print(f"Executing K-Means clustering (n_clusters={n_clusters})...")
        X = self.df[features].dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization if dims > 2
        X_pca = self.pca.fit_transform(X_scaled)
        
        results_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        results_df['Cluster'] = clusters
        
        return results_df, kmeans.inertia_

    def plot_clusters(self, results_df, output_path):
        print(f"Plotting K-Means cluster segmentation...")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', alpha=0.7)
        plt.title(f"K-Means Cluster Segmentation (PCA Reduced) - {self.story_name}")
        plt.savefig(output_path)
        plt.close()
        print(f"Cluster plot saved to {output_path}")

if __name__ == "__main__":
    pass
