import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    """
    Handles data ingestion, cleaning, and feature engineering.
    Ensures high data quality for downstream analytical engines.
    """
    def __init__(self, filepath, story_name):
        self.filepath = filepath
        self.story_name = story_name
        self.df = None
        self.clean_df = None

    def load_data(self):
        print(f"Loading {self.story_name} dataset...")
        try:
            # For Story A, we might need latin-1 encoding
            if "DataCo" in self.filepath:
                self.df = pd.read_csv(self.filepath, encoding='latin-1')
            else:
                self.df = pd.read_csv(self.filepath)
            print(f"Loaded {len(self.df)} rows.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def handle_missing_values(self):
        print("Handling missing values & cleaning numeric data...")
        
        # Specific cleaning for financial data (e.g., '-' in Startup dataset)
        if "startup" in self.filepath.lower():
            self.df['funding_total_usd'] = pd.to_numeric(self.df['funding_total_usd'].replace('-', '0'), errors='coerce').fillna(0)

        # Fill numeric missing with median, categorical with mode
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        for col in categorical_cols:
             if self.df[col].mode().empty:
                 self.df[col] = self.df[col].fillna("Unknown")
             else:
                 self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        print("Missing values and data types handled.")

    def detect_outliers_zscore(self, threshold=3):
        print("Detecting outliers via Z-score...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((self.df[numeric_cols] - self.df[numeric_cols].mean()) / self.df[numeric_cols].std())
        outlier_count = (z_scores > threshold).sum().sum()
        print(f"Detected {outlier_count} potential outliers.")
        # We won't remove them yet, just flag or handle in specific models

    def normalize_features(self, target_cols):
        print(f"Normalizing features: {target_cols}")
        scaler = StandardScaler()
        self.df[target_cols] = scaler.fit_transform(self.df[target_cols])

    def save_clean_data(self, output_dir="data"):
        filename = f"clean_{os.path.basename(self.filepath)}"
        output_path = os.path.join(output_dir, filename)
        self.df.to_csv(output_path, index=False)
        print(f"Saved clean data to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test with one of the datasets if needed
    pass
