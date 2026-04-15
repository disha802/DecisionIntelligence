import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class PredictiveEngine:
    """
    Core machine learning engine for training and evaluating 
    regression and classification models.
    """
    def __init__(self, df, story_name):
        self.df = df
        self.story_name = story_name
        self.models = {}
        self.results = {}

    def run_regression(self, features, target, model_type='linear'):
        print(f"Running {model_type} regression for {target}...")
        X = self.df[features]
        y = self.df[target]
        
        # One-hot encoding for categorical features if any
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        self.models[f'reg_{target}'] = model
        self.results[f'reg_{target}'] = {
            'rmse': rmse,
            'r2': r2,
            'features': list(X.columns)
        }
        
        print(f"Regression Completed: R²={r2:.4f}, RMSE={rmse:.4f}")
        return y_test, y_pred

    def run_classification(self, features, target, model_type='logistic'):
        print(f"Running {model_type} classification for {target}...")
        X = self.df[features]
        y = self.df[target]
        
        # Handle categorical target (label encoding)
        y = pd.factorize(y)[0]
        
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
        else:
            model = DecisionTreeClassifier()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        self.models[f'clf_{target}'] = model
        self.results[f'clf_{target}'] = {
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'features': list(X.columns)
        }
        
        print(f"Classification Completed: Accuracy={acc:.4f}")
        return y_test, y_pred

    def plot_regression_results(self, y_test, y_pred, target, output_path):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f"Regression: Actual vs Predicted ({target})")
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred, target, output_path):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {target} ({self.story_name})")
        plt.savefig(output_path)
        plt.close()

    def save_model(self, model_name, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.story_name}_{model_name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    pass
