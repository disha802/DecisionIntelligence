import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

class TimeSeriesEngine:
    """
    Analytical engine for time-series forecasting using ARIMA models.
    Supports trend analysis and future state prediction.
    """
    def __init__(self, df, story_name, date_col):
        self.df = df
        self.story_name = story_name
        self.date_col = date_col
        # Fix: Handle mixed formats and potentially invalid dates
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        # Remove null dates for time series
        self.df = self.df.dropna(subset=[date_col])

    def run_arima_forecast(self, target_col, periods=30, order=(5,1,0)):
        print(f"Running ARIMA forecast for {target_col} ({self.story_name})...")
        
        # Resample to daily frequency and handle missing dates
        ts = self.df.set_index(self.date_col)[target_col].resample('D').mean()
        ts = ts.ffill().dropna() # Forward fill gaps and drop leading NaNs
        
        if len(ts) < 10:
             print("Warning: Not enough data points for ARIMA. Returning dummy forecast.")
             dummy_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
             dummy_vals = [ts.iloc[-1]] * periods
             forecast_df = pd.DataFrame({'mean': dummy_vals}, index=dummy_dates)
             forecast_df['mean_ci_lower'] = dummy_vals
             forecast_df['mean_ci_upper'] = dummy_vals
             return ts, forecast_df

        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.get_forecast(steps=periods)
        forecast_df = forecast.summary_frame()
        
        return ts, forecast_df

    def plot_forecast(self, ts, forecast_df, target_col, output_path):
        plt.figure(figsize=(12, 6))
        plt.plot(ts, label='Historical')
        plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='red')
        plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3, label='95% Confidence Interval')
        plt.title(f"ARIMA Forecast: {target_col} ({self.story_name})")
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Forecast plot saved to {output_path}")

if __name__ == "__main__":
    pass
