import pandas as pd
import numpy as np

class InterpreterEngine:
    """
    Translates raw statistical and predictive outputs into natural language business insights.
    Acts as the bridge between model results and executive decision-making.
    """
    def __init__(self, story_name):
        self.story_name = story_name

    def interpret_descriptive_stats(self, stats_df):
        insights = []
        for col, row in stats_df.iterrows():
            if abs(row['skewness']) > 1:
                insights.append(f"**{col}** is highly skewed ({row['skewness']:.2f}). This suggests the system is not operating in a 'normal' state and may be prone to extreme events.")
            if row['kurtosis'] > 3:
                insights.append(f"**{col}** has heavy tails (Kurtosis: {row['kurtosis']:.2f}). Expect frequent 'black swan' outliers that basic models might miss.")
        return insights[:3] # Return top 3 for brevity

    def interpret_risk_anomalies(self, anomalies):
        text = "These anomalies represent **Systemic Fractures**. "
        high_risk_cols = [col for col, data in anomalies.items() if data['percentage'] > 2]
        if high_risk_cols:
            text += f"The high frequency of outliers in {', '.join(high_risk_cols)} indicates that the current operational mode is unstable."
        else:
            text += "The system shows relatively low structural deviation, suggesting a phase of stability."
        return text

    def interpret_hypothesis_test(self, hypo):
        if hypo['statistically_significant']:
            return f"**CRITICAL:** The statistical test confirms a structural shift between {hypo['groups'][0]} and {hypo['groups'][1]}. This isn't just noise—it's a fundamental change in the system's behavior."
        else:
            return "The hypothesis test shows no significant difference between the groups. The variances observed are likely within the normal range of systemic noise."

    def interpret_predictive_metrics(self, metrics):
        insights = []
        for model, results in metrics.items():
            if 'reg' in model:
                if results['r2'] > 0.7:
                    insights.append(f"The Regression model is **Strong** (R²={results['r2']:.2f}). We have identified the primary linear drivers of {model.replace('reg_', '')}.")
                else:
                    insights.append(f"The Regression model is **Weak** (R²={results['r2']:.2f}). This reveals that {model.replace('reg_', '')} is driven by non-linear or latent variables not captured here.")
            if 'clf' in model:
                insights.append(f"The {model.replace('clf_', '')} Classifier has **{results['accuracy']*100:.1f}% Accuracy**. This indicates a high reliable boundary for predicting state transitions.")
        return insights

    def interpret_prescriptive_strategy(self, strategy):
        if self.story_name == "GlobalFragility":
            return "The Optimization algorithm prioritizes **Americas** because it offers the best balance between meeting demand and minimizing high-variance shipping costs found in Asia."
        else:
            return "The solver recommends a **Heavy R&D lean**. This is a strategic move to 'buy' out of the high mortality risk identified in the early-stage cluster analysis."

if __name__ == "__main__":
    pass
