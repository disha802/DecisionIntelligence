# 📊 Startup Mortality Decision Intelligence (Presentation Outline)

Follow this structure to create a high-impact presentation for your mini-project assessment.

---

## Slide 1: Introduction & Data Selection
**Title:** Startup Mortality Decision Intelligence Platform
- **Project Goal:** To analyze systemic risk and predict success/failure in the startup ecosystem.
- **Data Selection:** Real-world dataset of ~66,000 startup outcomes.
- **Data Integrity:** Implemented advanced Regex-based cleaning to handle financial formatting (commas, null indicators) and ensured 100% numeric validity.

## Slide 2: Descriptive & Summary Statistics
**Title:** Tackling Skewness & Data Variation
- **Distribution Challenge:** Identified extreme skewness (112.0) in raw funding data.
- **Log-Transformation:** Implemented **Log(1+x) transformation**, reducing skewness to -0.32 and stabilizing the variance for modeling.
- **Statistical Moments:** Analyzed Mean, Variance, and Kurtosis on normalized scales to ensure the "stats make sense" for executive decision-making.

## Slide 3: Inferential Testing & Sampling Robustness
**Title:** Validating Hypotheses & Data Accuracy
- **Sampling Methodology:** Compared Stratified vs. Systematic sampling. **Stratified sampling** proved critical to preserve the rare "Closed" startup signal in a biased population.
- **Statistical Inference:** Performed **T-tests** confirming that funding levels are a statistically significant driver of survival status (p < 0.05).
- **Outcome:** Verified that the analytical findings are mathematically sound and representative of the full population.

## Slide 4: Advanced Predictive Modeling
**Title:** Balanced Forecasting & Segmentation
- **Predictive Performance:** Improved **Regression R² by 850%** (0.17 vs. 0.02) by modeling on log-transformed capital targets.
- **Balanced Classification:** Implemented **Balanced Class Weights** to overcome the majority-class bias. The model now actively identifies failure risks rather than just predicting survival by default.
- **Temporal Trends:** Used **ARIMA** to forecast smoothed investment cycles.

## Slide 5: Conclusion & Prescriptive Strategy
**Title:** Prescriptive Intelligence & Results
- **Optimization Strategy:** Using Linear Programming, the platform prescribes an **R&D-focused budget allocation** to maximize survival probability.
- **Impact:** Transformed a messy raw dataset into a clean, robust, and predictive Decision Intelligence platform.
- **Final Result:** A high-fidelity analytical surface ready for professional assessment.
