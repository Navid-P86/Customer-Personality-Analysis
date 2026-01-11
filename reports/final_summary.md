# Customer Analytics: Executive Summary

## 1. Project Objective
The goal of this project was to analyze customer behavior from the marketing campaign dataset and build predictive models to optimize future marketing spend.

## 2. Key Customer Insights (EDA)
- **High-Value Segment:** Customers with "Graduate" or "Postgrad" education levels represent the highest total spending.
- **Demographics:** There is a clear positive correlation between **Income** and **Total Spending**.
- **Parental Status:** Customers with children tend to spend significantly less than those without children, regardless of income level.

## 3. Model Performance Comparison
We implemented three different types of machine learning to tackle business problems:

| Model Task | Algorithm | Primary Metric | Result |
| :--- | :--- | :--- | :--- |
| **Spending Prediction** | Random Forest Regressor | R² Score | 0.778 |
| **Campaign Response** | Gradient Boosting | ROC-AUC | 0.806 |
| **Deep Learning** | Neural Network (MLP) | Accuracy | 0.842 |



## 4. Customer Segmentation (Clustering)
We identified **4 distinct personas**:
1. **Cluster 2 (The VIPs):** High Income, High Spending, few children. Focus premium offers here.
2. **Cluster 0 (Stable Seniors):** Moderate income, consistent spenders, older demographic.
3. **Cluster 1 & 3 (Budget/Family):** Lower disposable income, high number of children. Focus on value-based promotions.



## 5. Deployment Recommendations
- **Targeting:** Use the `best_classifier.joblib` model to score the customer database and only send emails to those with a predicted probability > 0.5.
- **Customization:** Tailor the marketing message based on the **Cluster ID**. VIPs should receive "Luxury" messaging, while parents should receive "Family Savings" messaging.

---
*Report Generated: 2026-01-11*