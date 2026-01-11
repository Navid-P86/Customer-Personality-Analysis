# Customer Analytics & ML Project

An end-to-end data science project for analyzing and predicting customer behavior.


##  Quick Start
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Navid-P86/Customer-Personality-Analysis.git]
2. Create and activate a virtual environment.
3. Install dependencies: pip install -r requirements.txt
4. Place the marketing_campaign.csv in data/raw/.





## Project Structure
- `data/raw`: Original marketing campaign dataset.
- `src/`: Modular Python scripts for data loading, cleaning, and modeling.
- `notebooks/`: Step-by-step analysis (EDA -> Regression -> Classification -> Clustering -> Deep Learning).
- `models/`: Saved `.joblib` and `.keras` models ready for deployment.

## Key Results
- **Regression:** R² = 0.778
- **Classification:** ROC-AUC = 0.830
- **Clustering:** 4 Segments identified (Premium, Mass, Young, Frugal).
- **Deep Learning:** Accuracy = 84.2%