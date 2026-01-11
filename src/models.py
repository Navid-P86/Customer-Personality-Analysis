import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import os

# Sklearn Core
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score

# Algorithms
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Internal Imports
from .config import RANDOM_STATE, CV_FOLDS

def get_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    """Creates a standardized preprocessor for all ML models."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

# --- REGRESSION TOURNAMENT ---
def get_regression_models(num_feat: List[str], cat_feat: List[str]) -> Dict[str, Pipeline]:
    pre = get_preprocessor(num_feat, cat_feat)
    return {
        "Linear": Pipeline([('pre', pre), ('reg', LinearRegression())]),
        "Ridge": Pipeline([('pre', pre), ('reg', Ridge(random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([('pre', pre), ('reg', RandomForestRegressor(random_state=RANDOM_STATE))])
    }

# --- CLASSIFICATION TOURNAMENT ---
def get_classification_models(num_feat: List[str], cat_feat: List[str]) -> Dict[str, Pipeline]:
    pre = get_preprocessor(num_feat, cat_feat)
    return {
        "Logistic": Pipeline([('pre', pre), ('clf', LogisticRegression(class_weight='balanced'))]),
        "Random Forest": Pipeline([('pre', pre), ('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([('pre', pre), ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))])
    }

# --- EVALUATION UTILITY ---
def run_model_tournament(models: Dict[str, Pipeline], X, y, scoring: str) -> pd.DataFrame:
    """Trains multiple models and compares them."""
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring=scoring, n_jobs=-1)
        results.append({"Model": name, "Mean Score": scores.mean(), "Std Dev": scores.std()})
    return pd.DataFrame(results).sort_values(by="Mean Score", ascending=False)