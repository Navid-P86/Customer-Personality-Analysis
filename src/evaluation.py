from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_regression(y_true, y_pred):
    """Prints standard regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    
    return {"r2": r2, "mae": mae, "rmse": rmse}

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_classification(y_true, y_pred, y_probs):
    """Prints classification metrics and ROC-AUC."""
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    
    auc = roc_auc_score(y_true, y_probs)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    return auc