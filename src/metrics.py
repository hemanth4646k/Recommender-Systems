"""
Evaluation metrics: MAE, RMSE, Recall@T, and Precision@T.
Reference: Eqs. (22)-(25) in Birim & Tumturk (2022).
"""
import numpy as np
from typing import Dict, Tuple, Optional


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (MAE — Eq. 22)."""
    if len(actual) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE — Eq. 23)."""
    if len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - predicted)**2)))


def compute_classification_metrics(
    user_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float = 4.0
) -> Tuple[float, float]:
    """
    Computes macro-averaged Recall and Precision at relevance threshold T (Eqs. 24, 25).
    
    Args:
        user_indices: 1D array of user index corresponding to each prediction.
        actual: 1D array of actual ratings.
        predicted: 1D array of predicted ratings.
        threshold: Score threshold T to define liked/recommended items (default 4.0).
        
    Returns:
        mean_recall: Average recall across users with at least one liked item.
        mean_precision: Average precision across users with at least one recommendation.
    """
    if len(actual) == 0:
        return 0.0, 0.0
        
    unique_users = np.unique(user_indices)
    user_recalls = []
    user_precisions = []
    
    for u in unique_users:
        mask = (user_indices == u)
        u_act = actual[mask]
        u_pred = predicted[mask]
        
        liked = (u_act >= threshold)
        recommended = (u_pred >= threshold)
        hit = liked & recommended
        
        n_liked = np.sum(liked)
        n_rec = np.sum(recommended)
        n_hit = np.sum(hit)
        
        if n_liked > 0:
            user_recalls.append(n_hit / n_liked)
            
        if n_rec > 0:
            user_precisions.append(n_hit / n_rec)
        else:
            user_precisions.append(0.0)
            
    mean_recall = float(np.mean(user_recalls)) if len(user_recalls) > 0 else 0.0
    mean_precision = float(np.mean(user_precisions)) if len(user_precisions) > 0 else 0.0
    
    return mean_recall, mean_precision


def evaluate_predictions(
    user_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float = 4.0
) -> Dict[str, float]:
    """
    Computes all four benchmark evaluation metrics: MAE, RMSE, Recall, Precision.
    """
    mae = compute_mae(actual, predicted)
    rmse = compute_rmse(actual, predicted)
    recall, precision = compute_classification_metrics(
        user_indices, actual, predicted, threshold=threshold
    )
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Recall": recall,
        "Precision": precision
    }

