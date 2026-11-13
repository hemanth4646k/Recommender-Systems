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


def compute_fcp(
    user_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Computes Fraction of Concordant Pairs (FCP) across all users.
    Reference: Koren (2008) / Surprise library FCP definition.
    """
    if len(actual) < 2:
        return 0.5

    unique_users = np.unique(user_indices)
    total_nc = 0
    total_nd = 0

    for u in unique_users:
        mask = (user_indices == u)
        act_u = actual[mask]
        pred_u = predicted[mask]
        n_items = len(act_u)
        if n_items < 2:
            continue

        diff_act = act_u[:, None] - act_u[None, :]
        diff_pred = pred_u[:, None] - pred_u[None, :]
        valid = diff_act > 0

        total_nc += int(np.sum((diff_pred > 0) & valid))
        total_nd += int(np.sum((diff_pred < 0) & valid))

    total = total_nc + total_nd
    return float(total_nc / total) if total > 0 else 0.5


def evaluate_predictions_v3(
    user_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    k_values: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60),
    threshold: float = 4.0
) -> Dict[str, float]:
    """
    Comprehensive benchmark evaluation computing:
    - MAE, RMSE, FCP (global)
    - Precision@K, Recall@K, F1@K, FCP@K, MAP@K, MRR@K for each K in k_values.
    """
    if len(actual) == 0:
        results = {"MAE": 0.0, "RMSE": 0.0, "FCP": 0.5}
        for k in k_values:
            results[f"Precision@{k}"] = 0.0
            results[f"Recall@{k}"] = 0.0
            results[f"F1@{k}"] = 0.0
            results[f"FCP@{k}"] = 0.5
            results[f"MAP@{k}"] = 0.0
            results[f"MRR@{k}"] = 0.0
        return results

    mae = compute_mae(actual, predicted)
    rmse = compute_rmse(actual, predicted)
    global_fcp = compute_fcp(user_indices, actual, predicted)

    unique_users = np.unique(user_indices)
    user_p = {k: [] for k in k_values}
    user_r = {k: [] for k in k_values}
    user_f1 = {k: [] for k in k_values}
    user_ap = {k: [] for k in k_values}
    user_rr = {k: [] for k in k_values}
    k_nc = {k: 0 for k in k_values}
    k_nd = {k: 0 for k in k_values}

    for u in unique_users:
        mask = (user_indices == u)
        act_u = actual[mask]
        pred_u = predicted[mask]
        n_items = len(act_u)
        if n_items == 0:
            continue

        order = np.argsort(-pred_u)
        act_sorted = act_u[order]
        pred_sorted = pred_u[order]
        is_rel = (act_sorted >= threshold).astype(float)
        n_rel = float(np.sum(is_rel))

        cum_rel = np.cumsum(is_rel)
        ranks = np.arange(1, n_items + 1, dtype=float)
        precisions = cum_rel / ranks

        for k in k_values:
            k_eval = min(k, n_items)
            hits = cum_rel[k_eval - 1] if k_eval > 0 else 0.0
            p_k = hits / float(k)
            user_p[k].append(p_k)

            # Top-K FCP pairs
            if k_eval >= 2:
                top_act = act_sorted[:k_eval]
                top_pred = pred_sorted[:k_eval]
                diff_act = top_act[:, None] - top_act[None, :]
                diff_pred = top_pred[:, None] - top_pred[None, :]
                valid = diff_act > 0
                k_nc[k] += int(np.sum((diff_pred > 0) & valid))
                k_nd[k] += int(np.sum((diff_pred < 0) & valid))

            if n_rel > 0:
                r_k = hits / n_rel
                f1_k = (2.0 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
                user_r[k].append(r_k)
                user_f1[k].append(f1_k)

                if k_eval > 0:
                    ap_k = float(np.sum(precisions[:k_eval] * is_rel[:k_eval])) / min(n_rel, float(k))
                else:
                    ap_k = 0.0
                user_ap[k].append(ap_k)

                hits_idx = np.where(is_rel[:k_eval] > 0)[0]
                rr_k = (1.0 / (hits_idx[0] + 1)) if len(hits_idx) > 0 else 0.0
                user_rr[k].append(rr_k)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "FCP": global_fcp,
    }

    for k in k_values:
        results[f"Precision@{k}"] = float(np.mean(user_p[k])) if len(user_p[k]) > 0 else 0.0
        results[f"Recall@{k}"] = float(np.mean(user_r[k])) if len(user_r[k]) > 0 else 0.0
        results[f"F1@{k}"] = float(np.mean(user_f1[k])) if len(user_f1[k]) > 0 else 0.0
        results[f"MAP@{k}"] = float(np.mean(user_ap[k])) if len(user_ap[k]) > 0 else 0.0
        results[f"MRR@{k}"] = float(np.mean(user_rr[k])) if len(user_rr[k]) > 0 else 0.0
        tot_k = k_nc[k] + k_nd[k]
        results[f"FCP@{k}"] = float(k_nc[k] / tot_k) if tot_k > 0 else 0.5

    return results

