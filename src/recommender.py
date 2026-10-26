"""
Collaborative Filtering Recommender Engine:
- Top-K neighbor selection among training users who rated the target item.
- Rating prediction via Eq. (21) weighted average or mean-centered Resnick formula.
"""
import numpy as np
from typing import Tuple, Optional, Literal


def predict_ratings(
    sim_matrix: np.ndarray,
    R_train: np.ndarray,
    R_test_train: np.ndarray,
    R_test_eval: np.ndarray,
    K: int = 20,
    method: Literal["weighted_average", "resnick"] = "weighted_average",
    r_min: float = 1.0,
    r_max: float = 5.0,
    r_med: float = 3.0,
    clip_ratings: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates rating predictions for all holdout test user-item pairs (R_test_eval > 0).
    
    Args:
        sim_matrix: (N_test, N_train) pairwise similarity matrix between test users and train users.
        R_train: (N_train, M) rating matrix of training users.
        R_test_train: (N_test, M) probe rating matrix of test users.
        R_test_eval: (N_test, M) holdout target rating matrix to be predicted.
        K: Number of nearest neighbors.
        method: 'weighted_average' (Eq. 21) or 'resnick' (mean-centered).
        r_min: Minimum rating value on scale.
        r_max: Maximum rating value on scale.
        r_med: Median rating value on scale.
        clip_ratings: Whether to clip predicted ratings to [r_min, r_max].
        
    Returns:
        test_u_indices: 1D array of test user indices for each evaluated pair.
        test_i_indices: 1D array of item indices for each evaluated pair.
        actual_ratings: 1D array of true ratings.
        predicted_ratings: 1D array of predicted ratings.
    """
    N_test, M = R_test_eval.shape
    N_train = R_train.shape[0]
    
    # Precompute user means over probe / train ratings
    mask_train = R_train > 0
    train_counts = np.sum(mask_train, axis=1)
    train_user_means = np.divide(
        np.sum(R_train, axis=1), 
        np.maximum(train_counts, 1)
    )
    train_user_means[train_counts == 0] = r_med
    
    mask_test_train = R_test_train > 0
    test_counts = np.sum(mask_test_train, axis=1)
    test_user_means = np.divide(
        np.sum(R_test_train, axis=1), 
        np.maximum(test_counts, 1)
    )
    test_user_means[test_counts == 0] = r_med
    
    # Locate all target (user, item) pairs to evaluate
    test_u_indices, test_i_indices = np.where(R_test_eval > 0)
    actual_ratings = R_test_eval[test_u_indices, test_i_indices]
    N_eval = len(test_u_indices)
    
    predicted_ratings = np.zeros(N_eval, dtype=float)
    
    for idx in range(N_eval):
        u = test_u_indices[idx]
        p = test_i_indices[idx]
        
        # Candidate neighbors: training users who rated item p
        candidate_users = np.where(R_train[:, p] > 0)[0]
        
        if len(candidate_users) == 0:
            predicted_ratings[idx] = test_user_means[u]
            continue
            
        candidate_sims = sim_matrix[u, candidate_users]
        
        # Select Top-K neighbors by descending similarity
        if len(candidate_users) > K:
            top_k_idx = np.argpartition(candidate_sims, -K)[-K:]
            candidate_users = candidate_users[top_k_idx]
            candidate_sims = candidate_sims[top_k_idx]
            
        abs_weights = np.abs(candidate_sims)
        sum_weights = np.sum(abs_weights)
        
        if sum_weights < 1e-9:
            predicted_ratings[idx] = test_user_means[u]
            continue
            
        if method == "weighted_average":
            # Eq. (21): sum(sim * r) / sum(|sim|)
            ratings_p = R_train[candidate_users, p]
            pred = np.sum(candidate_sims * ratings_p) / sum_weights
        elif method == "resnick":
            # Resnick: mean_u + sum(sim * (r - mean_cand)) / sum(|sim|)
            ratings_p = R_train[candidate_users, p]
            dev_ratings = ratings_p - train_user_means[candidate_users]
            pred = test_user_means[u] + (np.sum(candidate_sims * dev_ratings) / sum_weights)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
            
        if clip_ratings:
            pred = np.clip(pred, r_min, r_max)
            
        predicted_ratings[idx] = pred
        
    return test_u_indices, test_i_indices, actual_ratings, predicted_ratings

