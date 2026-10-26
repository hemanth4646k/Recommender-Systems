"""
Data loading and train/test splitting pipeline for MovieLens 100K and FilmTrust datasets.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from .config import DATASET_CONFIGS, DatasetConfig


def load_dataset(dataset_name: str = "ml-100k") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads rating dataset and constructs a dense (N_users, N_items) rating matrix.
    
    Args:
        dataset_name: 'ml-100k' or 'filmtrust'
        
    Returns:
        R: 2D numpy array of shape (N_users, N_items), 0.0 indicates unrated.
        metadata: Dict containing user/item counts and id mappings.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected one of {list(DATASET_CONFIGS.keys())}")
        
    cfg = DATASET_CONFIGS[dataset_name]
    if not os.path.exists(cfg.filepath):
        raise FileNotFoundError(f"Dataset file not found at: {cfg.filepath}")
        
    if dataset_name == "ml-100k":
        # Format: user_id item_id rating timestamp (tab-separated)
        df = pd.read_csv(
            cfg.filepath, 
            sep="\t", 
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="c"
        )
    elif dataset_name == "filmtrust":
        # Format: user_id item_id rating (space-separated)
        df = pd.read_csv(
            cfg.filepath, 
            sep=" ", 
            names=["user_id", "item_id", "rating"],
            engine="c"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    # Map raw user_ids and item_ids to contiguous 0-based indices
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    
    user_map = {raw_id: idx for idx, raw_id in enumerate(unique_users)}
    item_map = {raw_id: idx for idx, raw_id in enumerate(unique_items)}
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    u_idx = df["user_id"].map(user_map).values
    i_idx = df["item_id"].map(item_map).values
    ratings = df["rating"].values.astype(float)
    
    R = np.zeros((n_users, n_items), dtype=float)
    R[u_idx, i_idx] = ratings
    
    metadata = {
        "dataset_name": dataset_name,
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": len(df),
        "r_min": cfg.r_min,
        "r_max": cfg.r_max,
        "r_med": cfg.r_med,
        "relevance_threshold": cfg.relevance_threshold,
        "user_map": user_map,
        "item_map": item_map
    }
    
    return R, metadata


def split_train_test(
    R: np.ndarray,
    train_user_ratio: float = 0.70,
    train_item_ratio: float = 0.70,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions dataset according to paper section 4.1:
    1. 70% of all users are training users (U_train), 30% are test users (U_test).
    2. For each test user, 70% of their rated items are probe training ratings,
       and 30% are holdout target ratings for recommendation evaluation.
       
    Args:
        R: Full (N_users, N_items) rating matrix.
        train_user_ratio: Fraction of users in training set (default 0.70).
        train_item_ratio: Fraction of rated items in test user probe set (default 0.70).
        seed: Random seed for reproducibility.
        
    Returns:
        train_users: (N_train,) indices of training users.
        test_users: (N_test,) indices of testing users.
        R_train: (N_train, N_items) matrix of training users' ratings.
        R_test_train: (N_test, N_items) matrix of test users' probe ratings.
        R_test_eval: (N_test, N_items) matrix of test users' holdout target ratings.
    """
    n_users, n_items = R.shape
    rng = np.random.default_rng(seed)
    
    # 1. User Split
    shuffled_users = rng.permutation(n_users)
    n_train_users = int(np.round(n_users * train_user_ratio))
    
    train_users = np.sort(shuffled_users[:n_train_users])
    test_users = np.sort(shuffled_users[n_train_users:])
    
    R_train = np.copy(R[train_users])
    
    # 2. Test-User Item Split
    n_test_users = len(test_users)
    R_test_train = np.zeros((n_test_users, n_items), dtype=float)
    R_test_eval = np.zeros((n_test_users, n_items), dtype=float)
    
    for i, u_idx in enumerate(test_users):
        user_ratings = R[u_idx]
        rated_items = np.where(user_ratings > 0)[0]
        n_rated = len(rated_items)
        
        if n_rated == 0:
            continue
            
        shuffled_items = rng.permutation(rated_items)
        n_probe = int(np.round(n_rated * train_item_ratio))
        # Ensure at least 1 item in probe if n_rated >= 2, and at least 1 in test
        if n_rated >= 2:
            n_probe = max(1, min(n_probe, n_rated - 1))
            
        probe_items = shuffled_items[:n_probe]
        eval_items = shuffled_items[n_probe:]
        
        R_test_train[i, probe_items] = user_ratings[probe_items]
        R_test_eval[i, eval_items] = user_ratings[eval_items]
        
    return train_users, test_users, R_train, R_test_train, R_test_eval

