"""
Evaluation and benchmarking harness for NAS v3:
- Loads official ML-100K 5-fold CV splits (u1..u5)
- Loads FilmTrust ratings and trust network
- Executes full 5-fold CV across K in {5, 10, 20, 30, 40, 50, 60}
- Performs paired statistical hypothesis testing (p < 0.05) against Baseline NAS
- Verifies merge eligibility without F1@10 regression
"""
import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.similarities.nas_v3 import (
    NASv3Config,
    compute_nas_v3_similarity,
    predict_ratings_v3
)
from src.metrics import evaluate_predictions_v3
from src.config import DATASET_CONFIGS


def load_ml100k_fold(fold: int = 1) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Loads official ML-100K train (u{fold}.base) and test (u{fold}.test) matrices.
    Returns:
        R_train: (943, 1682) training rating matrix.
        R_test_eval: (943, 1682) holdout test rating matrix.
        r_min, r_max, r_med, threshold
    """
    base_file = f"data/ml-100k/u{fold}.base"
    test_file = f"data/ml-100k/u{fold}.test"
    if not os.path.exists(base_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"ML-100K fold {fold} files not found: {base_file}, {test_file}")

    df_base = pd.read_csv(base_file, sep="\t", names=["u", "i", "r", "t"])
    df_test = pd.read_csv(test_file, sep="\t", names=["u", "i", "r", "t"])

    n_users = 943
    n_items = 1682

    R_train = np.zeros((n_users, n_items), dtype=np.float64)
    R_train[df_base["u"].values - 1, df_base["i"].values - 1] = df_base["r"].values.astype(np.float64)

    R_test_eval = np.zeros((n_users, n_items), dtype=np.float64)
    R_test_eval[df_test["u"].values - 1, df_test["i"].values - 1] = df_test["r"].values.astype(np.float64)

    r_min, r_max, r_med, threshold = 1.0, 5.0, 3.0, 4.0
    return R_train, R_test_eval, r_min, r_max, r_med, threshold


def load_filmtrust_with_trust() -> Tuple[np.ndarray, Optional[np.ndarray], float, float, float, float]:
    """
    Loads FilmTrust ratings matrix and trust matrix mapped to contiguous user IDs.
    """
    ratings_file = "data/filmtrust/ratings.txt"
    trust_file = "data/filmtrust/trust.txt"

    df_r = pd.read_csv(ratings_file, sep=" ", names=["u", "i", "r"])
    unique_users = sorted(df_r["u"].unique())
    unique_items = sorted(df_r["i"].unique())

    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}

    n_users = len(unique_users)
    n_items = len(unique_items)

    R = np.zeros((n_users, n_items), dtype=np.float64)
    R[df_r["u"].map(user_map).values, df_r["i"].map(item_map).values] = df_r["r"].values.astype(np.float64)

    # Trust matrix
    trust_matrix = np.zeros((n_users, n_users), dtype=np.float64)
    if os.path.exists(trust_file):
        df_t = pd.read_csv(trust_file, sep=" ", names=["from", "to", "trust"])
        # Only retain trust relations among known rating users
        valid = df_t["from"].isin(user_map) & df_t["to"].isin(user_map)
        df_t_valid = df_t[valid]
        u_from = df_t_valid["from"].map(user_map).values
        u_to = df_t_valid["to"].map(user_map).values
        trust_matrix[u_from, u_to] = df_t_valid["trust"].values.astype(np.float64)

    r_min, r_max, r_med, threshold = 0.5, 4.0, 2.25, 3.0
    return R, trust_matrix, r_min, r_max, r_med, threshold


def evaluate_single_fold(
    R_train: np.ndarray,
    R_test_eval: np.ndarray,
    r_med: float,
    r_min: float,
    r_max: float,
    threshold: float,
    cfg: NASv3Config = NASv3Config(),
    K_neighbors: int = 20,
    k_values: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60),
    trust_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes NAS v3 similarity, predicts ratings, and computes all target metrics.
    """
    sim = compute_nas_v3_similarity(R_train, r_med=r_med, cfg=cfg, trust_matrix=trust_matrix)
    test_u, test_i, actual, pred = predict_ratings_v3(
        sim_matrix=sim,
        R_train=R_train,
        R_test_train=R_train,
        R_test_eval=R_test_eval,
        K=K_neighbors,
        cfg=cfg,
        r_min=r_min,
        r_max=r_max
    )
    metrics = evaluate_predictions_v3(
        user_indices=test_u,
        actual=actual,
        predicted=pred,
        k_values=k_values,
        threshold=threshold
    )
    return metrics


def run_5fold_cv_ml100k(
    cfg: NASv3Config = NASv3Config(),
    K_neighbors: int = 20,
    k_values: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60)
) -> Dict[str, Any]:
    """
    Executes 5-fold cross-validation across all official folds (u1..u5) of ML-100K.
    """
    fold_results = []
    for fold in range(1, 6):
        R_tr, R_te, r_min, r_max, r_med, threshold = load_ml100k_fold(fold)
        m = evaluate_single_fold(
            R_train=R_tr,
            R_test_eval=R_te,
            r_med=r_med,
            r_min=r_min,
            r_max=r_max,
            threshold=threshold,
            cfg=cfg,
            K_neighbors=K_neighbors,
            k_values=k_values
        )
        fold_results.append(m)

    # Compute mean and standard deviation across folds
    metric_keys = list(fold_results[0].keys())
    mean_metrics = {k: float(np.mean([f[k] for f in fold_results])) for k in metric_keys}
    std_metrics = {k: float(np.std([f[k] for f in fold_results])) for k in metric_keys}

    return {
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics
    }


def run_5fold_cv_filmtrust(
    cfg: NASv3Config = NASv3Config(),
    K_neighbors: int = 20,
    k_values: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60),
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes 5-fold cross-validation on FilmTrust ratings.
    """
    R, trust_mat, r_min, r_max, r_med, threshold = load_filmtrust_with_trust()
    n_users = R.shape[0]

    # Partition users into 5 folds
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_users)
    folds_idx = np.array_split(indices, 5)

    fold_results = []
    for f_idx, test_users in enumerate(folds_idx):
        train_users = np.setdiff1d(indices, test_users)
        R_train = np.copy(R)
        # In UBCF evaluation: test users' ratings are split 70% probe, 30% holdout target
        R_test_eval = np.zeros_like(R)
        for u in test_users:
            rated_items = np.where(R[u] > 0)[0]
            if len(rated_items) == 0:
                continue
            shuffled = rng.permutation(rated_items)
            n_probe = max(1, int(round(len(rated_items) * 0.70))) if len(rated_items) >= 2 else len(rated_items)
            eval_items = shuffled[n_probe:]
            R_train[u, eval_items] = 0.0
            R_test_eval[u, eval_items] = R[u, eval_items]

        m = evaluate_single_fold(
            R_train=R_train,
            R_test_eval=R_test_eval,
            r_med=r_med,
            r_min=r_min,
            r_max=r_max,
            threshold=threshold,
            cfg=cfg,
            K_neighbors=K_neighbors,
            k_values=k_values,
            trust_matrix=trust_mat
        )
        fold_results.append(m)

    metric_keys = list(fold_results[0].keys())
    mean_metrics = {k: float(np.mean([f[k] for f in fold_results])) for k in metric_keys}
    std_metrics = {k: float(np.std([f[k] for f in fold_results])) for k in metric_keys}

    return {
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics
    }


def compute_statistical_acceptance(
    baseline_cv: Dict[str, Any],
    candidate_cv: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Checks acceptance criteria:
    - Statistically significant reduction in 5-fold CV MAE (paired t-test, p < 0.05)
    - No regression in F1@10 (mean F1@10 candidate >= mean F1@10 baseline - 1e-4)
    """
    base_maes = [f["MAE"] for f in baseline_cv["folds"]]
    cand_maes = [f["MAE"] for f in candidate_cv["folds"]]

    # Paired t-test: alternative 'greater' tests if baseline_mae > candidate_mae
    res = stats.ttest_rel(base_maes, cand_maes, alternative="greater")
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    mae_diff = float(np.mean(base_maes) - np.mean(cand_maes))
    f1_10_base = baseline_cv["mean"]["F1@10"]
    f1_10_cand = candidate_cv["mean"]["F1@10"]
    f1_diff = float(f1_10_cand - f1_10_base)

    no_f1_regression = f1_diff >= -1e-4
    significant_mae_reduction = (p_val < 0.05) and (mae_diff > 0.0)

    eligible = bool(significant_mae_reduction and no_f1_regression)

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "baseline_mae": float(np.mean(base_maes)),
        "candidate_mae": float(np.mean(cand_maes)),
        "mae_reduction": mae_diff,
        "baseline_f1_10": f1_10_base,
        "candidate_f1_10": f1_10_cand,
        "f1_10_difference": f1_diff,
        "statistically_significant": significant_mae_reduction,
        "no_f1_regression": no_f1_regression,
        "eligible_for_merge": eligible
    }
