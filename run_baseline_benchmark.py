"""
Run 5-Fold Cross-Validation for Baseline NAS on ML-100K and FilmTrust.
Saves official baseline performance to results/baseline_nas_cv.json
used for paired t-test hypothesis testing (p < 0.05) and F1@10 regression checks.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".packages")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.similarities.nas_v2 import compute_nas_v2_matrix
from src.recommender import predict_ratings
from src.metrics import evaluate_predictions_v3
from src.evaluation_v3 import load_ml100k_fold, load_filmtrust_with_trust

K_EVAL_LIST = (5, 10, 20, 30, 40, 50, 60)


def run_baseline_ml100k():
    print("[BASELINE NAS] Running official 5-Fold Cross-Validation on ML-100K...")
    fold_results = []
    for fold in range(1, 6):
        t0 = time.time()
        R_tr, R_te, r_min, r_max, r_med, threshold = load_ml100k_fold(fold)

        # Baseline NAS
        sim = compute_nas_v2_matrix(R_tr, variant="tau", tau_prox=1.0, tau_sing=1.0)
        test_u, test_i, actual, pred = predict_ratings(
            sim_matrix=sim,
            R_train=R_tr,
            R_test_train=R_tr,
            R_test_eval=R_te,
            K=20,
            method="weighted_average",
            r_min=r_min,
            r_max=r_max
        )
        metrics = evaluate_predictions_v3(
            user_indices=test_u,
            actual=actual,
            predicted=pred,
            k_values=K_EVAL_LIST,
            threshold=threshold
        )
        fold_results.append(metrics)
        print(f"  Fold {fold}/5: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, F1@10={metrics['F1@10']:.4f}, FCP={metrics['FCP']:.4f} ({time.time() - t0:.2f}s)")

    metric_keys = list(fold_results[0].keys())
    mean_metrics = {k: float(np.mean([f[k] for f in fold_results])) for k in metric_keys}
    std_metrics = {k: float(np.std([f[k] for f in fold_results])) for k in metric_keys}

    print(f"--> Baseline NAS ML-100K 5-Fold Mean MAE: {mean_metrics['MAE']:.4f} +/- {std_metrics['MAE']:.4f}")
    print(f"--> Baseline NAS ML-100K 5-Fold Mean F1@10: {mean_metrics['F1@10']:.4f} +/- {std_metrics['F1@10']:.4f}")

    return {
        "dataset": "ml-100k",
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics
    }


def run_baseline_filmtrust(seed=42):
    print("\n[BASELINE NAS] Running 5-Fold Cross-Validation on FilmTrust...")
    R_full, _, r_min, r_max, r_med, threshold = load_filmtrust_with_trust()
    n_users = R_full.shape[0]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_users)
    folds_idx = np.array_split(indices, 5)

    fold_results = []
    for f_idx, test_users in enumerate(folds_idx):
        t0 = time.time()
        R_train = np.copy(R_full)
        R_test_eval = np.zeros_like(R_full)

        for u in test_users:
            rated = np.where(R_full[u] > 0)[0]
            if len(rated) >= 2:
                shuffled = rng.permutation(rated)
                n_probe = max(1, int(round(len(rated) * 0.70)))
                eval_items = shuffled[n_probe:]
                R_train[u, eval_items] = 0.0
                R_test_eval[u, eval_items] = R_full[u, eval_items]

        sim = compute_nas_v2_matrix(R_train, variant="tau", tau_prox=1.0, tau_sing=1.0)
        test_u, test_i, actual, pred = predict_ratings(
            sim_matrix=sim,
            R_train=R_train,
            R_test_train=R_train,
            R_test_eval=R_test_eval,
            K=20,
            method="weighted_average",
            r_min=r_min,
            r_max=r_max
        )
        metrics = evaluate_predictions_v3(
            user_indices=test_u,
            actual=actual,
            predicted=pred,
            k_values=K_EVAL_LIST,
            threshold=threshold
        )
        fold_results.append(metrics)
        print(f"  Fold {f_idx + 1}/5: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, F1@10={metrics['F1@10']:.4f}, FCP={metrics['FCP']:.4f} ({time.time() - t0:.2f}s)")

    metric_keys = list(fold_results[0].keys())
    mean_metrics = {k: float(np.mean([f[k] for f in fold_results])) for k in metric_keys}
    std_metrics = {k: float(np.std([f[k] for f in fold_results])) for k in metric_keys}

    print(f"--> Baseline NAS FilmTrust 5-Fold Mean MAE: {mean_metrics['MAE']:.4f} +/- {std_metrics['MAE']:.4f}")
    print(f"--> Baseline NAS FilmTrust 5-Fold Mean F1@10: {mean_metrics['F1@10']:.4f} +/- {std_metrics['F1@10']:.4f}")

    return {
        "dataset": "filmtrust",
        "folds": fold_results,
        "mean": mean_metrics,
        "std": std_metrics
    }


def main():
    os.makedirs("results", exist_ok=True)
    ml100k_base = run_baseline_ml100k()
    filmtrust_base = run_baseline_filmtrust()

    combined = {
        "ml-100k": ml100k_base,
        "filmtrust": filmtrust_base
    }
    with open("results/baseline_nas_cv.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("\nSaved baseline CV benchmarks to results/baseline_nas_cv.json")


if __name__ == "__main__":
    main()
