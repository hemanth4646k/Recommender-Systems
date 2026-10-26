"""
End-to-end experiment runner for benchmarking Novel Algorithmic Similarity (NAS)
against COS, CPC, PCC, PIP, and NHSM across ML-100K and FilmTrust datasets.
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Ensure workspace modules are available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".packages")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    DATASET_CONFIGS, K_VALUES, SIMILARITY_METRICS,
    RANDOM_SEED, TRAIN_USER_RATIO, TRAIN_ITEM_RATIO
)
from src.data_loader import load_dataset, split_train_test
from src.similarities.nas import compute_nas_matrix, compute_item_statistics
from src.similarities.traditional import (
    compute_cos_matrix, compute_pcc_matrix, compute_cpc_matrix
)
from src.similarities.heuristic import (
    compute_pip_matrix, compute_nhsm_matrix
)
from src.recommender import predict_ratings
from src.metrics import evaluate_predictions
from src.visualize import generate_all_paper_figures


def compute_similarity(
    sim_name: str,
    R_test_train: np.ndarray,
    R_train: np.ndarray,
    r_min: float,
    r_max: float,
    r_med: float,
    item_means: np.ndarray,
    item_b_hat: np.ndarray
) -> np.ndarray:
    """Dispatches to the selected vectorized similarity function."""
    if sim_name == "NAS":
        return compute_nas_matrix(
            R_query=R_test_train,
            R_ref=R_train,
            r_med=r_med,
            item_means=item_means,
            item_b_hat=item_b_hat
        )
    elif sim_name == "COS":
        return compute_cos_matrix(
            R_query=R_test_train,
            R_ref=R_train
        )
    elif sim_name == "PCC":
        return compute_pcc_matrix(
            R_query=R_test_train,
            R_ref=R_train,
            use_common_mean=False
        )
    elif sim_name == "CPC":
        return compute_cpc_matrix(
            R_query=R_test_train,
            R_ref=R_train,
            r_med=r_med
        )
    elif sim_name == "PIP":
        return compute_pip_matrix(
            R_query=R_test_train,
            R_ref=R_train,
            r_min=r_min,
            r_max=r_max,
            r_med=r_med,
            item_means=item_means
        )
    elif sim_name == "NHSM":
        return compute_nhsm_matrix(
            R_query=R_test_train,
            R_ref=R_train,
            r_med=r_med,
            item_means=item_means
        )
    else:
        raise ValueError(f"Unknown similarity metric: {sim_name}")


def run_benchmark(
    datasets: List[str],
    similarities: List[str],
    k_values: List[int],
    method: str = "weighted_average",
    seed: int = RANDOM_SEED,
    output_dir: str = "results",
    figures_dir: str = "figures"
) -> pd.DataFrame:
    """Executes the full benchmark pipeline across specified datasets and similarity metrics."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    records = []
    
    for d_name in datasets:
        print(f"\n{'='*70}\n[DATASET] Running benchmark for: {d_name.upper()}\n{'='*70}")
        t_start = time.time()
        
        # 1. Load data
        R, meta = load_dataset(d_name)
        r_min = meta["r_min"]
        r_max = meta["r_max"]
        r_med = meta["r_med"]
        threshold = meta["relevance_threshold"]
        
        print(f"Matrix shape: {R.shape} | Ratings: {meta['n_ratings']} | Scale: [{r_min}, {r_max}], Median={r_med}")
        
        # 2. Train/Test split
        train_users, test_users, R_train, R_test_train, R_test_eval = split_train_test(
            R, train_user_ratio=TRAIN_USER_RATIO, train_item_ratio=TRAIN_ITEM_RATIO, seed=seed
        )
        print(f"Split: {len(train_users)} train users, {len(test_users)} test users.")
        n_eval_ratings = np.sum(R_test_eval > 0)
        print(f"Evaluation target ratings: {n_eval_ratings}")
        
        # 3. Item statistics
        item_means, item_b_hat = compute_item_statistics(R_train, r_med=r_med)
        
        # 4. Iterate over similarity metrics
        for sim_name in similarities:
            sim_t0 = time.time()
            print(f"  --> Computing similarity matrix for {sim_name:5s}...", end=" ", flush=True)
            
            sim_mat = compute_similarity(
                sim_name=sim_name,
                R_test_train=R_test_train,
                R_train=R_train,
                r_min=r_min,
                r_max=r_max,
                r_med=r_med,
                item_means=item_means,
                item_b_hat=item_b_hat
            )
            sim_time = time.time() - sim_t0
            print(f"done in {sim_time:.2f}s.")
            
            # 5. Evaluate for each K
            for k in k_values:
                u_idx, i_idx, actual, predicted = predict_ratings(
                    sim_matrix=sim_mat,
                    R_train=R_train,
                    R_test_train=R_test_train,
                    R_test_eval=R_test_eval,
                    K=k,
                    method=method,
                    r_min=r_min,
                    r_max=r_max,
                    r_med=r_med
                )
                
                metrics = evaluate_predictions(
                    user_indices=u_idx,
                    actual=actual,
                    predicted=predicted,
                    threshold=threshold
                )
                
                for m_name, val in metrics.items():
                    records.append({
                        "dataset": d_name,
                        "similarity": sim_name,
                        "K": k,
                        "metric": m_name,
                        "value": val
                    })
                    
        print(f"[DATASET] Finished {d_name.upper()} in {time.time() - t_start:.2f}s.")
        
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[RESULTS] Saved tabular results to: {csv_path}")
    
    # Generate all paper figures
    print(f"[FIGURES] Generating reproduction figures in: {figures_dir}...")
    fig_paths = generate_all_paper_figures(df_results, output_dir=figures_dir)
    for fig_name, fig_path in fig_paths.items():
        print(f"  - Generated: {fig_path}")
        
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Run NAS Collaborative Filtering Benchmark")
    parser.add_argument(
        "--dataset", choices=["all", "ml-100k", "filmtrust"], default="all",
        help="Dataset(s) to benchmark (default: all)"
    )
    parser.add_argument(
        "--similarities", nargs="+", default=SIMILARITY_METRICS,
        help="Similarity metrics to include (default: all 6)"
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=K_VALUES,
        help="List of neighbor K values (default: 10 20 30 40 60 80)"
    )
    parser.add_argument(
        "--method", choices=["weighted_average", "resnick"], default="weighted_average",
        help="Prediction method (default: weighted_average)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save CSV results"
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures",
        help="Directory to save generated plots"
    )
    
    args = parser.parse_args()
    
    datasets = ["ml-100k", "filmtrust"] if args.dataset == "all" else [args.dataset]
    
    df_results = run_benchmark(
        datasets=datasets,
        similarities=args.similarities,
        k_values=args.k_values,
        method=args.method,
        seed=args.seed,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir
    )
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (MAE & RMSE for K=20, 30, 40)")
    print("="*80)
    summary_df = df_results[
        (df_results["metric"].isin(["MAE", "RMSE"])) & 
        (df_results["K"].isin([20, 30, 40]))
    ].pivot_table(
        index=["dataset", "metric", "similarity"], 
        columns="K", 
        values="value"
    )
    print(summary_df.round(4))
    print("="*80)


if __name__ == "__main__":
    main()

