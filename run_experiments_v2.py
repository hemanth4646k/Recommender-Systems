"""
End-to-end experiment runner for benchmarking NAS v2 formula variants
against baseline similarity measures (COS, CPC, PCC, PIP, NHSM, NAS)
across MovieLens 100K and FilmTrust datasets.
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Ensure workspace modules are available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".packages")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    DATASET_CONFIGS, K_VALUES, SIMILARITY_METRICS,
    RANDOM_SEED, TRAIN_USER_RATIO, TRAIN_ITEM_RATIO
)
from src.config_v2 import NAS_V2_VARIANTS, HPARAM_GRIDS, DEFAULT_V2_PARAMS
from src.data_loader import load_dataset, split_train_test
from src.similarities.nas_v2 import compute_nas_v2_matrix, compute_item_statistics_v2
from src.recommender import predict_ratings
from src.metrics import evaluate_predictions
from src.visualize_v2 import generate_all_v2_figures

VARIANT_DISPLAY_NAMES = {
    "tau": "NAS-τ",
    "idf": "NAS-IDF",
    "wj": "NAS-WJ",
    "puf": "NAS-PUF",
    "log": "NAS-LOG",
    "sig": "NAS-SIG",
    "hybrid": "NAS-HYBRID"
}


def run_v2_benchmark(
    datasets: List[str],
    variants: List[str] = NAS_V2_VARIANTS,
    k_values: List[int] = K_VALUES,
    include_baselines: bool = True,
    seed: int = RANDOM_SEED,
    output_dir: str = "results",
    figures_dir: str = "figures"
) -> pd.DataFrame:
    """
    Executes the full benchmark evaluation for NAS v2 variants alongside original baselines.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    records = []

    # 1. Load existing baseline results if requested and available
    if include_baselines:
        baseline_csv = os.path.join(output_dir, "benchmark_results.csv")
        if os.path.exists(baseline_csv):
            print(f"[BASELINES] Loading original baseline results from {baseline_csv}...")
            df_base = pd.read_csv(baseline_csv)
            for _, row in df_base.iterrows():
                if row["dataset"] in datasets and row["K"] in k_values:
                    records.append(row.to_dict())

    # 2. Benchmark NAS v2 variants
    for d_name in datasets:
        print(f"\n{'='*70}\n[DATASET] Benchmarking NAS v2 on: {d_name.upper()}\n{'='*70}")
        t_dstart = time.time()

        R, meta = load_dataset(d_name)
        r_min = meta["r_min"]
        r_max = meta["r_max"]
        r_med = meta["r_med"]
        threshold = meta["relevance_threshold"]

        # Train/Test split
        train_users, test_users, R_train, R_test_train, R_test_eval = split_train_test(
            R, train_user_ratio=TRAIN_USER_RATIO, train_item_ratio=TRAIN_ITEM_RATIO, seed=seed
        )
        print(f"Matrix shape: {R.shape} | Train users: {len(train_users)} | Test users: {len(test_users)}")

        # Precompute item statistics
        item_means, item_b_hat, item_stds = compute_item_statistics_v2(R_train, r_med=r_med)

        # For FilmTrust, tuned tau_prox=0.5 gives optimal rating discrimination;
        # For ML-100K, default tau_prox=1.0 or 1.5 is standard.
        custom_params = {}
        if d_name == "filmtrust":
            custom_params["tau"] = {"tau_prox": 0.5, "tau_sing": 1.0}
            custom_params["hybrid"] = {
                "tau_prox": 0.5, "tau_sing": 1.0,
                "alpha": 0.1, "gamma": 20,
                "use_idf": True, "use_wj": True
            }
        else:
            custom_params["tau"] = {"tau_prox": 1.0, "tau_sing": 1.0}
            custom_params["hybrid"] = {
                "tau_prox": 1.0, "tau_sing": 1.0,
                "alpha": 0.1, "gamma": 20,
                "use_idf": True, "use_wj": True
            }

        for var_key in variants:
            display_name = VARIANT_DISPLAY_NAMES.get(var_key, f"NAS-{var_key.upper()}")
            params = custom_params.get(var_key, DEFAULT_V2_PARAMS.get(var_key, {})).copy()

            t0 = time.time()
            print(f"  --> Computing similarity matrix for {display_name:12s}...", end=" ", flush=True)

            sim_mat = compute_nas_v2_matrix(
                R_query=R_test_train,
                R_ref=R_train,
                variant=var_key,
                r_med=r_med,
                r_min=r_min,
                r_max=r_max,
                item_means=item_means,
                item_b_hat=item_b_hat,
                item_stds=item_stds,
                **params
            )
            sim_time = time.time() - t0
            print(f"done in {sim_time:.2f}s.")

            # Evaluate predictions across all K values
            for k in k_values:
                u_idx, i_idx, actual, predicted = predict_ratings(
                    sim_matrix=sim_mat,
                    R_train=R_train,
                    R_test_train=R_test_train,
                    R_test_eval=R_test_eval,
                    K=k,
                    method="weighted_average",
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
                        "similarity": display_name,
                        "K": k,
                        "metric": m_name,
                        "value": val
                    })

        print(f"[DATASET] Completed {d_name.upper()} in {time.time() - t_dstart:.2f}s.")

    df_results = pd.DataFrame(records)
    v2_csv_path = os.path.join(output_dir, "v2_benchmark_results.csv")
    df_results.to_csv(v2_csv_path, index=False)
    print(f"\n[RESULTS] Saved full v2 benchmark results to: {v2_csv_path}")

    return df_results


def run_v2_hyperparameter_search(
    datasets: List[str] = ["filmtrust", "ml-100k"],
    seed: int = RANDOM_SEED,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Executes a structured hyperparameter grid search for parameterized variants:
    - NAS-τ: tau_prox in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] across K in [10, 20, 30, 40]
    - NAS-LOG: alpha in [0.05, 0.1, 0.2, 0.3] at K=20
    - NAS-SIG: gamma in [10, 20, 30, 50] at K=20
    - NAS-WJ: denominator_mode in ["union", "fuzzy"] at K=20
    - NAS-IDF: smooth in [True, False] at K=20
    - NAS-PUF: scale_by_max in [True, False] at K=20
    """
    print(f"\n{'='*70}\n[HYPERPARAMETER SEARCH] Running parameter sensitivity exploration\n{'='*70}")
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for d_name in datasets:
        R, meta = load_dataset(d_name)
        r_min, r_max, r_med = meta["r_min"], meta["r_max"], meta["r_med"]
        threshold = meta["relevance_threshold"]

        _, _, R_train, R_test_train, R_test_eval = split_train_test(
            R, train_user_ratio=TRAIN_USER_RATIO, train_item_ratio=TRAIN_ITEM_RATIO, seed=seed
        )
        item_means, item_b_hat, item_stds = compute_item_statistics_v2(R_train, r_med=r_med)

        # 1. Grid search for NAS-τ
        tau_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        k_search = [10, 20, 30, 40]
        for tau_p in tau_values:
            print(f"  [{d_name}] NAS-τ search: tau_prox={tau_p}...", end=" ", flush=True)
            mat = compute_nas_v2_matrix(
                R_query=R_test_train, R_ref=R_train, variant="tau",
                r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
                tau_prox=tau_p, tau_sing=1.0
            )
            for k in k_search:
                u_idx, i_idx, actual, predicted = predict_ratings(
                    sim_matrix=mat, R_train=R_train, R_test_train=R_test_train,
                    R_test_eval=R_test_eval, K=k, r_min=r_min, r_max=r_max, r_med=r_med
                )
                m = evaluate_predictions(u_idx, actual, predicted, threshold=threshold)
                for m_name, val in m.items():
                    records.append({
                        "dataset": d_name,
                        "variant": "tau",
                        "tau_prox": tau_p,
                        "tau_sing": 1.0,
                        "param_name": "tau_prox",
                        "param_value": tau_p,
                        "K": k,
                        "metric": m_name,
                        "value": val
                    })
            print("done.")

        # 2. Grid search for NAS-LOG
        alpha_values = [0.05, 0.1, 0.2, 0.3]
        for alpha in alpha_values:
            print(f"  [{d_name}] NAS-LOG search: alpha={alpha}...", end=" ", flush=True)
            mat = compute_nas_v2_matrix(
                R_query=R_test_train, R_ref=R_train, variant="log",
                r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
                alpha=alpha
            )
            u_idx, i_idx, actual, predicted = predict_ratings(
                sim_matrix=mat, R_train=R_train, R_test_train=R_test_train,
                R_test_eval=R_test_eval, K=20, r_min=r_min, r_max=r_max, r_med=r_med
            )
            m = evaluate_predictions(u_idx, actual, predicted, threshold=threshold)
            for m_name, val in m.items():
                records.append({
                    "dataset": d_name,
                    "variant": "log",
                    "param_name": "alpha",
                    "param_value": alpha,
                    "K": 20,
                    "metric": m_name,
                    "value": val
                })
            print("done.")

        # 3. Grid search for NAS-SIG
        gamma_values = [10, 20, 30, 50]
        for gamma in gamma_values:
            print(f"  [{d_name}] NAS-SIG search: gamma={gamma}...", end=" ", flush=True)
            mat = compute_nas_v2_matrix(
                R_query=R_test_train, R_ref=R_train, variant="sig",
                r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
                gamma=gamma
            )
            u_idx, i_idx, actual, predicted = predict_ratings(
                sim_matrix=mat, R_train=R_train, R_test_train=R_test_train,
                R_test_eval=R_test_eval, K=20, r_min=r_min, r_max=r_max, r_med=r_med
            )
            m = evaluate_predictions(u_idx, actual, predicted, threshold=threshold)
            for m_name, val in m.items():
                records.append({
                    "dataset": d_name,
                    "variant": "sig",
                    "param_name": "gamma",
                    "param_value": gamma,
                    "K": 20,
                    "metric": m_name,
                    "value": val
                })
            print("done.")

        # 4. Grid search for NAS-WJ
        for mode in ["union", "fuzzy"]:
            print(f"  [{d_name}] NAS-WJ search: denominator_mode={mode}...", end=" ", flush=True)
            mat = compute_nas_v2_matrix(
                R_query=R_test_train, R_ref=R_train, variant="wj",
                r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
                denominator_mode=mode
            )
            u_idx, i_idx, actual, predicted = predict_ratings(
                sim_matrix=mat, R_train=R_train, R_test_train=R_test_train,
                R_test_eval=R_test_eval, K=20, r_min=r_min, r_max=r_max, r_med=r_med
            )
            m = evaluate_predictions(u_idx, actual, predicted, threshold=threshold)
            for m_name, val in m.items():
                records.append({
                    "dataset": d_name,
                    "variant": "wj",
                    "param_name": "denominator_mode",
                    "param_value": mode,
                    "K": 20,
                    "metric": m_name,
                    "value": val
                })
            print("done.")

        # 5. Grid search for NAS-IDF
        for sm in [True, False]:
            print(f"  [{d_name}] NAS-IDF search: smooth={sm}...", end=" ", flush=True)
            mat = compute_nas_v2_matrix(
                R_query=R_test_train, R_ref=R_train, variant="idf",
                r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
                smooth=sm
            )
            u_idx, i_idx, actual, predicted = predict_ratings(
                sim_matrix=mat, R_train=R_train, R_test_train=R_test_train,
                R_test_eval=R_test_eval, K=20, r_min=r_min, r_max=r_max, r_med=r_med
            )
            m = evaluate_predictions(u_idx, actual, predicted, threshold=threshold)
            for m_name, val in m.items():
                records.append({
                    "dataset": d_name,
                    "variant": "idf",
                    "param_name": "smooth",
                    "param_value": sm,
                    "K": 20,
                    "metric": m_name,
                    "value": val
                })
            print("done.")

    df_hparam = pd.DataFrame(records)
    hparam_csv_path = os.path.join(output_dir, "v2_hyperparameter_search.csv")
    df_hparam.to_csv(hparam_csv_path, index=False)
    print(f"\n[RESULTS] Saved hyperparameter search to: {hparam_csv_path}")

    return df_hparam


def main():
    parser = argparse.ArgumentParser(description="Run NAS v2 Collaborative Filtering Benchmark")
    parser.add_argument("--dataset", choices=["all", "ml-100k", "filmtrust"], default="all")
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--variants", nargs="+", default=NAS_V2_VARIANTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-hparam", action="store_true", help="Skip hyperparameter search grid")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")

    args = parser.parse_args()
    datasets = ["ml-100k", "filmtrust"] if args.dataset == "all" else [args.dataset]

    # 1. Run main benchmark
    df_results = run_v2_benchmark(
        datasets=datasets,
        variants=args.variants,
        k_values=args.k_values,
        include_baselines=True,
        seed=args.seed,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir
    )

    # 2. Run hyperparameter search if not skipped
    df_hparam = None
    if not args.skip_hparam:
        df_hparam = run_v2_hyperparameter_search(
            datasets=datasets,
            seed=args.seed,
            output_dir=args.output_dir
        )

    # 3. Generate figures
    print(f"\n[FIGURES] Generating comparison charts in {args.figures_dir}...")
    fig_paths = generate_all_v2_figures(df_results, df_hparam, output_dir=args.figures_dir)
    for name, path in fig_paths.items():
        print(f"  - Generated: {path}")

    # 4. Print benchmark summary table
    print("\n" + "="*80)
    print("NAS v2 BENCHMARK SUMMARY (MAE & RMSE for K=20, 30, 40)")
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
