"""
End-to-End Benchmark and Sequential Ablation for NAS v3 Grand Synthesis.
Combines optimal components across Families A-F:
- Family A: Elastic NAS exponents (alpha=1.25, beta=0.5, gamma=0.35)
- Family B: Dual-Center SVS (eta_1=0.5, eta_2=0.2)
- Family C: Shrinkage Overlap (tau=12.0)
- Family D: Surprisal Attitude PS (p=1.0, rho=0.7, zeta=1.5, xi=0.5)
- Family E: Case Amplification & Global Baseline Predictor (q=1.0, reg_u=15, reg_i=25, blend_weight=0.3)
- Family F: 2-Hop Katz Trust Harmonic/Linear Fusion for FilmTrust (beta=0.1, gamma_cold=0.5)

Evaluates:
- 5-Fold Cross-Validation on ML-100K and FilmTrust
- Target Metrics: MAE, RMSE, FCP, Precision@K, Recall@K, F1@K, MAP@K, MRR@K for K in {5, 10, 20, 30, 40, 50, 60}
- Statistical significance hypothesis testing (p < 0.05) & F1@10 verification
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".packages")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.similarities.nas_v3 import NASv3Config
from src.evaluation_v3 import (
    run_5fold_cv_ml100k,
    run_5fold_cv_filmtrust,
    compute_statistical_acceptance
)

K_EVAL_LIST = (5, 10, 20, 30, 40, 50, 60)


def build_grand_synthesis_config() -> NASv3Config:
    return NASv3Config(
        alpha_ps=1.25,
        beta_svs=0.50,
        gamma_ovl=0.35,
        blend_mode="multiplicative",
        svs_mode="dual",
        eta_disp=0.5,
        eta_mean=0.2,
        overlap_type="shrinkage",
        shrinkage_tau=12.0,
        tversky_theta=0.8,
        tversky_symmetric=True,
        dist_p=1.0,
        rho_raw_vs_dev=0.70,
        ps_zeta=1.5,
        cross_median_penalty_xi=0.5,
        var_weight_kappa=0.0,
        use_surprisal=True,
        predictor_mode="case_amplification",
        case_amp_q=1.0,
        epsilon_reg=0.001,
        baseline_reg_u=15.0,
        baseline_reg_i=25.0,
        blend_baseline_weight=0.35,
        use_trust=True,
        beta_katz=0.1,
        gamma_cold=0.5,
        trust_fusion_mode="linear_blend"
    )


def run_sequential_ablation() -> pd.DataFrame:
    """Executes sequential step-by-step ablation study on ML-100K 5-fold CV."""
    print("\n" + "=" * 75)
    print("EXECUTING SEQUENTIAL ABLATION STUDY (ML-100K 5-FOLD CV)")
    print("=" * 75)

    ablation_steps = [
        ("Step 0: Baseline NAS", NASv3Config(
            alpha_ps=1.0, beta_svs=1.0, gamma_ovl=1.0,
            blend_mode="multiplicative", svs_mode="surprisal",
            overlap_type="jaccard", use_surprisal=False,
            predictor_mode="weighted_average", blend_baseline_weight=0.0
        )),
        ("Step 1: + Elastic Exponents (Family A)", NASv3Config(
            alpha_ps=1.25, beta_svs=0.50, gamma_ovl=0.35,
            blend_mode="multiplicative", svs_mode="surprisal",
            overlap_type="jaccard", use_surprisal=False,
            predictor_mode="weighted_average", blend_baseline_weight=0.0
        )),
        ("Step 2: + Dual-Center SVS (Family B)", NASv3Config(
            alpha_ps=1.25, beta_svs=0.50, gamma_ovl=0.35,
            blend_mode="multiplicative", svs_mode="dual",
            eta_disp=0.5, eta_mean=0.2,
            overlap_type="jaccard", use_surprisal=False,
            predictor_mode="weighted_average", blend_baseline_weight=0.0
        )),
        ("Step 3: + Shrinkage Overlap (Family C)", NASv3Config(
            alpha_ps=1.25, beta_svs=0.50, gamma_ovl=0.35,
            blend_mode="multiplicative", svs_mode="dual",
            eta_disp=0.5, eta_mean=0.2,
            overlap_type="shrinkage", shrinkage_tau=12.0,
            use_surprisal=False,
            predictor_mode="weighted_average", blend_baseline_weight=0.0
        )),
        ("Step 4: + Surprisal Attitude PS (Family D)", NASv3Config(
            alpha_ps=1.25, beta_svs=0.50, gamma_ovl=0.35,
            blend_mode="multiplicative", svs_mode="dual",
            eta_disp=0.5, eta_mean=0.2,
            overlap_type="shrinkage", shrinkage_tau=12.0,
            dist_p=1.0, rho_raw_vs_dev=0.70, ps_zeta=1.5,
            cross_median_penalty_xi=0.5, use_surprisal=True,
            predictor_mode="weighted_average", blend_baseline_weight=0.0
        )),
        ("Step 5: + Case Amp & Global Baseline (Grand Synthesis)", build_grand_synthesis_config())
    ]

    records = []
    base_mae = None
    for name, cfg in ablation_steps:
        t0 = time.time()
        cv = run_5fold_cv_ml100k(cfg=cfg, K_neighbors=20, k_values=K_EVAL_LIST)
        mae = cv["mean"]["MAE"]
        rmse = cv["mean"]["RMSE"]
        f1_10 = cv["mean"]["F1@10"]
        p_10 = cv["mean"]["Precision@10"]
        r_10 = cv["mean"]["Recall@10"]
        fcp = cv["mean"]["FCP"]
        map_10 = cv["mean"]["MAP@10"]
        mrr_10 = cv["mean"]["MRR@10"]

        if base_mae is None:
            base_mae = mae
            delta_mae = 0.0
        else:
            delta_mae = base_mae - mae

        print(f"  {name:50s} | MAE: {mae:.4f} (Δ={delta_mae:+.4f}) | RMSE: {rmse:.4f} | F1@10: {f1_10:.4f} | Time: {time.time() - t0:.2f}s")
        records.append({
            "Ablation Step": name,
            "MAE": mae,
            "ΔMAE": delta_mae,
            "RMSE": rmse,
            "Precision@10": p_10,
            "Recall@10": r_10,
            "F1@10": f1_10,
            "FCP": fcp,
            "MAP@10": map_10,
            "MRR@10": mrr_10
        })

    df_ablation = pd.DataFrame(records)
    df_ablation.to_csv("results/ablation_study.csv", index=False)
    print("Ablation study saved to results/ablation_study.csv")
    return df_ablation


def run_full_benchmark() -> Dict[str, Any]:
    """Runs full 5-fold CV evaluation on ML-100K and FilmTrust for Grand Synthesis."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    with open("results/baseline_nas_cv.json") as f:
        baseline_data = json.load(f)

    cfg_synth = build_grand_synthesis_config()

    print("\n" + "=" * 75)
    print("EXECUTING GRAND SYNTHESIS 5-FOLD CV BENCHMARK")
    print("=" * 75)

    # 1. ML-100K
    print("\n[EVALUATION] Dataset: ML-100K (5-fold CV)")
    t0 = time.time()
    cv_ml = run_5fold_cv_ml100k(cfg=cfg_synth, K_neighbors=20, k_values=K_EVAL_LIST)
    stat_ml = compute_statistical_acceptance(baseline_data["ml-100k"], cv_ml)
    print(f"ML-100K Benchmark completed in {time.time() - t0:.2f}s")
    print(f"  Baseline MAE: {stat_ml['baseline_mae']:.4f} -> Grand Synthesis MAE: {stat_ml['candidate_mae']:.4f} (Reduction: {stat_ml['mae_reduction']:+.4f})")
    print(f"  Paired t-test t-statistic: {stat_ml['t_statistic']:.4f}, p-value: {stat_ml['p_value']:.4e}")
    print(f"  Baseline F1@10: {stat_ml['baseline_f1_10']:.4f} -> Grand Synthesis F1@10: {stat_ml['candidate_f1_10']:.4f} (Diff: {stat_ml['f1_10_difference']:+.4f})")
    print(f"  Eligible for Merge: {stat_ml['eligible_for_merge']}")

    # 2. FilmTrust
    print("\n[EVALUATION] Dataset: FilmTrust (5-fold CV)")
    t1 = time.time()
    cv_ft = run_5fold_cv_filmtrust(cfg=cfg_synth, K_neighbors=20, k_values=K_EVAL_LIST)
    stat_ft = compute_statistical_acceptance(baseline_data["filmtrust"], cv_ft)
    print(f"FilmTrust Benchmark completed in {time.time() - t1:.2f}s")
    print(f"  Baseline MAE: {stat_ft['baseline_mae']:.4f} -> Grand Synthesis MAE: {stat_ft['candidate_mae']:.4f} (Reduction: {stat_ft['mae_reduction']:+.4f})")
    print(f"  Paired t-test t-statistic: {stat_ft['t_statistic']:.4f}, p-value: {stat_ft['p_value']:.4e}")
    print(f"  Baseline F1@10: {stat_ft['baseline_f1_10']:.4f} -> Grand Synthesis F1@10: {stat_ft['candidate_f1_10']:.4f} (Diff: {stat_ft['f1_10_difference']:+.4f})")
    print(f"  Eligible for Merge: {stat_ft['eligible_for_merge']}")

    results = {
        "model": "NAS-v3-Grand-Synthesis",
        "config": cfg_synth.__dict__,
        "ml-100k": {
            "cv": cv_ml,
            "statistical_acceptance": stat_ml
        },
        "filmtrust": {
            "cv": cv_ft,
            "statistical_acceptance": stat_ft
        },
        "both_eligible_for_merge": bool(stat_ml["eligible_for_merge"] and stat_ft["eligible_for_merge"])
    }

    with open("results/grand_synthesis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed CSV with metrics for all K
    csv_rows = []
    for d_name, d_cv, d_base in [("ml-100k", cv_ml, baseline_data["ml-100k"]), ("filmtrust", cv_ft, baseline_data["filmtrust"])]:
        # MAE, RMSE, FCP
        for m_name in ["MAE", "RMSE", "FCP"]:
            csv_rows.append({
                "dataset": d_name,
                "model": "Baseline-NAS",
                "K": "-",
                "metric": m_name,
                "mean": d_base["mean"][m_name],
                "std": d_base["std"][m_name]
            })
            csv_rows.append({
                "dataset": d_name,
                "model": "NAS-v3-Grand-Synthesis",
                "K": "-",
                "metric": m_name,
                "mean": d_cv["mean"][m_name],
                "std": d_cv["std"][m_name]
            })

        for k in K_EVAL_LIST:
            for m_base in ["Precision", "Recall", "F1", "FCP", "MAP", "MRR"]:
                full_m = f"{m_base}@{k}"
                csv_rows.append({
                    "dataset": d_name,
                    "model": "Baseline-NAS",
                    "K": k,
                    "metric": m_base,
                    "mean": d_base["mean"].get(full_m, 0.0),
                    "std": d_base["std"].get(full_m, 0.0)
                })
                csv_rows.append({
                    "dataset": d_name,
                    "model": "NAS-v3-Grand-Synthesis",
                    "K": k,
                    "metric": m_base,
                    "mean": d_cv["mean"].get(full_m, 0.0),
                    "std": d_cv["std"].get(full_m, 0.0)
                })

    pd.DataFrame(csv_rows).to_csv("results/grand_synthesis_metrics.csv", index=False)
    print("Saved comprehensive CSV to results/grand_synthesis_metrics.csv")

    # Generate Visualization Figures
    generate_figures(results)
    return results


def generate_figures(results: Dict[str, Any]):
    """Generates benchmark plots for publication/report."""
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Metric Curves across K on ML-100K
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("NAS v3 Grand Synthesis vs Baseline NAS across K in {5, 10, 20, 30, 40, 50, 60} (ML-100K)", fontsize=16, fontweight="bold")

    k_vals = list(K_EVAL_LIST)
    metrics_to_plot = ["Precision", "Recall", "F1", "MAP", "MRR", "FCP"]

    ml_cv = results["ml-100k"]["cv"]["mean"]
    with open("results/baseline_nas_cv.json") as f:
        base_data = json.load(f)["ml-100k"]["mean"]

    for idx, m_name in enumerate(metrics_to_plot):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]

        base_vals = [base_data.get(f"{m_name}@{k}", 0.0) for k in k_vals]
        synth_vals = [ml_cv.get(f"{m_name}@{k}", 0.0) for k in k_vals]

        ax.plot(k_vals, base_vals, "o--", color="#d95f02", label="Baseline NAS", linewidth=2.2, markersize=7)
        ax.plot(k_vals, synth_vals, "s-", color="#1b9e77", label="NAS v3 Grand Synthesis", linewidth=2.5, markersize=8)

        ax.set_title(f"{m_name}@K", fontsize=13, fontweight="bold")
        ax.set_xlabel("Neighborhood / Recommendation Cutoff (K)", fontsize=11)
        ax.set_ylabel(m_name, fontsize=11)
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig_path = "figures/grand_synthesis_metrics.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated metric curves figure: {fig_path}")

    # 2. MAE & RMSE Reduction Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Error Reduction: Baseline NAS vs NAS v3 Grand Synthesis (5-Fold CV)", fontsize=15, fontweight="bold")

    with open("results/baseline_nas_cv.json") as f:
        all_base = json.load(f)

    labels = ["ML-100K", "FilmTrust"]
    x = np.arange(len(labels))
    w = 0.35

    # MAE
    base_maes = [all_base["ml-100k"]["mean"]["MAE"], all_base["filmtrust"]["mean"]["MAE"]]
    synth_maes = [results["ml-100k"]["cv"]["mean"]["MAE"], results["filmtrust"]["cv"]["mean"]["MAE"]]

    ax1.bar(x - w/2, base_maes, w, label="Baseline NAS", color="#e7298a", alpha=0.85)
    ax1.bar(x + w/2, synth_maes, w, label="NAS v3 Synthesis", color="#7570b3", alpha=0.85)
    ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax1.set_title("MAE (Lower is Better)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.legend(fontsize=11)
    for i in range(len(labels)):
        ax1.text(x[i] - w/2, base_maes[i] + 0.01, f"{base_maes[i]:.4f}", ha="center", fontsize=10)
        ax1.text(x[i] + w/2, synth_maes[i] + 0.01, f"{synth_maes[i]:.4f}", ha="center", fontsize=10, fontweight="bold")

    # RMSE
    base_rmses = [all_base["ml-100k"]["mean"]["RMSE"], all_base["filmtrust"]["mean"]["RMSE"]]
    synth_rmses = [results["ml-100k"]["cv"]["mean"]["RMSE"], results["filmtrust"]["cv"]["mean"]["RMSE"]]

    ax2.bar(x - w/2, base_rmses, w, label="Baseline NAS", color="#e7298a", alpha=0.85)
    ax2.bar(x + w/2, synth_rmses, w, label="NAS v3 Synthesis", color="#7570b3", alpha=0.85)
    ax2.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=12)
    ax2.set_title("RMSE (Lower is Better)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.legend(fontsize=11)
    for i in range(len(labels)):
        ax2.text(x[i] - w/2, base_rmses[i] + 0.01, f"{base_rmses[i]:.4f}", ha="center", fontsize=10)
        ax2.text(x[i] + w/2, synth_rmses[i] + 0.01, f"{synth_rmses[i]:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig2_path = "figures/grand_synthesis_error_reduction.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated error reduction figure: {fig2_path}")


def main():
    # 1. Run sequential ablation
    run_sequential_ablation()

    # 2. Run full benchmark
    run_full_benchmark()


if __name__ == "__main__":
    main()
